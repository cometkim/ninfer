"""Build the fuller Qwen3.8-27B NVFP4 artifact from its three source roles.

Canonical invocation::

    python3 -m tools.convert.qwen3_8_27b.convert_nvfp4full \
      --model /path/to/Qwen3.8-27B \
      --quantized-model /path/to/Qwen3.8-27B-NVFP4 \
      --calibration out/qwen3_8_27b_nvfp4full_calibration.json \
      --out out/qwen3_8_27b_nvfp4full.ninfer
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import struct
import time
from typing import Mapping, Sequence

import torch

from tools.artifact.container import (
    ArtifactIdentity,
    ArtifactObject,
    ArtifactWriter,
)
from tools.artifact.layouts import encode_direct, encode_nvfp4
from tools.convert.common.quantize import pick_device
from tools.convert.common.safetensors import ShardReader
from tools.convert.qwen3_6.common import conversion as family_conversion
from tools.convert.qwen3_6_27b import convert as family_config
from tools.convert.qwen3_6_27b import draft_head

from . import convert as base_convert
from . import inventory_nvfp4full as inventory
from . import nvfp4_encode
from . import recipe_nvfp4full as recipe


RECIPE_ID = "qwen3_8_27b_nvfp4full-v2"
OUTPUT_BASENAME = "qwen3_8_27b_nvfp4full.ninfer"


@dataclass(frozen=True, slots=True)
class ConversionPreflight:
    dflash2_dir: Path | None
    dflash2_summary: dict[str, object] | None
    dflash2_source: object | None
    base_dir: Path
    quantized_dir: Path
    calibration_path: Path
    config_summary: dict[str, object]
    base_source: object
    quantized_dtype_counts: dict[str, int]
    resources: tuple[family_conversion.ResourcePayload, ...]
    draft: draft_head.DraftHeadContext
    object_plan: family_conversion.ObjectPlan
    divisor_table: recipe.LocalDivisorTable


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _validate_index(model_dir: Path) -> None:
    index_path = model_dir / "model.safetensors.index.json"
    value = family_conversion.load_json(index_path)
    weight_map = value.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"{index_path}: weight_map must be a nonempty object")
    referenced = set(weight_map.values())
    actual = {path.name for path in model_dir.glob("*.safetensors")}
    if actual != referenced:
        raise ValueError(f"{model_dir}: safetensors shard set does not match the index")
    for shard in sorted(referenced):
        path = model_dir / shard
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"{path}: indexed shard is missing or empty")


def preflight_inventory() -> None:
    inventory.validate_inventory()
    recipe.validate_recipe()
    nvfp4_encode.self_test()


def build_object_plan(
    resources: Mapping[str, bytes],
) -> family_conversion.ObjectPlan:
    preflight_inventory()
    return family_conversion.build_object_plan(inventory.OBJECT_SPECS, resources)


DFLASH2_REQUIRED = True

_DFLASH2_FLAT_MEMBERS = {
    "architectures": ["DFlash2DraftModel"],
    "hidden_size": 5120,
    "intermediate_size": 17408,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "num_hidden_layers": 5,
    "num_target_layers": 64,
    "sliding_window": 2048,
    "max_position_embeddings": 262144,
    "is_causal": False,
    "vocab_size": 248320,
    "rms_norm_eps": 1e-06,
    "layer_types": ["sliding_attention"] * 5,
}

_DFLASH2_NESTED_MEMBERS = {
    "dflash_config": {
        "block_size": 8,
        "conv_kernel_size": 2,
        "conv_group_size": 16,
        "selector_rank": 256,
        "selector_top_k": 16,
        "mask_token_id": 248070,
        "target_layer_ids": [5, 19, 33, 47, 61],
    },
    "rope_parameters": {"rope_theta": 10000000.0, "rope_type": "default"},
}


def validate_dflash2_config(config: Mapping[str, object]) -> dict[str, object]:
    summary: dict[str, object] = {}
    for member, expected in _DFLASH2_FLAT_MEMBERS.items():
        actual = config.get(member)
        if actual != expected:
            raise ValueError(
                f"DFlash2 config.{member} = {actual!r} != registered {expected!r}"
            )
        summary[member] = actual
    for section, members in _DFLASH2_NESTED_MEMBERS.items():
        nested = config.get(section)
        if not isinstance(nested, Mapping):
            raise ValueError(f"DFlash2 config.{section} must be a mapping")
        for member, expected in members.items():
            actual = nested.get(member)
            if actual != expected:
                raise ValueError(
                    f"DFlash2 config.{section}.{member} = {actual!r} "
                    f"!= registered {expected!r}"
                )
            summary[f"{section}.{member}"] = actual
    return summary


def preflight_conversion(
    base_dir: str | Path,
    quantized_dir: str | Path,
    calibration_path: str | Path,
    dflash2_dir: str | Path | None = None,
) -> ConversionPreflight:
    base = Path(base_dir)
    quantized = Path(quantized_dir)
    calibration = Path(calibration_path)
    dflash2 = Path(dflash2_dir) if dflash2_dir is not None else None
    dflash2_summary: dict[str, object] | None = None
    dflash2_source = None
    if dflash2 is not None:
        dflash2_summary = validate_dflash2_config(
            family_conversion.load_json(dflash2 / "config.json")
        )
        from tools.convert.qwen3_6.common import recipe as family_recipe

        with ShardReader.from_file(dflash2 / "model.safetensors") as dflash2_reader:
            dflash2_source = family_recipe.preflight_source_reader(
                dflash2_reader, recipe.DFLASH2_RECIPES
            )
    elif DFLASH2_REQUIRED:
        raise ValueError("this recipe requires --dflash2-model (the module is part "
                         "of the complete product image)")
    _validate_index(base)
    _validate_index(quantized)

    base_config = family_conversion.load_json(base / "config.json")
    if base_config.get("quantization_config") is not None:
        raise ValueError("official source must not declare quantization_config")
    base_summary = family_config.validate_config(base_config)
    quantized_summary = family_config.validate_config(
        family_conversion.load_json(quantized / "config.json")
    )
    if base_summary != quantized_summary:
        raise ValueError("official and quantized source model configs do not match")
    preflight_inventory()

    with ShardReader(quantized) as quantized_reader:
        quantized_dtype_counts = recipe.preflight_quantized_metadata(
            quantized_reader
        )
    divisor_table = recipe.LocalDivisorTable(calibration)
    from tools.convert.qwen3_6.common import recipe as family_recipe

    with ShardReader(base) as base_reader:
        base_source = family_recipe.preflight_source_reader(
            base_reader,
            (
                *recipe.BASE_DIRECT_RECIPES,
                *recipe.OFFICIAL_RECIPES_BY_NAME.values(),
            ),
        )

    resources = base_convert.load_resources(base)
    resource_map = {resource.name: resource.data for resource in resources}
    object_plan = build_object_plan(resource_map)
    ranking = _repo_root() / draft_head.DEFAULT_RANKING
    draft = draft_head.compute_shortlist(ranking, base)
    return ConversionPreflight(
        dflash2_dir=dflash2,
        dflash2_summary=dflash2_summary,
        dflash2_source=dflash2_source,
        base_dir=base,
        quantized_dir=quantized,
        calibration_path=calibration,
        config_summary=base_summary,
        base_source=base_source,
        quantized_dtype_counts=quantized_dtype_counts,
        resources=resources,
        draft=draft,
        object_plan=object_plan,
        divisor_table=divisor_table,
    )


def _encode_source_nvfp4_weight(
    spec: inventory.TensorSpec, reader: ShardReader
) -> bytes:
    selected = recipe.SOURCE_NVFP4_WEIGHTS_BY_NAME[spec.name]
    packed, scales, divisor = recipe.materialize_source_nvfp4_weight(
        selected, reader
    )
    return encode_nvfp4(packed, scales, divisor, spec.shape)


def _encode_local_nvfp4_weight(
    spec: inventory.TensorSpec, reader: ShardReader, device: torch.device
) -> tuple[bytes, float, float]:
    selected = recipe.LOCAL_NVFP4_WEIGHTS_BY_NAME[spec.name]
    parent = recipe.materialize_bf16_parent(selected.parts, reader)
    if tuple(parent.shape) != spec.shape or parent.dtype != torch.bfloat16:
        raise ValueError(f"{spec.name}: materialized parent signature mismatch")
    words = nvfp4_encode.quantize_nvfp4(parent, device=device)
    payload = encode_nvfp4(
        words.packed_codes.cpu(),
        words.natural_scales.cpu(),
        struct.pack("<f", words.weight_divisor),
        spec.shape,
    )
    error = nvfp4_encode.relative_frobenius_error(
        parent, nvfp4_encode.dequantize_nvfp4(words)
    )
    del parent
    return payload, float(words.weight_divisor), error


def _encode_bf16_exception(
    spec: inventory.TensorSpec, reader: ShardReader
) -> bytes:
    selected = recipe.BF16_WEIGHTS_BY_NAME[spec.name]
    parent = recipe.materialize_bf16_parent(selected.parts, reader)
    if tuple(parent.shape) != spec.shape or parent.dtype != torch.bfloat16:
        raise ValueError(f"{spec.name}: materialized BF16 parent signature mismatch")
    return encode_direct(parent, inventory.BF16)


def _materialize_base_direct(spec: inventory.TensorSpec, reader: ShardReader) -> bytes:
    tensor = family_recipe_materialize(
        recipe.BASE_DIRECT_BY_NAME[spec.name], reader
    )
    if tuple(tensor.shape) != spec.shape:
        raise ValueError(
            f"{spec.name}: materialized shape {tuple(tensor.shape)} != {spec.shape}"
        )
    return encode_direct(tensor, spec.format)


def family_recipe_materialize(
    tensor_recipe, reader: ShardReader, derived=None
) -> torch.Tensor:
    from tools.convert.qwen3_6.common import recipe as family_recipe

    return family_recipe.materialize_recipe(tensor_recipe, reader, derived)


def _materialize_official(
    spec: inventory.TensorSpec,
    reader: ShardReader,
    derived: Mapping[str, torch.Tensor],
    device: torch.device,
) -> bytes:
    tensor = family_recipe_materialize(
        recipe.OFFICIAL_RECIPES_BY_NAME[spec.name], reader, dict(derived)
    )
    if tuple(tensor.shape) != spec.shape:
        raise ValueError(
            f"{spec.name}: materialized shape {tuple(tensor.shape)} != {spec.shape}"
        )
    payload = family_conversion.encode_tensor_payload(tensor, spec, device)
    del tensor
    return payload


def _build_report(
    *,
    preflight: ConversionPreflight,
    output: Path,
    arguments: Mapping[str, object],
    objects: Sequence[ArtifactObject],
    elapsed_seconds: float,
    final_bytes: int,
    device: torch.device,
    quantization: Mapping[str, object],
) -> dict[str, object]:
    ranking = _repo_root() / draft_head.DEFAULT_RANKING
    report = family_conversion.build_conversion_report(
        identity=ArtifactIdentity(inventory.MODEL_ID, inventory.WEIGHTS_ID),
        target_key=inventory.TARGET_KEY,
        recipe_id=RECIPE_ID,
        repo_root=_repo_root(),
        model_dir=preflight.base_dir,
        out_path=output,
        arguments=arguments,
        config_summary=preflight.config_summary,
        source_preflight=preflight.base_source,
        objects=objects,
        elapsed_seconds=elapsed_seconds,
        final_bytes=final_bytes,
        device=device,
        ranking_path=ranking,
    )
    report["source"] = {
        "base": {
            "repository": recipe.BASE_REPOSITORY,
            "revision": recipe.BASE_REVISION,
            "model_path": str(preflight.base_dir.resolve()),
        },
        "quantized": {
            "repository": recipe.QUANTIZED_REPOSITORY,
            "revision": recipe.QUANTIZED_REVISION,
            "model_path": str(preflight.quantized_dir.resolve()),
            "note": (
                "document-pinned revision 60e813d4dbbdc5d64cf3f5a8caf2897bedf03679 "
                "is no longer reachable upstream; this reachable main revision "
                "carries the structurally validated mixed-precision allocation"
            ),
        },
        "calibration": {
            "path": str(preflight.calibration_path.resolve()),
            **preflight.divisor_table.provenance,
        },
        "ranking_path": str(ranking.resolve()),
    }
    report["local_nvfp4"] = quantization
    return report


def convert(
    base_dir: str | Path,
    quantized_dir: str | Path,
    calibration_path: str | Path,
    out_path: str | Path,
    *,
    device: str | torch.device = "cuda",
    dflash2_dir: str | Path | None = None,
) -> Path:
    """Run the closed three-source conversion and return its report path.

    The DFlash2 drafter module is a required member of the complete product
    image; `dflash2_dir` points at the incoai BF16 drafter checkpoint and is
    mandatory for this recipe.
    """

    started = time.perf_counter()
    output = Path(out_path)
    if output.name != OUTPUT_BASENAME:
        raise ValueError(
            f"nvfp4full converter output basename must be {OUTPUT_BASENAME!r}"
        )
    requested_device = str(device)
    resolved_device = pick_device(device)
    preflight = preflight_conversion(base_dir, quantized_dir, calibration_path, dflash2_dir)

    print(
        f"preflight complete: {len(preflight.object_plan.objects)} objects, "
        f"{len(recipe.SOURCE_NVFP4_WEIGHT_RECIPES)} source and "
        f"{len(recipe.LOCAL_NVFP4_WEIGHT_RECIPES)} local NVFP4 parents, "
        f"device={resolved_device}",
        flush=True,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    resources = {resource.name: resource.data for resource in preflight.resources}
    draft_ids = draft_head.materialize_draft_head_token_ids(preflight.draft)
    derived = {draft_head.DRAFT_HEAD_TOKEN_IDS_OBJECT: draft_ids}
    quantization_report: dict[str, object] = {
        "encoder_profile": recipe.LOCAL_ENCODER_PROFILE,
        "parents": {},
    }
    from contextlib import ExitStack

    with ExitStack() as sources:
        base_reader = sources.enter_context(ShardReader(preflight.base_dir))
        quantized_reader = sources.enter_context(ShardReader(preflight.quantized_dir))
        dflash2_reader = (
            sources.enter_context(
                ShardReader.from_file(preflight.dflash2_dir / "model.safetensors")
            )
            if preflight.dflash2_dir is not None
            else None
        )
        with ArtifactWriter(
            output,
            ArtifactIdentity(inventory.MODEL_ID, inventory.WEIGHTS_ID),
            preflight.object_plan.specs,
        ) as writer:
            if writer.objects != preflight.object_plan.objects:
                raise RuntimeError(
                    "writer object plan differs from completed preflight"
                )
            for index, spec in enumerate(inventory.OBJECT_SPECS, start=1):
                if isinstance(spec, inventory.ResourceSpec):
                    payload = resources[spec.name]
                elif spec.name in recipe.SOURCE_NVFP4_WEIGHTS_BY_NAME:
                    payload = _encode_source_nvfp4_weight(spec, quantized_reader)
                elif spec.name in recipe.LOCAL_NVFP4_WEIGHTS_BY_NAME:
                    payload, divisor, error = _encode_local_nvfp4_weight(
                        spec, base_reader, resolved_device
                    )
                    quantization_report["parents"][spec.name] = {
                        "weight_scale_divisor": divisor,
                        "relative_frobenius_error": error,
                    }
                elif spec.name in recipe.SOURCE_INPUT_DIVISORS_BY_NAME:
                    scalar = recipe.materialize_source_input_divisor(
                        recipe.SOURCE_INPUT_DIVISORS_BY_NAME[spec.name],
                        quantized_reader,
                    )
                    payload = encode_direct(scalar, inventory.FP32)
                elif spec.name in recipe.LOCAL_INPUT_DIVISORS_BY_NAME:
                    scalar = preflight.divisor_table.value(spec.name)
                    payload = encode_direct(scalar, inventory.FP32)
                elif spec.name in recipe.BF16_WEIGHTS_BY_NAME:
                    payload = _encode_bf16_exception(spec, base_reader)
                elif spec.name in recipe.BASE_DIRECT_BY_NAME:
                    payload = _materialize_base_direct(spec, base_reader)
                elif spec.name in recipe.DFLASH2_RECIPES_BY_NAME:
                    if dflash2_reader is None:
                        raise RuntimeError("dflash2 spec reached without a source reader")
                    tensor = family_recipe_materialize(
                        recipe.DFLASH2_RECIPES_BY_NAME[spec.name], dflash2_reader
                    )
                    if spec.format == inventory.NVFP4:
                        # Weight-only drafter parents: the fork's NVFP4 encoder,
                        # no activation calibration (the drafter runs A16).
                        payload = nvfp4_encode.encode_nvfp4_parent(
                            tensor, device=resolved_device
                        )
                        quantization_report["parents"][spec.name] = {
                            "weight_scale_divisor": None,
                            "relative_frobenius_error": (
                                nvfp4_encode.relative_frobenius_error(
                                    tensor,
                                    nvfp4_encode.dequantize_nvfp4(
                                        nvfp4_encode.quantize_nvfp4(
                                            tensor, device=resolved_device
                                        ),
                                        device=resolved_device,
                                    ),
                                )
                            ),
                        }
                    else:
                        payload = encode_direct(tensor, spec.format)
                    del tensor
                else:
                    payload = _materialize_official(
                        spec, base_reader, derived, resolved_device
                    )
                writer.write(spec.name, payload)
                del payload
                if index % 64 == 0 or index == len(inventory.OBJECT_SPECS):
                    print(
                        f"[{index}/{len(inventory.OBJECT_SPECS)}] objects written",
                        flush=True,
                    )

    errors = [
        entry["relative_frobenius_error"]
        for entry in quantization_report["parents"].values()
    ]
    quantization_report["relative_frobenius_error_max"] = max(errors)
    quantization_report["relative_frobenius_error_mean"] = sum(errors) / len(errors)

    elapsed = time.perf_counter() - started
    final_bytes = output.stat().st_size
    arguments = {
        "model": str(base_dir),
        "quantized_model": str(quantized_dir),
        "calibration": str(calibration_path),
        "out": str(out_path),
        "device": requested_device,
    }
    report = _build_report(
        preflight=preflight,
        output=output,
        arguments=arguments,
        objects=preflight.object_plan.objects,
        elapsed_seconds=elapsed,
        final_bytes=final_bytes,
        device=resolved_device,
        quantization=quantization_report,
    )
    report_path = Path(str(output) + ".conversion.json")
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(
        f"complete: {final_bytes} bytes in {elapsed:.1f}s; report={report_path}",
        flush=True,
    )
    return report_path


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--quantized-model", required=True, type=Path)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--dflash2-model", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args(argv)
    convert(
        arguments.model,
        arguments.quantized_model,
        arguments.calibration,
        arguments.out,
        device=arguments.device,
        dflash2_dir=arguments.dflash2_model,
    )


if __name__ == "__main__":
    main()
