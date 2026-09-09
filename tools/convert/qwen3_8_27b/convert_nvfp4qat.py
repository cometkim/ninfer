"""Build the QAT-sourced Qwen3.8-27B NVFP4 artifact from its three source roles.

Canonical invocation::

    python3 -m tools.convert.qwen3_8_27b.convert_nvfp4qat \
      --model /path/to/Qwen3.8-27B \
      --quantized-model /path/to/Qwen3.8-27B-QUASAR-NVFP4 \
      --dflash2-model /path/to/Qwen3.8-27B-DFlash2 \
      --out out/qwen3_8_27b_nvfp4qat.ninfer
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
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
from . import dflash2_recipe
from . import inventory_nvfp4qat as inventory
from . import recipe_nvfp4qat as recipe

DFLASH2_OBJECT_NAMES = frozenset(
    spec.name for spec in inventory.DFLASH2_TENSOR_SPECS
)


RECIPE_ID = "qwen3_8_27b_nvfp4qat-v2"
OUTPUT_BASENAME = "qwen3_8_27b_nvfp4qat.ninfer"


@dataclass(frozen=True, slots=True)
class ConversionPreflight:
    dflash2_dir: Path
    dflash2_summary: dict[str, object]
    dflash2_source: object
    base_dir: Path
    quantized_dir: Path
    config_summary: dict[str, object]
    base_source: object
    quantized_dtype_counts: dict[str, int]
    plain_tensor_identity: dict[str, object]
    resources: tuple[family_conversion.ResourcePayload, ...]
    draft: draft_head.DraftHeadContext
    object_plan: family_conversion.ObjectPlan


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


def _base_tensor_name(name: str) -> str:
    # The QAT export spells the GDN convolution ``convNd``; the official source
    # names it ``conv1d``. Every other unquantized tensor keeps its base name.
    return name.replace("linear_attn.convNd.", "linear_attn.conv1d.")


def _byte_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Exact byte comparison that is robust to NaN payloads."""

    if left.dtype == torch.bfloat16:
        return torch.equal(left.view(torch.uint8), right.view(torch.uint8))
    return torch.equal(left, right)


def _validate_plain_tensor_identity(
    quantized_reader: ShardReader, base_reader: ShardReader
) -> dict[str, object]:
    """Every unquantized QAT tensor must be bit-identical to the official base.

    This is the provenance boundary of the profile: it proves the QAT
    checkpoint differs from the official source only in its 496 quantized text
    linears, so routing every direct tensor through the base loses nothing.
    """

    from safetensors import safe_open

    suffixes = (
        ".weight_packed",
        ".weight_scale",
        ".weight_global_scale",
        ".input_global_scale",
    )
    plain = sorted(
        name
        for name in quantized_reader.names
        if not name.endswith(suffixes)
        and not name.startswith("__")
    )
    missing = [
        name for name in plain if _base_tensor_name(name) not in base_reader.names
    ]
    if missing:
        raise ValueError(f"QAT plain tensor {missing[0]} has no official counterpart")
    by_shard: dict[str, list[str]] = {}
    for name in plain:
        by_shard.setdefault(quantized_reader.weight_map[name], []).append(name)
    checked = 0
    for shard in sorted(by_shard):
        with safe_open(
            str(quantized_reader.model_dir / shard), framework="pt", device="cpu"
        ) as handle:
            for name in by_shard[shard]:
                quantized_tensor = handle.get_tensor(name)
                base_tensor = base_reader.get(_base_tensor_name(name))
                if quantized_tensor.dtype != base_tensor.dtype or not _byte_equal(
                    quantized_tensor, base_tensor
                ):
                    raise ValueError(
                        f"{name}: unquantized QAT tensor differs from the official source"
                    )
                checked += 1
                del quantized_tensor, base_tensor
    return {
        "checked_tensors": checked,
        "note": (
            "every unquantized QAT tensor is byte-identical to the official "
            "BF16 source (linear_attn.convNd maps to linear_attn.conv1d)"
        ),
    }


def preflight_inventory() -> None:
    inventory.validate_inventory()
    recipe.validate_recipe()
    dflash2_recipe.validate_recipe_coverage()


def build_object_plan(
    resources: Mapping[str, bytes],
) -> family_conversion.ObjectPlan:
    preflight_inventory()
    return family_conversion.build_object_plan(inventory.OBJECT_SPECS, resources)


def preflight_conversion(
    base_dir: str | Path,
    quantized_dir: str | Path,
    dflash2_dir: str | Path,
) -> ConversionPreflight:
    base = Path(base_dir)
    quantized = Path(quantized_dir)
    dflash2 = Path(dflash2_dir)
    from tools.convert.qwen3_6.common import recipe as family_recipe

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
        raise ValueError("official and QAT source model configs do not match")
    quantization_config = family_conversion.load_json(quantized / "config.json").get(
        "quantization_config"
    )
    if not isinstance(quantization_config, Mapping):
        raise ValueError("QAT source must declare quantization_config")
    if quantization_config.get("format") != "nvfp4-pack-quantized":
        raise ValueError("QAT source is not an nvfp4-pack-quantized checkpoint")
    if quantization_config.get("quant_method") != "compressed-tensors":
        raise ValueError("QAT source is not a compressed-tensors checkpoint")
    dflash2_summary = dflash2_recipe.validate_config(
        family_conversion.load_json(dflash2 / "config.json")
    )
    dflash2_recipe.validate_base_compatibility(base_summary, dflash2_summary)
    dflash2_source = dflash2_recipe.preflight_sources(dflash2)
    preflight_inventory()

    with ShardReader(quantized) as quantized_reader, ShardReader(base) as base_reader:
        quantized_dtype_counts = recipe.preflight_quantized_metadata(quantized_reader)
        plain_tensor_identity = _validate_plain_tensor_identity(
            quantized_reader, base_reader
        )
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
        config_summary=base_summary,
        base_source=base_source,
        quantized_dtype_counts=quantized_dtype_counts,
        plain_tensor_identity=plain_tensor_identity,
        resources=resources,
        draft=draft,
        object_plan=object_plan,
    )


def _encode_source_nvfp4_weight(
    spec: inventory.TensorSpec, reader: ShardReader
) -> bytes:
    selected = recipe.SOURCE_NVFP4_WEIGHTS_BY_NAME[spec.name]
    packed, scales, divisor = recipe.materialize_source_nvfp4_weight(
        selected, reader
    )
    return encode_nvfp4(packed, scales, divisor, spec.shape)


def _family_materialize(tensor_recipe, reader: ShardReader, derived=None) -> torch.Tensor:
    from tools.convert.qwen3_6.common import recipe as family_recipe

    return family_recipe.materialize_recipe(tensor_recipe, reader, derived)


def _materialize_official(
    spec: inventory.TensorSpec,
    reader: ShardReader,
    derived: Mapping[str, torch.Tensor],
    device: torch.device,
) -> bytes:
    tensor = _family_materialize(
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
            "method": "QUASAR loss-aware NVFP4 QAT distillation (arXiv 2608.13966)",
            "plain_tensor_identity": preflight.plain_tensor_identity,
        },
        "ranking_path": str(ranking.resolve()),
    }
    report["dflash2"] = {
        "summary": preflight.dflash2_summary,
        "repository": dflash2_recipe.REPOSITORY,
        "revision": dflash2_recipe.REVISION,
        "model_path": str(preflight.dflash2_dir.resolve()),
        "note": "upstream persistent suffix contract (W8G32_F16S module)",
    }
    return report


def convert(
    base_dir: str | Path,
    quantized_dir: str | Path,
    dflash2_dir: str | Path,
    out_path: str | Path,
    *,
    device: str | torch.device = "cuda",
) -> Path:
    """Run the closed three-source conversion and return its report path."""

    started = time.perf_counter()
    output = Path(out_path)
    if output.name != OUTPUT_BASENAME:
        raise ValueError(
            f"nvfp4qat converter output basename must be {OUTPUT_BASENAME!r}"
        )
    requested_device = str(device)
    resolved_device = pick_device(device)
    preflight = preflight_conversion(base_dir, quantized_dir, dflash2_dir)

    print(
        f"preflight complete: {len(preflight.object_plan.objects)} objects, "
        f"{len(recipe.SOURCE_NVFP4_WEIGHT_RECIPES)} QAT NVFP4 parents, "
        f"{len(recipe.DECODED_CONTROL_RECIPES)} decoded control parents, "
        f"device={resolved_device}",
        flush=True,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    resources = {resource.name: resource.data for resource in preflight.resources}
    draft_ids = draft_head.materialize_draft_head_token_ids(preflight.draft)
    derived = {draft_head.DRAFT_HEAD_TOKEN_IDS_OBJECT: draft_ids}
    from contextlib import ExitStack

    with ExitStack() as sources:
        base_reader = sources.enter_context(ShardReader(preflight.base_dir))
        quantized_reader = sources.enter_context(ShardReader(preflight.quantized_dir))
        dflash2_reader = sources.enter_context(
            ShardReader.from_file(preflight.dflash2_dir / "model.safetensors")
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
                elif spec.name in recipe.SOURCE_INPUT_DIVISORS_BY_NAME:
                    scalar = recipe.materialize_source_input_divisor(
                        recipe.SOURCE_INPUT_DIVISORS_BY_NAME[spec.name],
                        quantized_reader,
                    )
                    payload = encode_direct(scalar, inventory.FP32)
                elif spec.name in recipe.DECODED_CONTROLS_BY_NAME:
                    tensor = recipe.materialize_decoded_control(
                        recipe.DECODED_CONTROLS_BY_NAME[spec.name],
                        quantized_reader,
                        resolved_device,
                    )
                    payload = encode_direct(tensor, inventory.BF16)
                    del tensor
                elif spec.name in recipe.BASE_DIRECT_BY_NAME:
                    tensor = _family_materialize(
                        recipe.BASE_DIRECT_BY_NAME[spec.name], base_reader
                    )
                    if tuple(tensor.shape) != spec.shape:
                        raise ValueError(
                            f"{spec.name}: materialized shape "
                            f"{tuple(tensor.shape)} != {spec.shape}"
                        )
                    payload = encode_direct(tensor, spec.format)
                    del tensor
                elif spec.name in DFLASH2_OBJECT_NAMES:
                    tensor = dflash2_recipe.materialize_tensor(
                        spec.name, dflash2_reader
                    )
                    if tuple(tensor.shape) != spec.shape:
                        raise ValueError(
                            f"{spec.name}: materialized shape "
                            f"{tuple(tensor.shape)} != {spec.shape}"
                        )
                    payload = family_conversion.encode_tensor_payload(
                        tensor, spec, resolved_device
                    )
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

    elapsed = time.perf_counter() - started
    final_bytes = output.stat().st_size
    arguments = {
        "model": str(base_dir),
        "quantized_model": str(quantized_dir),
        "dflash2_model": str(dflash2_dir),
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
    parser.add_argument("--dflash2-model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args(argv)
    convert(
        arguments.model,
        arguments.quantized_model,
        arguments.dflash2_model,
        arguments.out,
        device=arguments.device,
    )


if __name__ == "__main__":
    main()
