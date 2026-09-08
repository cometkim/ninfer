"""Verify the QAT-sourced Qwen3.8-27B NVFP4 artifact against its three source roles."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import struct
from typing import Sequence

import torch

from tools.artifact.container import (
    Artifact,
    ArtifactIdentity,
    ResourceObject,
    TensorObject,
    object_alignment,
)
from tools.artifact.layouts import (
    align_up,
    decode_direct,
    decode_nvfp4_words,
    encoded_size,
)
from tools.artifact.numeric import valid_positive_fp32_word
from tools.convert.common.safetensors import ShardReader

from . import convert_nvfp4qat as convert
from . import inventory_nvfp4qat as inventory
from . import nvfp4_encode
from . import recipe_nvfp4qat as recipe
from tools.convert.qwen3_8_27b import verify_nvfp4full as family_verify
from tools.convert.qwen3_6_27b import verify as base_verify


class VerificationError(ValueError):
    """The artifact does not satisfy the registered nvfp4qat contract."""


@dataclass(frozen=True, slots=True)
class VerificationSummary:
    objects: int
    tensors: int
    resources: int
    w8_endpoint_weights: int
    w8_endpoint_rows: int
    w8_endpoint_groups: int
    source_nvfp4_weights: int
    source_input_divisors: int
    decoded_control_parents: int
    dflash2_tensors: int
    divisor_derivation_ratio_min: float
    divisor_derivation_ratio_median: float
    divisor_derivation_ratio_max: float
    payload_bytes: int


def _error(message: str) -> None:
    raise VerificationError(message)


def validate_structure(artifact: Artifact) -> int:
    expected_identity = ArtifactIdentity(inventory.MODEL_ID, inventory.WEIGHTS_ID)
    if artifact.identity != expected_identity:
        _error(
            f"artifact identity is {artifact.identity!r}, expected "
            f"{expected_identity!r}"
        )
    if len(artifact.objects) != len(inventory.OBJECT_SPECS):
        _error(
            f"artifact has {len(artifact.objects)} objects, "
            f"expected {len(inventory.OBJECT_SPECS)}"
        )
    cursor = 0
    formats: Counter[str] = Counter()
    layouts: Counter[str] = Counter()
    for position, (actual, expected) in enumerate(
        zip(artifact.objects, inventory.OBJECT_SPECS)
    ):
        if actual.name != expected.name:
            _error(
                f"object {position} is {actual.name!r}, expected {expected.name!r}"
            )
        expected_offset = align_up(cursor, object_alignment(actual))
        if actual.offset != expected_offset:
            _error(f"{actual.name}: offset {actual.offset}, expected {expected_offset}")
        if isinstance(expected, inventory.TensorSpec):
            if not isinstance(actual, TensorObject):
                _error(f"{actual.name}: expected tensor descriptor")
            signature = (actual.shape, actual.format, actual.layout)
            registered = (expected.shape, expected.format, expected.layout)
            if signature != registered:
                _error(f"{actual.name}: signature {signature} != {registered}")
            if actual.bytes != encoded_size(actual.layout, actual.format, actual.shape):
                _error(f"{actual.name}: encoded byte count is invalid")
            formats[actual.format] += 1
            layouts[actual.layout] += 1
        else:
            if not isinstance(actual, ResourceObject):
                _error(f"{actual.name}: expected resource descriptor")
            if actual.encoding != expected.encoding:
                _error(f"{actual.name}: resource encoding is invalid")
        cursor = actual.offset + actual.bytes
    if dict(formats) != inventory.FORMAT_COUNTS:
        _error(f"numeric-format counts are {dict(formats)}")
    if dict(layouts) != inventory.LAYOUT_COUNTS:
        _error(f"layout counts are {dict(layouts)}")
    payload_bytes = artifact.file_bytes - artifact.payload_offset
    if cursor != payload_bytes:
        _error(f"payload ends at {cursor}, file contains {payload_bytes} bytes")
    return payload_bytes


def _verify_resources(artifact: Artifact, base_dir: Path) -> None:
    for spec in inventory.RESOURCE_SPECS:
        obj = artifact.find(spec.name)
        if not isinstance(obj, ResourceObject):
            _error(f"{spec.name}: expected resource")
        source = (base_dir / spec.name.removeprefix("frontend/")).read_bytes()
        if bytes(artifact.payload(obj)) != source:
            _error(f"{spec.name}: resource payload differs from base source")


def _verify_source_nvfp4_weights(
    artifact: Artifact,
    reader: ShardReader,
    device: torch.device,
) -> tuple[int, float, float, float]:
    """Word-for-word comparison against the QAT source plus the divisor
    derivation cross-check ``d_w = 2688 / amax`` on the represented values."""

    ratios: list[float] = []
    for selected in recipe.SOURCE_NVFP4_WEIGHT_RECIPES:
        obj = artifact.find(selected.object_name)
        if not isinstance(obj, TensorObject):
            _error(f"{selected.object_name}: expected tensor")
        packed, scales, divisor = recipe.materialize_source_nvfp4_weight(
            selected, reader
        )
        stored_packed, stored_scales, stored_divisor = decode_nvfp4_words(
            artifact.payload(obj), obj.shape
        )
        if not torch.equal(stored_packed, packed):
            _error(f"{obj.name}: packed E2M1 words differ from source")
        if not torch.equal(stored_scales, scales):
            _error(f"{obj.name}: E4M3FN scale words differ from source")
        if bytes(stored_divisor.reshape(1).view(torch.uint8).numpy()) != divisor:
            _error(f"{obj.name}: weight divisor differs from source")
        reconstructed = nvfp4_encode.dequantize_nvfp4(
            nvfp4_encode.Nvfp4Words(
                packed_codes=stored_packed,
                natural_scales=stored_scales,
                weight_divisor=stored_divisor.item(),
            ),
            device=device,
        )
        amax = reconstructed.abs().amax().item()
        if amax > 0.0:
            implied = struct.unpack("<f", struct.pack("<f", 2688.0 / amax))[0]
            stored = stored_divisor.item()
            ratios.append(implied / stored)
        del reconstructed
    if not ratios:
        _error("no divisor derivation ratios were computed")
    ordered = sorted(ratios)
    return (
        len(recipe.SOURCE_NVFP4_WEIGHT_RECIPES),
        ordered[0],
        ordered[len(ordered) // 2],
        ordered[-1],
    )


def _verify_source_input_divisors(
    artifact: Artifact,
    reader: ShardReader,
) -> None:
    for selected in recipe.SOURCE_INPUT_DIVISOR_RECIPES:
        obj = artifact.find(selected.object_name)
        if not isinstance(obj, TensorObject):
            _error(f"{selected.object_name}: expected tensor")
        expected = recipe.materialize_source_input_divisor(selected, reader)
        stored = decode_direct(artifact.payload(obj), obj.format, obj.shape)
        word = int(stored.view(torch.int32).item()) & 0xFFFFFFFF
        if not valid_positive_fp32_word(word):
            _error(f"{obj.name}: input divisor is not finite and positive")
        if not torch.equal(stored.view(torch.int32), expected.view(torch.int32)):
            _error(f"{obj.name}: input divisor differs from source")


def _verify_decoded_control_parents(
    artifact: Artifact,
    reader: ShardReader,
    device: torch.device,
) -> None:
    for selected in recipe.DECODED_CONTROL_RECIPES:
        obj = artifact.find(selected.object_name)
        if not isinstance(obj, TensorObject):
            _error(f"{selected.object_name}: expected tensor")
        expected = recipe.materialize_decoded_control(selected, reader, device)
        stored = decode_direct(artifact.payload(obj), obj.format, obj.shape)
        if not torch.equal(stored.view(torch.int16), expected.view(torch.int16)):
            _error(f"{obj.name}: BF16 words differ from the QAT decode oracle")
        del expected


def verify_artifact(
    artifact: Artifact,
    base_dir: str | Path,
    quantized_dir: str | Path,
    dflash2_dir: str | Path,
    *,
    device: str | torch.device = "cuda",
) -> VerificationSummary:
    base = Path(base_dir)
    quantized = Path(quantized_dir)
    dflash2 = Path(dflash2_dir)
    payload_bytes = validate_structure(artifact)
    convert.preflight_conversion(base, quantized_dir, dflash2_dir)
    _verify_resources(artifact, base)
    resolved_device = torch.device(device)
    family_verify._verify_dflash2_module(artifact, dflash2, resolved_device)
    with ShardReader(base) as base_reader:
        w8_endpoint_rows, w8_endpoint_groups = family_verify._verify_w8_endpoints(
            artifact, base_reader
        )
    with ShardReader(quantized) as quantized_reader:
        source_weights, ratio_min, ratio_median, ratio_max = (
            _verify_source_nvfp4_weights(artifact, quantized_reader, resolved_device)
        )
        _verify_source_input_divisors(artifact, quantized_reader)
        _verify_decoded_control_parents(artifact, quantized_reader, resolved_device)
    # Derivation cross-check: implied = binary32(2688 / decode amax) must equal
    # the stored divisor exactly wherever the site's max element round-trips
    # (every single-tensor site family measures exactly 1.0), and may exceed it
    # only by site-scale sharing (the GDN a/b control tensors share the qkvz
    # site scale but are decoded into a different parent) or by one saturating
    # E2M1 step at the top element. A wrong scale convention would show as a
    # systematic offset, not this envelope.
    if (
        ratio_min < 1.0 - 1e-6
        or not 1.0 - 1e-6 <= ratio_median <= 1.0 + 1e-6
        or ratio_max > 1.25
    ):
        _error(
            "weight divisors are inconsistent with the 2688/amax derivation: "
            f"min={ratio_min:.4f} median={ratio_median:.4f} max={ratio_max:.4f}"
        )
    return VerificationSummary(
        objects=len(artifact.objects),
        tensors=len(inventory.TENSOR_SPECS),
        resources=len(inventory.RESOURCE_SPECS),
        w8_endpoint_weights=len(family_verify.W8_ENDPOINT_SPECS),
        w8_endpoint_rows=w8_endpoint_rows,
        w8_endpoint_groups=w8_endpoint_groups,
        source_nvfp4_weights=source_weights,
        source_input_divisors=len(recipe.SOURCE_INPUT_DIVISOR_RECIPES),
        decoded_control_parents=len(recipe.DECODED_CONTROL_RECIPES),
        dflash2_tensors=len(inventory.DFLASH2_TENSOR_SPECS),
        divisor_derivation_ratio_min=ratio_min,
        divisor_derivation_ratio_median=ratio_median,
        divisor_derivation_ratio_max=ratio_max,
        payload_bytes=payload_bytes,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--quantized-model", type=Path, required=True)
    parser.add_argument("--dflash2-model", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args(argv)
    with Artifact.open(arguments.artifact) as artifact:
        summary = verify_artifact(
            artifact,
            arguments.model,
            arguments.quantized_model,
            arguments.dflash2_model,
            device=arguments.device,
        )
    print(json.dumps(asdict(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
