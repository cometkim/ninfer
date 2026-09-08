"""Persistent-object contract for the QAT-sourced Qwen3.8-27B NVFP4 artifact."""

from __future__ import annotations

from tools.convert.qwen3_6.common.inventory import (
    BF16,
    CONTIGUOUS_LAYOUT,
    FP32,
    I32,
    LogicalAliasSpec,
    LogicalRowViewSpec,
    Q4,
    Q5,
    Q6,
    RESOURCE_SPECS,
    ROW_SPLIT_LAYOUT,
    ResourceSpec,
    StoredObjectSpec,
    TensorSpec,
    W8,
    build_vision_specs,
)
from tools.convert.qwen3_8_27b.inventory_nvfp4full import (
    DFLASH2_TENSOR_SPECS as _UPSTREAM_DFLASH2_TENSOR_SPECS,
    LOGICAL_ROW_VIEW_SPECS as _FAMILY_ROW_VIEW_SPECS,
    ALIAS_SPECS as _FAMILY_ALIAS_SPECS,
    _build_draft_head_specs,
    _build_mtp_specs,
)


MODEL_ID = "qwen3.8-27b"
WEIGHTS_ID = "nvfp4qat"
TARGET_KEY = "qwen3_8_27b"

NVFP4 = "NVFP4"
BLOCK_SCALE_LAYOUT = "blockscale-k16-m128x4-v1"

FULL_ATTENTION_LAYERS = tuple(range(3, 64, 4))
GDN_LAYERS = tuple(layer for layer in range(64) if layer not in FULL_ATTENTION_LAYERS)

# The QUASAR QAT checkpoint quantizes every text linear; this profile carries no
# BF16 exception parents and no locally quantized parents.
FORMAT_NAMES = (BF16, FP32, I32, Q4, Q5, Q6, W8, NVFP4)
LAYOUT_NAMES = (CONTIGUOUS_LAYOUT, ROW_SPLIT_LAYOUT, BLOCK_SCALE_LAYOUT)


def tensor_spec(
    name: str,
    shape: tuple[int, ...],
    numeric_format: str,
) -> TensorSpec:
    if numeric_format in (BF16, FP32, I32):
        layout = CONTIGUOUS_LAYOUT
    elif numeric_format in (Q4, Q5, Q6, W8):
        layout = ROW_SPLIT_LAYOUT
    elif numeric_format == NVFP4:
        layout = BLOCK_SCALE_LAYOUT
    else:
        raise ValueError(f"unsupported Qwen3.8 nvfp4qat format: {numeric_format}")
    return TensorSpec(name, shape, numeric_format, layout)


def _input_scale_divisor_name(matrix_name: str) -> str | None:
    prefix, suffix = matrix_name.rsplit("/", 1)
    if suffix == "query_key_gate_value" and prefix.endswith("/attention"):
        return prefix + "/input_projection/input_scale_divisor"
    if suffix == "output" and prefix.endswith("/attention"):
        return prefix + "/output_projection/input_scale_divisor"
    if suffix == "query_key_value_z" and prefix.endswith("/gdn"):
        return prefix + "/input_projection/input_scale_divisor"
    if suffix == "output" and prefix.endswith("/gdn"):
        return prefix + "/output_projection/input_scale_divisor"
    if suffix == "gate_up" and prefix.endswith("/mlp"):
        return prefix + "/gate_up_projection/input_scale_divisor"
    if suffix == "down" and prefix.endswith("/mlp"):
        return prefix + "/down_projection/input_scale_divisor"
    return None


def _build_text_core_specs() -> tuple[TensorSpec, ...]:
    specs: list[TensorSpec] = [tensor_spec("text/token_embedding", (248320, 5120), W8)]

    def emit(name: str, shape: tuple[int, ...], numeric_format: str) -> None:
        specs.append(tensor_spec(name, shape, numeric_format))
        if numeric_format == NVFP4:
            specs.append(tensor_spec(_input_scale_divisor_name(name), (), FP32))

    for layer in range(64):
        prefix = f"text/layers/{layer}/"
        emit(prefix + "input_norm", (5120,), BF16)
        if layer in FULL_ATTENTION_LAYERS:
            emit(prefix + "attention/query_key_gate_value", (14336, 5120), NVFP4)
            emit(prefix + "attention/query_norm", (256,), BF16)
            emit(prefix + "attention/key_norm", (256,), BF16)
            emit(prefix + "attention/output", (5120, 6144), NVFP4)
        else:
            emit(prefix + "gdn/a_log", (48,), FP32)
            emit(prefix + "gdn/dt_bias", (48,), FP32)
            emit(prefix + "gdn/convolution", (4, 10240), BF16)
            emit(prefix + "gdn/a_b_projection", (96, 5120), BF16)
            emit(prefix + "gdn/query_key_value_z", (16384, 5120), NVFP4)
            emit(prefix + "gdn/norm", (128,), BF16)
            emit(prefix + "gdn/output", (5120, 6144), NVFP4)
        emit(prefix + "post_attention_norm", (5120,), BF16)
        emit(prefix + "mlp/gate_up", (34816, 5120), NVFP4)
        emit(prefix + "mlp/down", (5120, 17408), NVFP4)

    emit("text/final_norm", (5120,), BF16)
    emit("text/output_head", (248320, 5120), W8)
    return tuple(specs)


TEXT_CORE_TENSOR_SPECS = _build_text_core_specs()
DRAFT_HEAD_TENSOR_SPECS = _build_draft_head_specs()
MTP_TENSOR_SPECS = _build_mtp_specs()
VISION_TENSOR_SPECS = build_vision_specs(5120)
# The DFlash2 module follows the upstream persistent suffix contract
# (matrices W8G32_F16S), identical to the registered profiles.
DFLASH2_TENSOR_SPECS = _UPSTREAM_DFLASH2_TENSOR_SPECS

# The DFlash2 module appends after the Vision merger objects exactly as in the
# nvfp4full image: one complete product image, no optional profiles.
TENSOR_SPECS = (
    TEXT_CORE_TENSOR_SPECS
    + DRAFT_HEAD_TENSOR_SPECS
    + MTP_TENSOR_SPECS
    + VISION_TENSOR_SPECS
    + DFLASH2_TENSOR_SPECS
)
OBJECT_SPECS: tuple[StoredObjectSpec, ...] = RESOURCE_SPECS + TENSOR_SPECS

FORMAT_COUNTS = {
    numeric_format: sum(spec.format == numeric_format for spec in TENSOR_SPECS)
    for numeric_format in FORMAT_NAMES
}
LAYOUT_COUNTS = {
    layout: sum(spec.layout == layout for spec in TENSOR_SPECS)
    for layout in LAYOUT_NAMES
}

# The fused-parent row views and aliases are the family geometry: the same
# fused parents with the same row order as every other 27B profile.
LOGICAL_ROW_VIEW_SPECS = _FAMILY_ROW_VIEW_SPECS
ALIAS_SPECS = _FAMILY_ALIAS_SPECS

NVFP4_TENSOR_SPECS = tuple(spec for spec in TENSOR_SPECS if spec.format == NVFP4)
INPUT_SCALE_DIVISOR_SPECS = tuple(
    spec
    for spec in TENSOR_SPECS
    if spec.format == FP32 and spec.name.endswith("/input_scale_divisor")
)
# Every non-module NVFP4 parent is copied word-for-word from the QAT source.
SOURCE_NVFP4_WEIGHT_SPECS = tuple(
    spec
    for spec in NVFP4_TENSOR_SPECS
    if not spec.name.startswith("dflash2/")
)


def validate_inventory() -> None:
    names = tuple(spec.name for spec in OBJECT_SPECS)
    if len(names) != len(set(names)):
        raise ValueError("Qwen3.8 nvfp4qat inventory contains duplicate names")
    if (
        len(TEXT_CORE_TENSOR_SPECS),
        len(DRAFT_HEAD_TENSOR_SPECS),
        len(MTP_TENSOR_SPECS),
        len(VISION_TENSOR_SPECS),
        len(DFLASH2_TENSOR_SPECS),
        len(TENSOR_SPECS),
        len(OBJECT_SPECS),
        len(NVFP4_TENSOR_SPECS),
        len(INPUT_SCALE_DIVISOR_SPECS),
        len(SOURCE_NVFP4_WEIGHT_SPECS),
    ) != (915, 2, 12, 333, 66, 1328, 1334, 256, 256, 256):
        raise ValueError("Qwen3.8 nvfp4qat inventory is incomplete")
    if FORMAT_COUNTS != {
        BF16: 579,
        FP32: 352,
        I32: 1,
        Q4: 55,
        Q5: 54,
        Q6: 1,
        W8: 30,
        NVFP4: 256,
    }:
        raise ValueError(f"unexpected numeric allocation: {FORMAT_COUNTS}")
    if LAYOUT_COUNTS != {
        CONTIGUOUS_LAYOUT: 932,
        ROW_SPLIT_LAYOUT: 140,
        BLOCK_SCALE_LAYOUT: 256,
    }:
        raise ValueError(f"unexpected layout allocation: {LAYOUT_COUNTS}")
    for spec in NVFP4_TENSOR_SPECS:
        if _input_scale_divisor_name(spec.name) is None:
            raise ValueError(f"NVFP4 parent {spec.name} has no divisor site name")
    divisor_names = {spec.name for spec in INPUT_SCALE_DIVISOR_SPECS}
    expected = {_input_scale_divisor_name(spec.name) for spec in NVFP4_TENSOR_SPECS}
    if divisor_names != expected:
        raise ValueError("NVFP4 divisor sites do not cover the NVFP4 parents")
    if any(
        spec.format == BF16
        and spec.name.endswith(
            ("attention/query_key_gate_value", "attention/output", "gdn/output")
        )
        for spec in TEXT_CORE_TENSOR_SPECS
    ):
        raise ValueError("the QAT profile carries no BF16 exception parents")


validate_inventory()


__all__ = [
    "ALIAS_SPECS",
    "BF16",
    "BLOCK_SCALE_LAYOUT",
    "CONTIGUOUS_LAYOUT",
    "DRAFT_HEAD_TENSOR_SPECS",
    "FORMAT_COUNTS",
    "FORMAT_NAMES",
    "FP32",
    "FULL_ATTENTION_LAYERS",
    "GDN_LAYERS",
    "I32",
    "INPUT_SCALE_DIVISOR_SPECS",
    "LAYOUT_COUNTS",
    "LAYOUT_NAMES",
    "LOGICAL_ROW_VIEW_SPECS",
    "MODEL_ID",
    "MTP_TENSOR_SPECS",
    "NVFP4",
    "NVFP4_TENSOR_SPECS",
    "OBJECT_SPECS",
    "Q4",
    "Q5",
    "Q6",
    "RESOURCE_SPECS",
    "ROW_SPLIT_LAYOUT",
    "ResourceSpec",
    "SOURCE_NVFP4_WEIGHT_SPECS",
    "StoredObjectSpec",
    "TARGET_KEY",
    "TENSOR_SPECS",
    "TEXT_CORE_TENSOR_SPECS",
    "TensorSpec",
    "VISION_TENSOR_SPECS",
    "W8",
    "WEIGHTS_ID",
    "tensor_spec",
    "validate_inventory",
]
