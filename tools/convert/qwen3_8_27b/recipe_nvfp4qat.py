"""Closed multi-source recipe for the QAT-sourced Qwen3.8-27B NVFP4 artifact.

Three fixed source roles:

- the QUASAR QAT checkpoint supplies every text linear NVFP4 parent
  word-for-word together with its site weight and input divisors, and the
  quantized GDN control parents decoded to BF16;
- the official BF16 base supplies all direct tensors, both W8 endpoints, and
  the draft head, MTP, and Vision components (the QAT checkpoint's
  unquantized tensors are validated bit-identical to this source);
- the incoai DFlash2 drafter supplies the drafter module of the complete
  product image.

The QAT factory emits one ``weight_global_scale``/``input_global_scale`` pair
per quantization site, shared by every constituent tensor of a fused parent;
``_same_divisor`` enforces that sharing before any word is copied.
"""

from __future__ import annotations

from dataclasses import dataclass
import struct
from typing import Iterable

import torch

from tools.artifact.numeric import valid_positive_fp32_word
from tools.convert.common.safetensors import ShardReader
from tools.convert.qwen3_6.common import recipe as family_recipe

from . import inventory_nvfp4qat as inventory
from . import nvfp4_encode
from .recipe_nvfp4full import (
    MatrixPart,
    MatrixSource,
    RowRange,
    SourceInputDivisorRecipe,
    SourceNvfp4WeightRecipe,
    _all,
    _q_part,
    _select_rows,
    _same_divisor,
    _word,
    _source as _matrix_source,
)


BASE_REPOSITORY = "Qwen/Qwen3.8-27B"
BASE_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
QUANTIZED_REPOSITORY = "QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4"
QUANTIZED_REVISION = "d8e6fbfa3e3a78899b440222b827430045a05b44"
LOCAL_ENCODER_PROFILE = nvfp4_encode.ENCODER_PROFILE


@dataclass(frozen=True, slots=True)
class DecodedControlRecipe:
    """GDN control parent materialized by decoding the QAT NVFP4 words to BF16."""

    object_name: str
    shape: tuple[int, int]
    parts: tuple[MatrixPart, ...]
    divisor_sources: tuple[MatrixSource, ...]


def _source(name: str, n: int, k: int) -> MatrixSource:
    return _matrix_source(name, n, k)


def _build_matrix_recipes() -> tuple[
    tuple[SourceNvfp4WeightRecipe, ...],
    tuple[SourceInputDivisorRecipe, ...],
    tuple[DecodedControlRecipe, ...],
]:
    source_weights: list[SourceNvfp4WeightRecipe] = []
    source_divisors: list[SourceInputDivisorRecipe] = []
    decoded_controls: list[DecodedControlRecipe] = []

    for layer in range(64):
        source_prefix = f"model.language_model.layers.{layer}."
        object_prefix = f"text/layers/{layer}/"
        if layer in inventory.FULL_ATTENTION_LAYERS:
            query = _source(source_prefix + "self_attn.q_proj", 12288, 5120)
            key = _source(source_prefix + "self_attn.k_proj", 1024, 5120)
            value = _source(source_prefix + "self_attn.v_proj", 1024, 5120)
            output = _source(source_prefix + "self_attn.o_proj", 5120, 6144)
            qkgv_name = object_prefix + "attention/query_key_gate_value"
            qkgv_parts = (
                _q_part(query, False),
                _all(key),
                _q_part(query, True),
                _all(value),
            )
            qkgv_site = (query, key, value)
            source_weights.append(
                SourceNvfp4WeightRecipe(
                    qkgv_name, (14336, 5120), qkgv_parts, qkgv_site
                )
            )
            source_divisors.append(
                SourceInputDivisorRecipe(
                    object_prefix
                    + "attention/input_projection/input_scale_divisor",
                    qkgv_site,
                    (qkgv_name,),
                )
            )
            output_name = object_prefix + "attention/output"
            source_weights.append(
                SourceNvfp4WeightRecipe(
                    output_name, output.shape, (_all(output),), (output,)
                )
            )
            source_divisors.append(
                SourceInputDivisorRecipe(
                    object_prefix
                    + "attention/output_projection/input_scale_divisor",
                    (output,),
                    (output_name,),
                )
            )
        else:
            query_key_value = _source(
                source_prefix + "linear_attn.in_proj_qkv", 10240, 5120
            )
            z = _source(source_prefix + "linear_attn.in_proj_z", 6144, 5120)
            control_a = _source(source_prefix + "linear_attn.in_proj_a", 48, 5120)
            control_b = _source(source_prefix + "linear_attn.in_proj_b", 48, 5120)
            output = _source(source_prefix + "linear_attn.out_proj", 5120, 6144)
            qkvz_name = object_prefix + "gdn/query_key_value_z"
            qkvz_parts = (_all(query_key_value), _all(z))
            qkvz_site = (query_key_value, z)
            source_weights.append(
                SourceNvfp4WeightRecipe(
                    qkvz_name, (16384, 5120), qkvz_parts, qkvz_site
                )
            )
            source_divisors.append(
                SourceInputDivisorRecipe(
                    object_prefix + "gdn/input_projection/input_scale_divisor",
                    qkvz_site,
                    (qkvz_name,),
                )
            )
            decoded_controls.append(
                DecodedControlRecipe(
                    object_prefix + "gdn/a_b_projection",
                    (96, 5120),
                    (_all(control_a), _all(control_b)),
                    (control_a, control_b),
                )
            )
            output_name = object_prefix + "gdn/output"
            source_weights.append(
                SourceNvfp4WeightRecipe(
                    output_name, output.shape, (_all(output),), (output,)
                )
            )
            source_divisors.append(
                SourceInputDivisorRecipe(
                    object_prefix + "gdn/output_projection/input_scale_divisor",
                    (output,),
                    (output_name,),
                )
            )

        gate = _source(source_prefix + "mlp.gate_proj", 17408, 5120)
        up = _source(source_prefix + "mlp.up_proj", 17408, 5120)
        down = _source(source_prefix + "mlp.down_proj", 5120, 17408)
        gate_up_name = object_prefix + "mlp/gate_up"
        down_name = object_prefix + "mlp/down"
        gate_up_site = (gate, up)
        source_weights.extend(
            (
                SourceNvfp4WeightRecipe(
                    gate_up_name, (34816, 5120), (_all(gate), _all(up)), gate_up_site
                ),
                SourceNvfp4WeightRecipe(
                    down_name, down.shape, (_all(down),), (down,)
                ),
            )
        )
        source_divisors.extend(
            (
                SourceInputDivisorRecipe(
                    object_prefix + "mlp/gate_up_projection/input_scale_divisor",
                    gate_up_site,
                    (gate_up_name,),
                ),
                SourceInputDivisorRecipe(
                    object_prefix + "mlp/down_projection/input_scale_divisor",
                    (down,),
                    (down_name,),
                ),
            )
        )

    return (
        tuple(source_weights),
        tuple(source_divisors),
        tuple(decoded_controls),
    )


(
    SOURCE_NVFP4_WEIGHT_RECIPES,
    SOURCE_INPUT_DIVISOR_RECIPES,
    DECODED_CONTROL_RECIPES,
) = _build_matrix_recipes()

SOURCE_NVFP4_WEIGHTS_BY_NAME = {
    item.object_name: item for item in SOURCE_NVFP4_WEIGHT_RECIPES
}
SOURCE_INPUT_DIVISORS_BY_NAME = {
    item.object_name: item for item in SOURCE_INPUT_DIVISOR_RECIPES
}
DECODED_CONTROLS_BY_NAME = {item.object_name: item for item in DECODED_CONTROL_RECIPES}

NVFP4_SOURCES = tuple(
    dict.fromkeys(
        [
            source
            for recipe in SOURCE_NVFP4_WEIGHT_RECIPES
            for source in recipe.divisor_sources
        ]
        + [
            source
            for recipe in DECODED_CONTROL_RECIPES
            for source in recipe.divisor_sources
        ]
    )
)


def _build_base_direct_recipes() -> tuple[family_recipe.TensorRecipe, ...]:
    """Direct base tensors: identical to the nvfp4full allocation except that
    the GDN control parent is owned by the QAT decode route."""
    from . import recipe_nvfp4full as nvfp4full

    return tuple(
        recipe
        for recipe in nvfp4full.BASE_DIRECT_RECIPES
        if not recipe.object_name.endswith("/gdn/a_b_projection")
    )


BASE_DIRECT_RECIPES = _build_base_direct_recipes()
BASE_DIRECT_BY_NAME = {item.object_name: item for item in BASE_DIRECT_RECIPES}
BASE_DIRECT_SPECS = tuple(
    spec
    for spec in inventory.TEXT_CORE_TENSOR_SPECS
    if spec.name in BASE_DIRECT_BY_NAME
)

# Endpoints, draft head, MTP, and Vision reuse the registered official-source
# recipe objects of the nvfp4full profile; they are quantized from the BF16
# base with the canonical grouped-int encoder.
from .recipe_nvfp4full import OFFICIAL_RECIPES_BY_NAME  # noqa: E402

OFFICIAL_TENSOR_SPECS = tuple(
    spec
    for spec in inventory.TENSOR_SPECS
    if spec.name in OFFICIAL_RECIPES_BY_NAME
)


def _validate_parts(
    object_name: str, shape: tuple[int, int], parts: tuple[MatrixPart, ...]
) -> None:
    rows = sum(part.output_rows for part in parts)
    if not parts or (rows, parts[0].source.shape[1]) != shape:
        raise ValueError(f"{object_name}: invalid fused row geometry")
    if any(part.source.shape[1] != shape[1] for part in parts):
        raise ValueError(f"{object_name}: incompatible source K")


def validate_recipe() -> None:
    family_recipe.validate_recipe_coverage(BASE_DIRECT_RECIPES, BASE_DIRECT_SPECS)
    family_recipe.validate_recipe_coverage(
        tuple(OFFICIAL_RECIPES_BY_NAME[spec.name] for spec in OFFICIAL_TENSOR_SPECS),
        OFFICIAL_TENSOR_SPECS,
    )
    if (
        len(SOURCE_NVFP4_WEIGHT_RECIPES),
        len(SOURCE_INPUT_DIVISOR_RECIPES),
        len(DECODED_CONTROL_RECIPES),
        len(NVFP4_SOURCES),
        len(BASE_DIRECT_RECIPES),
    ) != (256, 256, 48, 496, 353):
        raise ValueError("Qwen3.8 nvfp4qat source recipe is incomplete")

    inventory_nvfp4_names = {spec.name for spec in inventory.NVFP4_TENSOR_SPECS}
    if set(SOURCE_NVFP4_WEIGHTS_BY_NAME) != inventory_nvfp4_names:
        raise ValueError("source NVFP4 routes do not cover the NVFP4 parents")
    divisor_names = {spec.name for spec in inventory.INPUT_SCALE_DIVISOR_SPECS}
    if set(SOURCE_INPUT_DIVISORS_BY_NAME) != divisor_names:
        raise ValueError("input-divisor routes do not cover the divisor sites")
    bound_source = {
        name for site in SOURCE_INPUT_DIVISOR_RECIPES for name in site.weight_names
    }
    if bound_source != set(SOURCE_NVFP4_WEIGHTS_BY_NAME):
        raise ValueError("source input-divisor sites do not bind their parents once")
    gdn_control_names = {
        f"text/layers/{layer}/gdn/a_b_projection" for layer in inventory.GDN_LAYERS
    }
    if set(DECODED_CONTROLS_BY_NAME) != gdn_control_names:
        raise ValueError("decoded control routes do not cover the GDN control parents")

    ownership = (
        set(SOURCE_NVFP4_WEIGHTS_BY_NAME),
        set(SOURCE_INPUT_DIVISORS_BY_NAME),
        set(DECODED_CONTROLS_BY_NAME),
        set(BASE_DIRECT_BY_NAME),
        set(OFFICIAL_RECIPES_BY_NAME),
        # The DFlash2 suffix rides the upstream dflash2_recipe route.
        {spec.name for spec in inventory.DFLASH2_TENSOR_SPECS},
    )
    names: set[str] = set()
    for route in ownership:
        if names & route:
            raise ValueError("more than one source route owns an artifact tensor")
        names.update(route)
    if names != {spec.name for spec in inventory.TENSOR_SPECS}:
        missing = {spec.name for spec in inventory.TENSOR_SPECS} - names
        extra = names - {spec.name for spec in inventory.TENSOR_SPECS}
        raise ValueError(
            f"source routes do not cover the inventory: {sorted(missing)[:2]} {sorted(extra)[:2]}"
        )

    for recipe_item in SOURCE_NVFP4_WEIGHT_RECIPES:
        _validate_parts(recipe_item.object_name, recipe_item.shape, recipe_item.parts)
    for recipe_item in DECODED_CONTROL_RECIPES:
        _validate_parts(recipe_item.object_name, recipe_item.shape, recipe_item.parts)


def source_field_requirements() -> dict[str, tuple[tuple[int, ...], str]]:
    requirements: dict[str, tuple[tuple[int, ...], str]] = {}
    for source in NVFP4_SOURCES:
        n, k = source.shape
        for suffix, shape, dtype in (
            ("weight_packed", (n, k // 2), "U8"),
            ("weight_scale", (n, k // 16), "F8_E4M3"),
            ("weight_global_scale", (1,), "F32"),
            ("input_global_scale", (1,), "F32"),
        ):
            name = source.field(suffix)
            previous = requirements.setdefault(name, (shape, dtype))
            if previous != (shape, dtype):
                raise ValueError(f"inconsistent source declaration for {name}")
    return requirements


def preflight_quantized_metadata(reader: ShardReader) -> dict[str, int]:
    """Validate that the QAT source carries the complete text-linear allocation."""

    requirements = source_field_requirements()
    missing = set(requirements).difference(reader.names)
    if missing:
        raise ValueError(f"QAT source is missing {sorted(missing)[0]}")
    metadata = reader.metadata(sorted(requirements))
    dtype_counts: dict[str, int] = {}
    for name, (shape, dtype) in requirements.items():
        actual = metadata[name]
        if actual.shape != shape or actual.dtype != dtype:
            raise ValueError(
                f"{name}: source signature {(actual.shape, actual.dtype)} "
                f"!= {(shape, dtype)}"
            )
        dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
    if dtype_counts != {"U8": 496, "F8_E4M3": 496, "F32": 992}:
        raise ValueError(f"unexpected QAT-source allocation: {dtype_counts}")
    # Prove the checkpoint is the all-linear NVFP4 allocation: outside the
    # 496 site scale planes there must be no other E4M3 matrix plane.
    stray_e4m3 = sorted(
        name
        for name, item in reader.metadata(reader.names).items()
        if item.dtype == "F8_E4M3" and not name.endswith(".weight_scale")
    )
    if stray_e4m3:
        raise ValueError(f"QAT source carries unexpected E4M3 fields: {stray_e4m3[:3]}")
    return dtype_counts


def materialize_source_nvfp4_weight(
    recipe_item: SourceNvfp4WeightRecipe,
    reader: ShardReader,
) -> tuple[torch.Tensor, torch.Tensor, bytes]:
    packed_parts: list[torch.Tensor] = []
    scale_parts: list[torch.Tensor] = []
    source_words: dict[MatrixSource, tuple[torch.Tensor, torch.Tensor]] = {}
    for part in recipe_item.parts:
        words = source_words.get(part.source)
        if words is None:
            n, k = part.source.shape
            source_packed = reader.get(part.source.field("weight_packed"))
            source_scales = reader.get(part.source.field("weight_scale"))
            if (
                source_packed.dtype != torch.uint8
                or tuple(source_packed.shape) != (n, k // 2)
                or source_scales.dtype != torch.float8_e4m3fn
                or tuple(source_scales.shape) != (n, k // 16)
            ):
                raise ValueError(
                    f"{part.source.name}: materialized NVFP4 source signature mismatch"
                )
            words = (source_packed, source_scales.view(torch.uint8))
            source_words[part.source] = words
        packed_parts.append(_select_rows(words[0], part))
        scale_parts.append(_select_rows(words[1], part))
    packed = (
        packed_parts[0].contiguous()
        if len(packed_parts) == 1
        else torch.cat(packed_parts, dim=0)
    )
    scales = (
        scale_parts[0].contiguous()
        if len(scale_parts) == 1
        else torch.cat(scale_parts, dim=0)
    )
    divisor = _same_divisor(reader, recipe_item.divisor_sources, "weight_global_scale")
    if tuple(packed.shape) != (recipe_item.shape[0], recipe_item.shape[1] // 2) or tuple(
        scales.shape
    ) != (recipe_item.shape[0], recipe_item.shape[1] // 16):
        raise ValueError(
            f"{recipe_item.object_name}: materialized NVFP4 shape mismatch"
        )
    return packed, scales, struct.pack("<I", divisor)


def materialize_source_input_divisor(
    recipe_item: SourceInputDivisorRecipe, reader: ShardReader
) -> torch.Tensor:
    word = _same_divisor(reader, recipe_item.sources, "input_global_scale")
    return torch.frombuffer(
        bytearray(struct.pack("<I", word)), dtype=torch.float32
    ).reshape(())


def materialize_decoded_control(
    recipe_item: DecodedControlRecipe,
    reader: ShardReader,
    device: torch.device,
) -> torch.Tensor:
    """Decode the QAT control words to BF16: the represented FP32 values of the
    quantized source, rounded once to the artifact's BF16 storage."""

    divisor = _same_divisor(reader, recipe_item.divisor_sources, "weight_global_scale")
    divisor32 = struct.unpack("<f", struct.pack("<I", divisor))[0]
    pieces: list[torch.Tensor] = []
    for part in recipe_item.parts:
        n, k = part.source.shape
        packed = _select_rows(
            reader.get(part.source.field("weight_packed")), part
        ).contiguous()
        scales = _select_rows(
            reader.get(part.source.field("weight_scale")).view(torch.uint8), part
        ).contiguous()
        if tuple(packed.shape) != (n, k // 2) or tuple(scales.shape) != (n, k // 16):
            raise ValueError(f"{part.source.name}: control source shape mismatch")
        decoded = nvfp4_encode.dequantize_nvfp4(
            nvfp4_encode.Nvfp4Words(
                packed_codes=packed,
                natural_scales=scales,
                weight_divisor=divisor32,
            ),
            device=device,
        )
        pieces.append(decoded.to(torch.bfloat16).cpu())
        del decoded
    matrix = pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=0)
    if tuple(matrix.shape) != recipe_item.shape or matrix.dtype != torch.bfloat16:
        raise ValueError(f"{recipe_item.object_name}: decoded control signature mismatch")
    return matrix


validate_recipe()


__all__ = [
    "BASE_DIRECT_BY_NAME",
    "BASE_DIRECT_RECIPES",
    "BASE_DIRECT_SPECS",
    "BASE_REPOSITORY",
    "BASE_REVISION",
    "DECODED_CONTROLS_BY_NAME",
    "DECODED_CONTROL_RECIPES",
    "DecodedControlRecipe",
    "LOCAL_ENCODER_PROFILE",
    "NVFP4_SOURCES",
    "OFFICIAL_RECIPES_BY_NAME",
    "OFFICIAL_TENSOR_SPECS",
    "QUANTIZED_REPOSITORY",
    "QUANTIZED_REVISION",
    "SOURCE_INPUT_DIVISORS_BY_NAME",
    "SOURCE_INPUT_DIVISOR_RECIPES",
    "SOURCE_NVFP4_WEIGHTS_BY_NAME",
    "SOURCE_NVFP4_WEIGHT_RECIPES",
    "materialize_decoded_control",
    "materialize_source_input_divisor",
    "materialize_source_nvfp4_weight",
    "preflight_quantized_metadata",
    "validate_recipe",
]
