"""Saved fork QAT profile, not an alias of the official mixed-FP8 profile.

Use --model BF16_BASE --source quantized=QAT_SOURCE with this recipe file.
All Text attention/GDN/MLP parents import NVFP4 words and divisors. GDN controls
are decoded from that quantized source and explicitly cast to BF16. Vocabulary,
direct weights and optional components retain the full profile's base sources.
"""
from tools.convert.methods import cast_direct, import_encoded
from tools.convert.recipes.qwen3_8_27b_nvfp4full import base_profile, group_text_parents


def configure(model, recipe, sources):
    base_profile(model, recipe)
    group_text_parents(model, recipe)
    quantized = sources['quantized']
    for name, parameter in model.parameters.items():
        if not name.startswith('text/layers/') or not parameter.projection:
            continue
        source = model.source(name, quantized, 'nvfp4')
        if name.endswith(('/gdn/a_projection', '/gdn/b_projection')):
            recipe.assign(name, format='bf16', method=cast_direct, source=source)
        else:
            recipe.assign(name, format='nvfp4', method=import_encoded,
                          source=source, activation_policy='AllowA4')
