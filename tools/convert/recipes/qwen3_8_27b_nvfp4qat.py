"""QAT NVFP4: import trained projections and decode GDN controls to BF16."""
from tools.convert.methods import cast_direct, import_encoded
from tools.convert.recipes.qwen3_8_profile import base_profile, group_text_parents


def configure(model, recipe, sources):
    base_profile(model, recipe)
    group_text_parents(model, recipe)
    quantized = sources['quantized']
    for name, parameter in model.parameters.items():
        if not name.startswith('text/layers/') or not parameter.projection:
            continue
        _, _, _, family, role = name.split('/')
        source = model.source(name, quantized, 'nvfp4')
        if family == 'gdn' and role in ('a_projection', 'b_projection'):
            recipe.assign(name, format='bf16', method=cast_direct, source=source)
        else:
            recipe.assign(name, format='nvfp4', method=import_encoded,
                          source=source, activation_policy='AllowA4')
