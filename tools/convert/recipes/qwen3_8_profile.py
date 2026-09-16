"""Shared Qwen3.8 full/QAT vocabulary, optional-component and parent allocation."""
from tools.convert.methods import grouped_absmax
from tools.convert.official_recipes import _optional, Q8


def base_profile(model, recipe):
    if model.config.get('num_hidden_layers') != 64 or 'num_experts' in model.config:
        raise ValueError('fork profiles require the 64-layer Qwen3.8-27B dense model')
    _optional(model, recipe)
    for name in ('text/token_embedding', 'text/output_head'):
        recipe.assign(name, format=Q8, method=grouped_absmax)


def group_text_parents(model, recipe):
    for layer in range(64):
        prefix = f'text/layers/{layer}/'
        if prefix + 'attention/query' in model.parameters:
            recipe.group(tuple(prefix + 'attention/' + role for role in ('query', 'key', 'gate', 'value')))
        else:
            recipe.group(tuple(prefix + 'gdn/' + role for role in ('query', 'key', 'value', 'z')))
            recipe.group((prefix + 'gdn/a_projection', prefix + 'gdn/b_projection'))
        recipe.group((prefix + 'mlp/gate', prefix + 'mlp/up'))
