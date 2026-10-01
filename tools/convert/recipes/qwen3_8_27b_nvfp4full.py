"""Saved fork full profile on v3 logical parameters (not the mixed-FP8 recipe).

python -m tools.convert --model BASE --source quantized=QUANTIZED \
  --source calibration=CALIBRATION.json --recipe tools/convert/recipes/qwen3_8_27b_nvfp4full.py \
  --out OUTPUT.ninfer

Optional Vision/MTP/DFlash2 use the base/companion BF16 sources and the saved
W8 module allocation. The calibration JSON retains the baseline measured_sites
keys; there is no target inventory or runtime identity extension.
"""
import json

from tools.convert.methods import AuxiliaryValue, grouped_absmax, import_encoded
from tools.convert.official_recipes import _optional, Q8
from tools.convert.quantization.nvfp4 import nvfp4_maxabs


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


def local_site(name):
    prefix, family, role = name.rsplit('/', 2)
    if family == 'mlp':
        site = 'down_projection' if role == 'down' else 'gate_up_projection'
    else:
        site = 'output_projection' if role == 'output' else 'input_projection'
    return f'{prefix}/{family}/{site}/input_scale_divisor'


def configure(model, recipe, sources):
    base_profile(model, recipe)
    quantized = sources['quantized']
    local = []
    for name, parameter in model.parameters.items():
        if not name.startswith('text/layers/') or not parameter.projection:
            continue
        layer = int(name.split('/')[2])
        if name.endswith(('/gdn/a_projection', '/gdn/b_projection')):
            continue
        exception = (('/attention/' in name and not name.endswith('/output') and layer < 24)
                     or (name.endswith('/attention/output') and layer in (3, 7))
                     or (name.endswith('/gdn/output') and layer == 4))
        if exception:
            continue
        if '/mlp/' in name and layer < 56:
            recipe.assign(name, format='nvfp4', method=import_encoded,
                          source=model.source(name, quantized, 'nvfp4'), activation_policy='AllowA4')
        else:
            recipe.assign(name, format='nvfp4', method=nvfp4_maxabs, activation_policy='AllowA4')
            local.append(name)
    # One parent owns each baseline global weight divisor. Keep explicit groups
    # independent of future upstream changes to preferred physical packing.
    group_text_parents(model, recipe)
    path = sources.path('calibration')
    document = json.loads(path.read_text(encoding='utf-8'))
    measured = document.get('measured_sites', {})
    expected = {local_site(name) for name in local}
    if set(measured) != expected:
        raise ValueError('calibration site set differs from the full profile local sites')
    for name in local:
        value = AuxiliaryValue.activation_divisor(float(measured[local_site(name)]['input_scale_divisor']))
        for input_name in model.parameters[name].inputs:
            recipe.use(name, input_name, auxiliaries={'activation_input_divisor': value})
