"""Full NVFP4: imported early MLP, calibrated local projections, nine BF16 parents."""
from dataclasses import dataclass
import json

from tools.convert.methods import AuxiliaryValue, import_encoded
from tools.convert.recipes.qwen3_8_profile import base_profile, group_text_parents
from tools.convert.quantization.nvfp4 import nvfp4_maxabs


@dataclass(frozen=True)
class Projection:
    name: str
    layer: int
    family: str
    role: str

    @property
    def retains_bf16(self):
        return ((self.family == 'gdn' and self.role in ('a_projection', 'b_projection'))
                or (self.family == 'attention' and self.role != 'output' and self.layer < 24)
                or (self.family == 'attention' and self.role == 'output' and self.layer in (3, 7))
                or (self.family == 'gdn' and self.role == 'output' and self.layer == 4))

    @property
    def calibration_site(self):
        if self.family == 'mlp':
            parent = 'down_projection' if self.role == 'down' else 'gate_up_projection'
        else:
            parent = 'output_projection' if self.role == 'output' else 'input_projection'
        return f'text/layers/{self.layer}/{self.family}/{parent}/input_scale_divisor'


def calibrated_divisors(path, projections):
    document = json.loads(path.read_text(encoding='utf-8'))
    measured = document.get('measured_sites', {})
    expected = {projection.calibration_site for projection in projections}
    if set(measured) != expected:
        raise ValueError('calibration site set differs from the full profile local sites')
    return {
        site: AuxiliaryValue.activation_divisor(float(measured[site]['input_scale_divisor']))
        for site in expected
    }


def configure(model, recipe, sources):
    base_profile(model, recipe)
    group_text_parents(model, recipe)
    quantized = sources['quantized']
    local = []
    for name, parameter in model.parameters.items():
        if not name.startswith('text/layers/') or not parameter.projection:
            continue
        _, _, layer, family, role = name.split('/')
        projection = Projection(name, int(layer), family, role)
        if projection.retains_bf16:
            continue
        if family == 'mlp' and projection.layer < 56:
            recipe.assign(name, format='nvfp4', method=import_encoded,
                          source=model.source(name, quantized, 'nvfp4'), activation_policy='AllowA4')
        else:
            recipe.assign(name, format='nvfp4', method=nvfp4_maxabs, activation_policy='AllowA4')
            local.append(projection)
    divisors = calibrated_divisors(sources.path('calibration'), local)
    for projection in local:
        for input_name in model.parameters[projection.name].inputs:
            recipe.use(projection.name, input_name,
                       auxiliaries={'activation_input_divisor': divisors[projection.calibration_site]})
