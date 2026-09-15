import json
import struct
from types import SimpleNamespace

import unittest

import torch

from tools.convert.model import Model, Parameter
from tools.convert.recipe import Recipe
from tools.convert.methods import cast_direct, grouped_absmax, import_encoded
from tools.convert.sources.logical import LogicalSource
from tools.convert.quantization.nvfp4 import e2m1_rne_codes, nvfp4_maxabs
from tools.convert.recipes import qwen3_8_27b_nvfp4full as full
from tools.convert.recipes import qwen3_8_27b_nvfp4qat as qat
from tools.convert.qwen3_5 import _Builder


def logical_model():
    config = dict(num_hidden_layers=64, hidden_size=5120, num_attention_heads=24,
                  num_key_value_heads=4, head_dim=256, linear_num_key_heads=16,
                  linear_key_head_dim=128, linear_num_value_heads=48,
                  linear_value_head_dim=128, linear_conv_kernel_dim=4)
    model = Model({'text': {'config': config}})
    # Exercise the real adapter's logical selectors and packing candidates without weights.
    store = SimpleNamespace(path='synthetic', config={})
    builder = _Builder(model)
    for i in range(64):
        prefix = f'text/layers/{i}/'
        (builder.attention if i % 4 == 3 else builder.gdn)(prefix, f'layers.{i}.', store, config)
        builder.dense(prefix, f'layers.{i}.', store, 5120, 17408)
    for name in ('text/token_embedding', 'text/output_head'):
        builder.add(name, store, name, (248320, 5120), inputs=('vocab',))
    return model, store


def _synthetic_sources(recipe):
    from dataclasses import replace
    from tools.convert.sources.logical import EncodedRows
    for name, selections in recipe.selections.items():
        def source_for(selection):
            shape = selection.source.shape
            k = shape[-1] if shape else 1
            def encoded(begin, end):
                return EncodedRows('nvfp4', torch.zeros((end-begin, k//2), dtype=torch.uint8),
                                   torch.zeros((end-begin, k//16), dtype=torch.uint8), struct.pack('<f', 2.))
            return LogicalSource(shape, name, lambda a, b: torch.zeros(b-a), encoded,
                                 lambda: struct.pack('<f', 2.), lambda: struct.pack('<f', 3.))
        recipe.selections[name] = [replace(selection, source=source_for(selection)) for selection in selections]


def test_full_allocation_and_calibration(tmp_path):
    model, quantized = logical_model()
    expected = set()
    for i in range(64):
        prefix = f'text/layers/{i}/'
        family = 'attention' if i % 4 == 3 else 'gdn'
        if family == 'gdn' or i >= 24:
            expected.add(prefix + family + '/input_projection/input_scale_divisor')
        if (family == 'attention' and i not in (3, 7)) or (family == 'gdn' and i != 4):
            expected.add(prefix + family + '/output_projection/input_scale_divisor')
        if i >= 56:
            expected.update(prefix + 'mlp/' + role + '_projection/input_scale_divisor'
                            for role in ('gate_up', 'down'))
    assert len(expected) == 135
    path = tmp_path / 'calibration.json'
    path.write_text(json.dumps({'measured_sites': {name: {'input_scale_divisor': 3.25} for name in expected}}))
    class Sources(dict):
        def path(self, name):
            assert name == 'calibration'
            return path
    recipe = Recipe(model)
    full.configure(model, recipe, Sources(quantized=quantized))
    for name, selections in recipe.selections.items():
        selected = selections[0]
        if name in ('text/token_embedding', 'text/output_head'):
            assert (selected.format, selected.method) == ('q8_g32_fp16', grouped_absmax)
        elif '/mlp/' in name:
            assert selected.format == 'nvfp4'
            assert selected.method is (import_encoded if int(name.split('/')[2]) < 56 else nvfp4_maxabs)
    assert recipe.selections['text/layers/3/attention/query'][0].format == 'bf16'
    assert recipe.selections['text/layers/4/gdn/output'][0].format == 'bf16'
    assert recipe.selections['text/layers/27/attention/query'][0].method is nvfp4_maxabs
    assert {full.local_site(key[0]) for key in recipe.auxiliary_overrides} == expected
    # Exercise actual v3 preparation at model shapes with bounded synthetic reads.
    _synthetic_sources(recipe)
    prepared = recipe.prepare(device='cpu')
    assert sum(job.method_name.endswith('nvfp4_maxabs') for job in prepared.weights) == 135
    assert sum(job.method_name.endswith('import_encoded') for job in prepared.weights) == 112
    path.write_text('{"measured_sites": {}}')
    with unittest.TestCase().assertRaisesRegex(ValueError, 'calibration site set'):
        full.configure(model, Recipe(model), Sources(quantized=quantized))


def test_qat_distinct_from_official():
    model, quantized = logical_model()
    recipe = Recipe(model)
    qat.configure(model, recipe, {'quantized': quantized})
    for name, parameter in model.parameters.items():
        selected = recipe.selections[name][0]
        if not name.startswith('text/layers/') or not parameter.projection:
            continue
        if name.endswith(('/a_projection', '/b_projection')):
            assert (selected.format, selected.method) == ('bf16', cast_direct)
            assert selected.source is not parameter.source
        else:
            assert (selected.format, selected.method) == ('nvfp4', import_encoded)
            assert all(recipe.policies[(name, input_name)] == 'AllowA4' for input_name in parameter.inputs)


def test_nvfp4_module_override_31_parents():
    from tools.convert.recipes.dflash2_nvfp4 import configure
    from tools.convert.official_recipes import _optional
    config = dict(num_hidden_layers=5, head_dim=128, num_attention_heads=32,
                  num_key_value_heads=8, intermediate_size=17408,
                  dflash_config=dict(target_layer_ids=[1, 2, 3, 4, 5], conv_kernel_size=2,
                                     conv_group_size=16, selector_rank=256))
    model = Model({'text': {'config': {'hidden_size': 5120, 'vocab_size': 248320}},
                   'dflash2': {'config': config}})
    _Builder(model).draft(SimpleNamespace(path='synthetic', config={}), config, model.config, 'dflash2')
    recipe = Recipe(model)
    _optional(model, recipe)
    configure(model, recipe, {})
    _synthetic_sources(recipe)
    prepared = recipe.prepare(device='cpu')
    assert sum(job.spec.format == 'nvfp4' for job in prepared.weights) == 31
    assert recipe.selections['dflash2/feature_projection'][0].format == 'q8_g32_fp16'
    assert recipe.selections['dflash2/candidate_selector/predecessor_codebook'][0].format == 'bf16'
    assert all(use['activation_policy'] == 'A16Only' for use in prepared.uses)
    for i in range(5):
        prefix = f'dflash2/layers/{i}/attention/'
        assert prepared.bindings[prefix+'context_key'] == prepared.bindings[prefix+'key']


def test_e2m1_exact_ties_and_signed_zero():
    values = torch.tensor([0., -0., .25, .75, 1.25, 1.75, 2.5, 3.5, 5., 7., -.75])
    assert e2m1_rne_codes(values).tolist() == [0, 8, 0, 2, 2, 4, 4, 6, 6, 7, 10]


def test_import_words_divisors_and_control_decode(tmp_path):
    from safetensors.torch import save_file
    from tools.convert.sources.safetensors import SafetensorsSource
    from tools.convert.sources.compressed_tensors import compressed_matrix_source
    from tools.convert.methods import MethodInput, PrepareRequest
    from tools.artifact.schema import TensorSpec
    packed = torch.arange(256, dtype=torch.uint8).reshape(128, 2).repeat(1, 8)
    scales = torch.full((128, 2), 0x38, dtype=torch.uint8)
    path = tmp_path / 'encoded.safetensors'
    save_file({'matrix.weight_packed': packed,
               'matrix.weight_scale': scales.view(torch.float8_e4m3fn),
               'matrix.weight_global_scale': torch.tensor(2., dtype=torch.float32),
               'matrix.input_global_scale': torch.tensor(7., dtype=torch.float32)}, path)
    with SafetensorsSource(path) as store:
        source = compressed_matrix_source(store, 'matrix', (128, 32), 'nvfp4')
        use = ('matrix', 'input')
        spec = TensorSpec('parent', (128, 32), 'nvfp4', 'block_scale_k16_m128x4_v1')
        request = PrepareRequest(spec, (MethodInput('matrix', source, (use,)),), {use: 'AllowA4'}, {}, device='cpu')
        job = import_encoded(request)
        result = []
        job.produce(SimpleNamespace(write_codes=lambda start, c, s, d: result.append((start, c, s, d))))
        assert torch.equal(result[0][1], packed) and torch.equal(result[0][2], scales)
        assert result[0][3] == struct.pack('<f', 2.)
        assert job.auxiliaries[(*use, 'activation_input_divisor')].data == struct.pack('<f', 7.)
        # Exact scalar decode oracle, including signed codes; BF16 boundary is explicit.
        magnitudes = (0., .5, 1., 1.5, 2., 3., 4., 6.)
        expected = torch.tensor([(-1 if code & 8 else 1) * magnitudes[code & 7] / 2.
                                 for byte in packed.flatten().tolist() for code in (byte & 15, byte >> 4)], dtype=torch.bfloat16)
        direct = TensorSpec('controls', (128, 32), 'bf16', 'contiguous_le_v1')
        request = PrepareRequest(direct, (MethodInput('controls', source, ()),), {}, {}, device='cpu')
        decoded = []
        cast_direct(request).produce(SimpleNamespace(write_values=lambda begin, values: decoded.append(values.flatten())))
        assert torch.equal(torch.cat(decoded), expected)


def test_global_parent_divisor_independent_of_chunk():
    from tools.convert.methods import MethodInput, PrepareRequest
    from tools.artifact.schema import TensorSpec
    # Unequal slice maxima: quantizing each logical input independently is incorrect.
    values = torch.zeros(256, 16)
    values[:128] = 1.
    values[128:] = 6.
    sources = [LogicalSource((128, 16), str(i), lambda a, b, i=i: values[i*128:(i+1)*128].reshape(-1)[a:b]) for i in range(2)]
    spec = TensorSpec('parent', (256, 16), 'nvfp4', 'block_scale_k16_m128x4_v1')
    def run(chunk):
        result = []
        request = PrepareRequest(spec, tuple(MethodInput(str(i), src, ()) for i, src in enumerate(sources)), {}, {}, device='cpu', rows_per_chunk=chunk)
        nvfp4_maxabs(request).produce(SimpleNamespace(write_codes=lambda start, c, s, d: result.append((c, s, d))))
        return torch.cat([x[0] for x in result]), torch.cat([x[1] for x in result]), {x[2] for x in result}
    a, b = run(128), run(512)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    assert a[2] == b[2] == {struct.pack('<f', 448.)}
    # Independent nearest representable E2M1 oracle, ties choose even code.
    grid = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.], dtype=torch.float64)
    scales = a[1].view(torch.float8_e4m3fn).double().repeat_interleave(16, 1)
    ratio = values.double() * 448. / scales
    distances = (ratio[..., None] - grid).abs()
    nearest = distances.min(-1, keepdim=True).values
    priority = torch.arange(8) + (torch.arange(8) % 2) * 8
    codes = torch.where(distances == nearest, priority, 100).argmin(-1).to(torch.uint8)
    assert torch.equal(a[0], codes[:, 0::2] | (codes[:, 1::2] << 4))
