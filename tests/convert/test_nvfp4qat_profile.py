import json
import struct
from types import SimpleNamespace

import unittest

import torch

from tools.convert.model import Model, Parameter
from tools.convert.recipe import Recipe
from tools.convert.methods import cast_direct, grouped_absmax, import_encoded
from tools.convert.sources.logical import LogicalSource
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
    for name in ('text/token_embedding', 'text/output_head'):
        selected = recipe.selections[name][0]
        assert (selected.format, selected.method) == ('q8_g32_fp16', grouped_absmax)
    _synthetic_sources(recipe)
    prepared = recipe.prepare(device='cpu')
    assert sum(job.spec.format == 'nvfp4' for job in prepared.weights) == 256
    assert sum(job.spec.format == 'bf16' and job.spec.shape == (96, 5120)
               for job in prepared.weights) == 48
    assert all(use['activation_policy'] == 'AllowA4' for use in prepared.uses
               if use['parameter'].startswith('text/layers/')
               and not use['parameter'].endswith(('/a_projection', '/b_projection')))


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


if __name__ == '__main__':
    import inspect
    import tempfile
    from pathlib import Path
    for name, function in list(globals().items()):
        if name.startswith('test_'):
            with tempfile.TemporaryDirectory() as directory:
                function(Path(directory)) if inspect.signature(function).parameters else function()
            print(name, 'PASS')
