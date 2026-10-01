import os

import torch

from tools.artifact.reader import Artifact
from tools.artifact.writer import ArtifactWriter
from tools.artifact.schema import TensorSpec, ResourceSpec
from tools.convert.model import Model, Parameter
from tools.convert.recipe import Recipe
from tools.convert.methods import grouped_absmax
from tools.convert.sources.logical import LogicalSource
from tools.convert.qwen3_8_27b.splice_w8_module import splice_prepared


def test_positional_binary_read_preserves_position(tmp_path):
    path = tmp_path / 'binary'
    path.write_bytes(bytes(range(256)))
    fd = os.open(path, os.O_RDONLY)
    try:
        os.lseek(fd, 9, os.SEEK_SET)
        assert os.pread(fd, 32, 16) == bytes(range(16, 48))
        assert os.lseek(fd, 0, os.SEEK_CUR) == 9
        assert os.pread(fd, 10, 255) == b'\xff'
        assert os.pread(fd, 10, 256) == b''
        assert os.lseek(fd, 0, os.SEEK_CUR) == 9
    finally:
        os.close(fd)


def test_module_splice_preserves_base_bytes_and_bindings(tmp_path):
    name = 'dflash2/feature_projection'
    values = torch.arange(256, dtype=torch.float32).reshape(2, 128) / 17
    model = Model({'text': {'config': {}}, 'dflash2': {'config': {}}})
    model.add(Parameter(name, (2, 128), LogicalSource((2, 128), 'bf16', lambda a, b: values.flatten()[a:b]), inputs=('dflash2/input',)))
    recipe = Recipe(model)
    recipe.assign(name, format='q8_g32_fp16', method=grouped_absmax)
    prepared = recipe.prepare(device='cpu')
    specs = [ResourceSpec('resource', 10001), TensorSpec('base', (256,), 'bf16', 'contiguous_le_v1'),
             TensorSpec('old-module', (2, 128), 'bf16', 'contiguous_le_v1'),
             TensorSpec('old-aux', (), 'fp32', 'contiguous_le_v1')]
    base_bindings = {'text/base': {'object': 'base'}}
    bindings = {**base_bindings, name: {'object': 'old-module'}}
    uses = [{'parameter': 'text/base', 'input': 'text/input', 'activation_policy': 'A16Only'},
            {'parameter': name, 'input': 'dflash2/input', 'activation_policy': 'A16Only',
             'auxiliaries': {'activation_input_divisor': {'object': 'old-aux'}}}]
    payloads = {'resource': bytes(range(251)) * 39 + bytes(212), 'base': bytes(range(256))*2,
                'old-module': bytes(512), 'old-aux': b'\0\0\x80?'}
    components = {'text': {'config': {}, 'resources': {'tokenizer': 'resource'}}, 'dflash2': {'config': {}}}
    source_path, out = tmp_path / 'source.ninfer', tmp_path / 'out.ninfer'
    with ArtifactWriter(source_path, specs, components=components, bindings=bindings, uses=uses,
                        metadata={'name': 'keep'}, max_file_bytes=12288) as writer:
        for key, data in payloads.items():
            writer.write_region(key, 0, data)
    with Artifact.open(source_path) as source:
        splice_prepared(source, prepared, out, max_file_bytes=12288)
    with Artifact.open(out) as result:
        for key in ('resource', 'base'):
            assert result.read_object(key) == payloads[key]
        assert result.directory.bindings['text/base'] == base_bindings['text/base']
        assert result.directory.uses[0] == uses[0]
        assert result.directory.components == components
        assert result.directory.metadata == {'name': 'keep'}
        assert 'old-module' not in result.by_id and 'old-aux' not in result.by_id
        replacement = result.directory.bindings[name]['object']
        assert result.by_id[replacement].format == 'q8_g32_fp16'
        assert len(result.directory.files) > 1
