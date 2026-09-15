"""Replace only DFlash2 in a v3 artifact from its BF16 companion checkpoint.

python -m tools.convert.qwen3_8_27b.splice_w8_module SOURCE.ninfer \
  --dflash2-model CHECKPOINT --out OUTPUT.ninfer --device cpu

Every non-module object is streamed byte-exact, retaining its descriptor,
binding, Use and metadata. No base checkpoint, target inventory or base
re-quantization is needed. Source and destination must be distinct new files.
"""
from copy import deepcopy
from dataclasses import replace
import argparse
from pathlib import Path

from tools.artifact.reader import Artifact
from tools.artifact.schema import ResourceSpec, TensorSpec
from tools.artifact.tensor_output import TensorOutput
from tools.artifact.writer import ArtifactWriter
from tools.convert.model import Model
from tools.convert.official_recipes import _optional
from tools.convert.qwen3_5 import _Builder, draft_config
from tools.convert.recipe import Recipe
from tools.convert.sources.safetensors import SafetensorsSource


def _references(value):
    if isinstance(value, dict):
        if 'object' in value:
            yield value['object']
        for child in value.values():
            yield from _references(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _references(child)


def _rename(value, names):
    if isinstance(value, dict):
        return {key: names[child] if key == 'object' else _rename(child, names)
                for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_rename(child, names) for child in value]
    return value


def splice_prepared(source, prepared, out_path, *, max_file_bytes=32_000_000_000):
    """Splice a prepared DFlash2-only recipe; shared base parents stay untouched."""
    directory = source.directory
    is_module = lambda name: name.startswith('dflash2/')
    if not prepared.bindings or any(not is_module(name) for name in prepared.bindings):
        raise ValueError('replacement must contain only DFlash2 logical parameters')
    previous = {name for name in directory.bindings if is_module(name)}
    if previous != set(prepared.bindings):
        raise ValueError('replacement DFlash2 logical parameter set differs')
    bindings = {name: value for name, value in directory.bindings.items() if not is_module(name)}
    uses = [use for use in directory.uses if not is_module(use['parameter'])]
    module_refs = set(_references([value for name, value in directory.bindings.items() if is_module(name)]))
    module_refs.update(_references([use for use in directory.uses if is_module(use['parameter'])]))
    retained_refs = set(_references([bindings, uses, directory.components]))
    removed = module_refs - retained_refs
    retained = [obj for obj in source.objects if obj.id not in removed]
    specs = [TensorSpec(obj.id, obj.shape, obj.format, obj.layout) if obj.kind == 'tensor'
             else ResourceSpec(obj.id, obj.bytes, obj.encoding) for obj in retained]
    jobs = list(prepared.weights)
    additions = [job.spec for job in jobs] + [spec for spec, _ in prepared.auxiliaries]
    names = {spec.id: f'splice/dflash2/{index:06d}' for index, spec in enumerate(additions)}
    if set(names.values()) & {obj.id for obj in retained}:
        raise ValueError('replacement object IDs collide with retained objects')
    specs.extend(replace(spec, id=names[spec.id]) for spec in additions)
    bindings.update(_rename(prepared.bindings, names))
    uses.extend(_rename(prepared.uses, names))
    provenance = deepcopy(directory.provenance)
    provenance['dflash2_splice'] = {'method': 'q8_g32_fp16', 'source': str(source.path)}
    with ArtifactWriter(out_path, specs, components=directory.components,
                        bindings=bindings, uses=uses, metadata=directory.metadata,
                        provenance=provenance, max_file_bytes=max_file_bytes) as writer:
        for obj in retained:
            offset = 0
            for block in source.iter_object(obj.id):
                writer.write_region(obj.id, offset, block)
                offset += len(block)
        for job in jobs:
            job.prepared.produce(TensorOutput(writer, names[job.spec.id]))
        for spec, data in prepared.auxiliaries:
            writer.write_region(names[spec.id], 0, data)


def splice(source_path, dflash2_dir, out_path, *, device='cpu', rows_per_chunk=512,
           max_file_bytes=32_000_000_000):
    with Artifact.open(source_path) as source, SafetensorsSource(dflash2_dir) as store:
        target = source.directory.components['text']['config']
        config = draft_config(store.config, target, 'dflash2')
        if config != source.directory.components['dflash2']['config']:
            raise ValueError('companion configuration differs from resident DFlash2')
        model = Model({name: deepcopy(source.directory.components[name])
                       for name in ('text', 'dflash2')})
        _Builder(model).draft(store, config, target, 'dflash2')
        recipe = Recipe(model)
        _optional(model, recipe)
        prepared = recipe.prepare(device=device, rows_per_chunk=rows_per_chunk)
        splice_prepared(source, prepared, out_path, max_file_bytes=max_file_bytes)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--dflash2-model', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--rows-per-chunk', type=int, default=512)
    parser.add_argument('--max-file-bytes', type=int, default=32_000_000_000)
    args = parser.parse_args(argv)
    splice(args.source, args.dflash2_model, args.out, device=args.device,
           rows_per_chunk=args.rows_per_chunk, max_file_bytes=args.max_file_bytes)


if __name__ == '__main__':
    main()
