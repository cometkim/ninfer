"""Splice the upstream W8G32_F16S DFlash2 module into an existing fork image.

Rebuilds a v2 nvfp4full/nvfp4qat image (fork NVFP4 module) as the upstream-schema
v3/v2 form: every non-`dflash2/` object is copied byte-for-byte from the source
artifact, and the 66 companion-bundle objects are encoded fresh from the fixed
z-lab source through the same dflash2_recipe machinery the registered profiles
use. The base payloads are provably unchanged; only the module encoding moves.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time
from typing import Sequence

import torch

from tools.artifact.container import Artifact, ArtifactIdentity, ArtifactWriter
from tools.convert.common.safetensors import ShardReader
from tools.convert.qwen3_6.common import conversion as family_conversion

from . import dflash2_recipe
from . import inventory_nvfp4full as full_inventory
from . import inventory_nvfp4qat as qat_inventory
from . import recipe_nvfp4full as full_recipe


def _splice(
    source_path: Path,
    dflash2_dir: Path,
    out_path: Path,
    inventory,
    recipe,
    recipe_id: str,
    device: torch.device,
) -> None:
    module_names = {spec.name for spec in inventory.DFLASH2_TENSOR_SPECS}
    with Artifact.open(source_path) as source:
        base_names = [o.name for o in source.objects if not o.name.startswith("dflash2/")]
        expected_base = len(inventory.OBJECT_SPECS) - len(module_names)
        if len(base_names) != expected_base:
            raise ValueError(
                f"{source_path}: expected {expected_base} base objects, found {len(base_names)}"
            )
        print(f"source: {len(base_names)} base objects + {len(module_names)} module objects")

        resources = {
            spec.name: bytes(source.payload(spec.name))
            for spec in inventory.OBJECT_SPECS
            if isinstance(spec, inventory.ResourceSpec)
        }
        object_plan = family_conversion.build_object_plan(inventory.OBJECT_SPECS, resources)

        with ShardReader.from_file(dflash2_dir / "model.safetensors") as module_reader:
            dflash2_recipe.validate_recipe_coverage()
            with ArtifactWriter(
                out_path,
                ArtifactIdentity(inventory.MODEL_ID, inventory.WEIGHTS_ID),
                object_plan.specs,
            ) as writer:
                index = 0
                for spec in inventory.OBJECT_SPECS:
                    index += 1
                    if isinstance(spec, inventory.ResourceSpec):
                        payload = resources[spec.name]
                    elif spec.name in module_names:
                        tensor = dflash2_recipe.materialize_tensor(spec.name, module_reader)
                        if tuple(tensor.shape) != spec.shape:
                            raise ValueError(
                                f"{spec.name}: source shape {tuple(tensor.shape)} != {spec.shape}"
                            )
                        payload = family_conversion.encode_tensor_payload(tensor, spec, device)
                        del tensor
                    else:
                        payload = bytes(source.payload(spec.name))
                    writer.write(spec.name, payload)
                    del payload
                    if index % 128 == 0 or index == len(inventory.OBJECT_SPECS):
                        print(f"[{index}/{len(inventory.OBJECT_SPECS)}] objects written",
                              flush=True)
    print(f"complete: {out_path} ({out_path.stat().st_size} bytes, recipe {recipe_id})")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="existing fork-module image (v2)")
    parser.add_argument("--dflash2-model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--profile", choices=("nvfp4full", "nvfp4qat"), default="nvfp4full")
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args(argv)

    started = time.perf_counter()
    if arguments.profile == "nvfp4full":
        _splice(
            arguments.source,
            arguments.dflash2_model,
            arguments.out,
            full_inventory,
            full_recipe,
            "qwen3_8_27b_nvfp4full-v3",
            torch.device(arguments.device),
        )
    else:
        from . import inventory_nvfp4qat, recipe_nvfp4qat

        _splice(
            arguments.source,
            arguments.dflash2_model,
            arguments.out,
            inventory_nvfp4qat,
            recipe_nvfp4qat,
            "qwen3_8_27b_nvfp4qat-v2",
            torch.device(arguments.device),
        )
    print(f"elapsed {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
