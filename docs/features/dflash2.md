# Fork DFlash2 NVFP4 integration

This cumulative branch adds weight-only NVFP4 DFlash2 routes to the Qwen3.8-27B engine.
Its scope includes projection, context materialization, dynamic convolution and selector Ops,
plus full/QAT conversion recipes, a module override and a W8 module-splice tool.
Full/QAT v2 migration uses the standard `tools/upgrade_ninfer_v2_to_v3.py` entry point.

For an artifact containing the companion component, select `--spec dflash2 --draft-tokens 7`.
The draft-count range is `1..15`; component storage and runtime selection are separate choices.
See [DFlash semantics](../maintainer/dflash.md) and [conversion recipes](../../tools/convert/recipes/README.md).

Runtime support, conversion allocation and artifact provenance must be checked separately.
Inspect the recipe's component allocation before treating it as reproduction of an artifact.
