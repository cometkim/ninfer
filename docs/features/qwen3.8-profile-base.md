# Shared Qwen3.8 profile allocation

This helper is the common prerequisite for the sibling full and QAT recipes.
It requires a 64-layer dense model, assigns Q8 vocabulary endpoints,
uses upstream optional-component allocation and groups related Text projections.
The implementation is [qwen3_8_profile.py](../../tools/convert/recipes/qwen3_8_profile.py).

It is not a standalone conversion recipe and selects no full/QAT weight encoding.
Call its helpers from an owning recipe under the [conversion contract](../weight-conversion.md).
It introduces no runtime identity, artifact registry or DFlash execution route.
