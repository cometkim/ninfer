#!/usr/bin/env python3
"""Fork extension of the one-time v2->v3 upgrade: admits the fork-registered weight
profiles (qwen3.8-27b/nvfp4full and qwen3.8-27b/nvfp4qat, including their NVFP4-encoded
DFlash2 modules) on the same object-name contract as the official inputs. Run:
python tools/upgrade_ninfer_v2_to_v3_fork.py INPUT.ninfer OUTPUT.ninfer
"""

from __future__ import annotations

import sys

import tools.upgrade_ninfer_v2_to_v3 as base

base.KNOWN_COUNTS[("qwen3.8-27b", "nvfp4full")] = (1325,)
base.KNOWN_COUNTS[("qwen3.8-27b", "nvfp4qat")] = (1334,)


def main() -> None:
    argv = sys.argv[:]
    argv[0] = base.__file__
    sys.argv = argv
    base.main()


if __name__ == "__main__":
    main()
