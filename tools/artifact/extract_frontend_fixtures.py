"""Extract the frontend tokenizer trio from a .ninfer artifact (Q5 fixture source).

Reads the v2 container per docs/maintainer/artifact-container.md: 16-byte prefix
(magic + LE64 json_bytes), JSON directory at offset 16, payload at
align_up(16 + json_bytes, 4096); writes each `frontend/<name>` resource under --out.
"""
import argparse
import json
import struct
import sys
from pathlib import Path

WANT = {"tokenizer.json", "tokenizer_config.json", "generation_config.json"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()

    with open(args.artifact, "rb") as handle:
        prefix = handle.read(16)
        if prefix[:8] != b"NINFER\x00\x02":
            print("not a v2 .ninfer artifact", file=sys.stderr)
            return 2
        (json_bytes,) = struct.unpack_from("<Q", prefix, 8)
        directory = json.loads(handle.read(json_bytes))
        metadata_end = 16 + json_bytes
        payload_offset = (metadata_end + 4095) & ~4095
        written = []
        for obj in directory["objects"]:
            if obj.get("kind") != "resource":
                continue
            name = obj["name"].removeprefix("frontend/")
            if name not in WANT:
                continue
            handle.seek(payload_offset + obj["offset"])
            data = handle.read(obj["bytes"])
            target = args.out / name
            target.write_bytes(data)
            written.append((name, obj["bytes"]))
    for name, size in sorted(written):
        print(f"{name}: {size} bytes")
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
