#!/usr/bin/env python3
"""Rebuild a cumulative curated stack into a NEW ref, without checkout or staging.

Run with uv run --python 3.11 --no-project tools/maintainer/rebuild_dev.py --help.
All input refs are pinned to commits before work. Only committed overlay files are
read; dirty/untracked files are never captured. No hooks, merges, or pushes run.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import tempfile


class RebuildError(RuntimeError):
    pass


class Git:
    def __init__(self, repo: Path, index: str | None = None):
        self.repo = repo
        self.env = os.environ.copy()
        # Never inherit an alternate caller index, object store, or repository.
        for key in list(self.env):
            if key.startswith("GIT_"):
                del self.env[key]
        self.env["GIT_NO_REPLACE_OBJECTS"] = "1"
        if index:
            self.env["GIT_INDEX_FILE"] = index

    def run(self, *args: str, data: bytes | None = None) -> bytes:
        result = subprocess.run(["git", "-C", str(self.repo), *args], input=data,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                env=self.env, check=False)
        if result.returncode:
            raise RebuildError(f"git {' '.join(args)}: {result.stderr.decode(errors='replace').strip()}")
        return result.stdout

    def commit(self, ref: str) -> str:
        return self.run("rev-parse", "--verify", "--end-of-options", ref + "^{commit}").decode().strip()

    def tree(self, ref: str) -> dict[bytes, bytes]:
        entries = self.run("ls-tree", "-rz", "--full-tree", ref).split(b"\0")
        return dict((entry.split(b"\t", 1)[1], entry.split(b"\t", 1)[0])
                    for entry in entries if entry)


def changed(a: dict, b: dict) -> set:
    return {path for path in a.keys() | b.keys() if a.get(path) != b.get(path)}


def paths_text(paths: set[bytes]) -> str:
    return ", ".join(os.fsdecode(p) for p in sorted(paths))


def update(git: Git, before: dict, after: dict, paths: set[bytes]) -> None:
    # Two passes permit file <-> directory transitions without checkout filters.
    removed = b"".join(b"0 " + b"0" * 40 + b"\t" + p + b"\0"
                       for p in sorted(paths) if p in before)
    added = b"".join(after[p] + b"\t" + p + b"\0" for p in sorted(paths) if p in after)
    for records in (removed, added):
        if records:
            git.run("update-index", "-z", "--index-info", data=records)


def manifest_paths(values: list[str]) -> set[bytes]:
    if not isinstance(values, list) or not all(isinstance(p, str) for p in values):
        raise RebuildError("Manifest paths must be an explicit list of file paths")
    for p in values:
        if (not p or PurePosixPath(p).is_absolute() or "\\" in p
                or any(part in ("", ".", "..") for part in p.split("/"))
                or any(c in p for c in "*?[]\0\n\r:") or p.startswith(".git/")):
            raise RebuildError(f"Not an exact repository-relative file path: {p!r}")
    if len(values) != len(set(values)):
        raise RebuildError("Duplicate manifest path")
    return {p.encode("utf-8") for p in values}


FEATURE_START = b"<!-- ninfer:features:start -->"
FEATURE_END = b"<!-- ninfer:features:end -->"
FEATURE_HEADER = (b"| feat branch | stacked on | status | squashed on dev as |\n"
                  b"|---|---|---|---|\n")
EMPTY_FEATURE_BLOCK = FEATURE_START + b"\n" + FEATURE_HEADER + FEATURE_END
README = b"README.md"


def readme_parts(data: bytes) -> tuple[bytes, bytes, bytes]:
    """Split one exact feature-table block without normalizing committed bytes."""
    if data.count(FEATURE_START) != 1 or data.count(FEATURE_END) != 1:
        raise RebuildError("README must contain exactly one marked feature table")
    before, rest = data.split(FEATURE_START)
    inside, after = rest.split(FEATURE_END)
    prefix = b"\n" + FEATURE_HEADER
    if not inside.startswith(prefix):
        raise RebuildError("README feature table has an unexpected header")
    rows = inside[len(prefix):]
    if rows and (not rows.endswith(b"\n") or any(
            not row.startswith(b"| ") or not row.endswith(b" |") or row.count(b"|") != 5
            for row in rows.splitlines())):
        raise RebuildError("README feature block must contain only four-column table rows")
    return before, FEATURE_START + inside + FEATURE_END, after


def rebuild(repo: Path, manifest: Path, base_ref: str, overlay_ref: str,
            features: list[tuple[str, str]], output: str) -> str:
    git = Git(repo)
    if not output.startswith("refs/heads/"):
        raise RebuildError("Output must be a new full refs/heads/... name")
    git.run("check-ref-format", output)
    if git.run("for-each-ref", "--format=%(refname)", output).strip():
        raise RebuildError(f"Output already exists: {output}")
    if not features or len({name for name, _ in features}) != len(features):
        raise RebuildError("Supply a nonempty ordered stack with unique feature names")
    if any(not name or any(c in name for c in "\n\r:") for name, _ in features):
        raise RebuildError("Invalid feature name")
    config = json.loads(manifest.read_text(encoding="utf-8"))
    owned = manifest_paths(config["overlay_paths"])
    excluded = manifest_paths(config["excluded_from_overlay"])
    readme_sections = config.get("readme_feature_table", False)
    if not isinstance(readme_sections, bool):
        raise RebuildError("readme_feature_table must be boolean")
    if readme_sections and README not in owned:
        raise RebuildError("Section-owned README must be inventoried in overlay_paths")
    if owned & excluded:
        raise RebuildError("Overlay and excluded paths overlap")
    baseline = git.commit(config["baseline"])
    base = git.commit(base_ref)
    overlay = git.commit(overlay_ref)
    tips = [(name, git.commit(ref)) for name, ref in features]
    # Audit the historical fork base rather than trusting an obsolete whitelist.
    baseline_parents = git.run("rev-list", "--parents", "-n", "1", baseline).split()
    if len(baseline_parents) != 2:
        raise RebuildError("Overlay baseline must have exactly one parent")
    inventory = changed(git.tree(baseline_parents[1].decode()), git.tree(baseline))
    missing = inventory - owned - excluded
    if missing:
        raise RebuildError("Unclassified fork-base paths: " + paths_text(missing))
    # The baseline audits ownership, not ancestry. A reconstructed dev deliberately
    # descends from the new upstream base rather than the historical fork baseline.
    # Its explicitly selected committed tree remains the overlay authority; only
    # inventoried paths are copied and their exact blobs/modes are verified below.
    previous = base
    for name, tip in tips:
        if tip == previous:
            raise RebuildError(f"Empty cumulative ancestry step: {name}")
        git.run("merge-base", "--is-ancestor", previous, tip)
        previous = tip

    base_tree, overlay_tree = git.tree(base), git.tree(overlay)
    # Manifest entries name files, never a whole directory/subtree.
    all_paths = set(base_tree) | set(overlay_tree)
    for _, tip in tips:
        all_paths.update(git.tree(tip))
    if any(any(p.startswith(owner + b"/") for p in all_paths) for owner in owned):
        raise RebuildError("Overlay manifest names a directory; enumerate its files")

    readme_entries = {}
    if readme_sections:
        def contents(tree: dict) -> bytes:
            entry = tree.get(README, b"")
            if not entry.startswith(b"100644 blob "):
                raise RebuildError("Section-owned README must be a regular 100644 file")
            return git.run("cat-file", "blob", entry.split()[-1].decode())

        upstream_readme = contents(base_tree)
        if FEATURE_START in upstream_readme or FEATURE_END in upstream_readme:
            raise RebuildError("Upstream README already contains a fork feature marker")
        shell_before, _, shell_after = readme_parts(contents(overlay_tree))

        def composed(block: bytes) -> bytes:
            blob = git.run("hash-object", "-w", "--stdin",
                           data=shell_before + block + shell_after).strip()
            return b"100644 blob " + blob

        readme_entries[base] = composed(EMPTY_FEATURE_BLOCK)
        for name, tip in tips:
            source = contents(git.tree(tip))
            if source == upstream_readme:
                block = EMPTY_FEATURE_BLOCK
            else:
                before, block, after = readme_parts(source)
                # A feature branch appends only the marked table, not fork boilerplate.
                if before != upstream_readme + b"\n" or after != b"\n":
                    raise RebuildError(f"Feature {name} changes README outside the feature table")
            readme_entries[tip] = composed(block)

    with tempfile.TemporaryDirectory(prefix="ninfer-rebuild-") as directory:
        scratch = Git(repo, str(Path(directory) / "index"))
        scratch.run("read-tree", base)
        expected = dict(base_tree)
        for path in owned:
            expected.pop(path, None)
            if path in overlay_tree:
                expected[path] = overlay_tree[path]
        if readme_sections:
            expected[README] = readme_entries[base]
        update(scratch, base_tree, expected, changed(base_tree, expected))

        def snapshot(parent: str, message: str, wanted: dict) -> str:
            tree = scratch.run("write-tree").decode().strip()
            differences = changed(scratch.tree(tree), wanted)
            if differences:
                raise RebuildError("Exact tree verification failed: " + paths_text(differences))
            return scratch.run("commit-tree", tree, "-p", parent,
                               data=(message + "\n").encode()).decode().strip()

        parent = snapshot(base, f"chore(fork): overlay from {overlay}", expected)
        previous_tree = base_tree
        for name, tip in tips:
            feature_tree = git.tree(tip)
            delta = changed(previous_tree, feature_tree)
            if readme_sections:
                delta.discard(README)
            if delta & owned:
                raise RebuildError(f"Feature {name} changes fork-owned paths: " + paths_text(delta & owned))
            update(scratch, previous_tree, feature_tree, delta)
            for path in delta:
                expected.pop(path, None)
                if path in feature_tree:
                    expected[path] = feature_tree[path]
            if readme_sections:
                entry = readme_entries[tip]
                update(scratch, expected, {README: entry}, {README})
                expected[README] = entry
            parent = snapshot(parent, f"squash(feat/{name}): adjacent delta through {tip}", expected)
            previous_tree = feature_tree
        actual = scratch.tree(parent)
        feature_errors = changed(actual, previous_tree) - owned
        overlay_errors = changed(actual, overlay_tree) & owned
        if readme_sections:
            overlay_errors.discard(README)
            if actual.get(README) != readme_entries[tips[-1][1]]:
                overlay_errors.add(README)
        if feature_errors or overlay_errors:
            raise RebuildError("Final ownership parity failed: " + paths_text(feature_errors | overlay_errors))
        # Atomic create-only transaction also rejects a concurrent creator.
        git.run("update-ref", "--stdin", data=f"create {output} {parent}\n".encode())
    return parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("fork_overlay.json"))
    parser.add_argument("--base", required=True, help="upstream commit preceding the first cumulative feature")
    parser.add_argument("--overlay-source", required=True, help="explicit committed overlay authority (including previously rebuilt dev); never the working tree")
    parser.add_argument("--feature", action="append", required=True, metavar="NAME=REF", help="repeat in cumulative ancestor order")
    parser.add_argument("--output-ref", required=True, help="new refs/heads/...; existing refs are never overwritten")
    args = parser.parse_args()
    try:
        features = []
        for value in args.feature:
            name, separator, ref = value.partition("=")
            if not separator or not ref:
                raise RebuildError("Feature must be NAME=REF")
            features.append((name, ref))
        result = rebuild(args.repo, args.manifest, args.base, args.overlay_source, features, args.output_ref)
    except (RebuildError, OSError, ValueError, KeyError) as error:
        parser.exit(1, f"rebuild refused: {error}\n")
    print(f"Created {args.output_ref} at {result}; exact feature/overlay parity verified.")


if __name__ == "__main__":
    main()
