#!/usr/bin/env python3
"""Exact rebuild regression checks; all Git objects/refs live in a temporary repo."""
import json
from pathlib import Path
import tempfile
import unittest

from rebuild_dev import (EMPTY_FEATURE_BLOCK, FEATURE_END, Git, RebuildError,
                         readme_parts, rebuild, update)


class RebuildTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="ninfer-rebuild-test-")
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name)
        self.git = Git(self.repo)
        self.git.run("init", "--quiet")
        self.git.run("config", "user.name", "Rebuild test")
        self.git.run("config", "user.email", "rebuild-test@example.invalid")
        self.git.run("config", "commit.gpgsign", "false")
        self.index = Git(self.repo, str(self.repo / "scratch-index"))
        self.base = self.commit(None, {
            "engine": ("100644", b"base\n"),
            "rename-old": ("100644", b"rename\n"),
            "delete": ("100644", b"delete\n"),
            "node": ("100644", b"file becomes directory\n"),
            "README.md": ("100644", b"upstream\n"),
        })
        self.baseline = self.commit(self.base, {
            "README.md": ("100644", b"fork\n"),
            "preset.bat": ("100644", b"old preset\n"),
            "local-only": ("100644", b"must not overlay\n"),
        })
        self.overlay = self.commit(self.baseline, {
            "preset.bat": ("100644", b"updated preset\n"),
            "HANDOFF.md": ("100644", b"session\n"),
            "engine": ("100644", b"dev-only fix must not leak\n"),
        })
        self.one = self.commit(self.base, {
            "engine": ("100755", b"feature one\n"),
            "rename-old": None,
            "rename-new": ("100644", b"rename\n"),
            "binary": ("100644", b"\0\xff\r\n\x01"),
            "link": ("120000", b"engine"),
        })
        self.two = self.commit(self.one, {
            "engine": ("100755", b"feature two\n"),
            "delete": None,
            "node": None,
            "node/child": ("100644", b"directory transition\n"),
            "local-only": ("100644", b"feature-owned, not overlay\n"),
        })
        self.manifest = self.repo / "manifest.json"
        self.config = {"baseline": self.baseline,
                       "overlay_paths": ["README.md", "preset.bat", "HANDOFF.md"],
                       "excluded_from_overlay": ["local-only"]}
        self.write_manifest()
        self.git.run("update-ref", "refs/heads/main", self.overlay)
        self.git.run("symbolic-ref", "HEAD", "refs/heads/main")
        self.git.run("read-tree", self.overlay)
        (self.repo / "engine").write_bytes(b"dirty untracked/working content\n")
        blob = self.git.run("hash-object", "-w", "--stdin", data=b"dirty staged\n").strip()
        self.git.run("update-index", "--cacheinfo", "100644", blob.decode(), "engine")

    def write_manifest(self):
        self.manifest.write_text(json.dumps(self.config), encoding="utf-8")

    def commit(self, parent, changes):
        before = self.git.tree(parent) if parent else {}
        after = dict(before)
        for name, item in changes.items():
            path = name.encode()
            after.pop(path, None)
            if item:
                mode, content = item
                blob = self.git.run("hash-object", "-w", "--stdin", data=content).strip()
                after[path] = mode.encode() + b" blob " + blob
        self.index.run("read-tree", parent) if parent else self.index.run("read-tree", "--empty")
        update(self.index, before, after, set(before) | set(after))
        tree = self.index.run("write-tree").decode().strip()
        args = ["commit-tree", tree]
        if parent:
            args += ["-p", parent]
        return self.git.run(*args, data=b"fixture\n").decode().strip()

    def rebuild(self, features=None, output="refs/heads/rebuilt"):
        return rebuild(self.repo, self.manifest, self.base, self.overlay,
                       features or [("one", self.one), ("two", self.two)], output)

    def test_exact_trees_history_and_dirty_checkout_untouched(self):
        head = self.git.run("rev-parse", "HEAD")
        index = (self.repo / ".git/index").read_bytes()
        work = (self.repo / "engine").read_bytes()
        result = self.rebuild()
        owned = {p.encode() for p in self.config["overlay_paths"]}
        expected = {p: entry for p, entry in self.git.tree(self.two).items() if p not in owned}
        expected.update({p: entry for p, entry in self.git.tree(self.overlay).items() if p in owned})
        self.assertEqual(self.git.tree(result), expected)
        self.assertEqual(self.git.run("rev-parse", "HEAD"), head)
        self.assertEqual((self.repo / ".git/index").read_bytes(), index)
        self.assertEqual((self.repo / "engine").read_bytes(), work)
        commits = self.git.run("rev-list", "--reverse", f"{self.base}..{result}").splitlines()
        self.assertEqual(len(commits), 3)
        for commit, feature in zip(commits[1:], [self.one, self.two]):
            tree = self.git.tree(commit.decode())
            self.assertEqual({p: v for p, v in tree.items() if p not in owned},
                             {p: v for p, v in self.git.tree(feature).items() if p not in owned})
        with self.assertRaisesRegex(RebuildError, "already exists"):
            self.rebuild()
        self.assertEqual(self.git.commit("refs/heads/rebuilt"), result)

    def test_reject_unclassified_baseline_and_overlap(self):
        self.config["overlay_paths"].remove("preset.bat")
        self.write_manifest()
        with self.assertRaisesRegex(RebuildError, "Unclassified.*preset.bat"):
            self.rebuild()
        self.config["overlay_paths"].append("preset.bat")
        self.write_manifest()
        bad = self.commit(self.two, {"README.md": ("100644", b"feature collision\n")})
        with self.assertRaisesRegex(RebuildError, "fork-owned.*README.md"):
            self.rebuild([("one", self.one), ("two", bad)])
        self.assertFalse(self.git.run("for-each-ref", "refs/heads/rebuilt").strip())

    def test_reject_non_cumulative_stack(self):
        with self.assertRaises(RebuildError):
            self.rebuild([("two", self.two), ("one", self.one)])
        self.assertFalse(self.git.run("for-each-ref", "refs/heads/rebuilt").strip())

    def test_rebuild_from_rebuilt_dev_preserves_overlay(self):
        first = self.rebuild()
        self.overlay = self.commit(first, {"HANDOFF.md": ("100644", b"next session\n")})
        second = self.rebuild(output="refs/heads/rebuilt-again")
        owned = {p.encode() for p in self.config["overlay_paths"]}
        self.assertEqual({p: v for p, v in self.git.tree(second).items() if p in owned},
                         {p: v for p, v in self.git.tree(self.overlay).items() if p in owned})
        self.assertEqual({p: v for p, v in self.git.tree(second).items() if p not in owned},
                         {p: v for p, v in self.git.tree(self.two).items() if p not in owned})

    def feature_readmes(self):
        self.config["readme_feature_table"] = True
        self.write_manifest()
        self.shell = b"# Fork\n\n" + EMPTY_FEATURE_BLOCK + b"\n\nStable usage.\n"
        self.overlay = self.commit(self.overlay, {"README.md": ("100644", self.shell)})
        self.block_one = EMPTY_FEATURE_BLOCK.replace(
            FEATURE_END, b"| one | master | [docs](one.md) | squash(one) |\n" + FEATURE_END)
        self.block_two = self.block_one.replace(
            FEATURE_END, b"| two | one | [docs](two.md) | squash(two) |\n" + FEATURE_END)
        self.one = self.commit(self.one, {
            "README.md": ("100644", b"upstream\n\n" + self.block_one + b"\n")})
        self.two = self.commit(self.one, {
            "README.md": ("100644", b"upstream\n\n" + self.block_two + b"\n")})

    def test_section_owned_readme_and_repeated_rebuild(self):
        self.feature_readmes()
        head = self.git.run("rev-parse", "HEAD")
        index = (self.repo / ".git/index").read_bytes()
        work = (self.repo / "engine").read_bytes()
        result = self.rebuild()
        commits = self.git.run("rev-list", "--reverse", f"{self.base}..{result}").splitlines()
        for commit, block in zip(commits, [EMPTY_FEATURE_BLOCK, self.block_one, self.block_two]):
            actual = self.git.run("show", commit.decode() + ":README.md")
            self.assertEqual(actual, self.shell.replace(EMPTY_FEATURE_BLOCK, block))
            self.assertEqual(readme_parts(actual)[1], block)
        self.overlay = result
        again = self.rebuild(output="refs/heads/again")
        self.assertEqual(self.git.tree(result), self.git.tree(again))
        # A shorter stack must not retain the old second feature from the overlay source.
        shorter = self.rebuild(features=[("one", self.one)], output="refs/heads/shorter")
        self.assertEqual(self.git.run("show", shorter + ":README.md"),
                         self.shell.replace(EMPTY_FEATURE_BLOCK, self.block_one))
        self.assertEqual(self.git.run("rev-parse", "HEAD"), head)
        self.assertEqual((self.repo / ".git/index").read_bytes(), index)
        self.assertEqual((self.repo / "engine").read_bytes(), work)

    def test_section_readme_rejects_body_edits_and_malformed_blocks(self):
        self.feature_readmes()
        valid = b"upstream\n\n" + self.block_two + b"\n"
        for content in [valid.replace(b"upstream", b"feature body edit"),
                        valid.replace(FEATURE_END, b""),
                        valid + EMPTY_FEATURE_BLOCK,
                        valid.replace(b"| two |", b"not a row |"),
                        valid.replace(b"| status |", b"| description |"),
                        valid.replace(b"| two |", b"| two | extra |"),
                        valid.replace(b"| two |", b"| two")]:
            with self.subTest(content=content):
                bad = self.commit(self.two, {"README.md": ("100644", content)})
                with self.assertRaises(RebuildError):
                    self.rebuild(features=[("one", self.one), ("bad", bad)])
                self.assertFalse(self.git.run("for-each-ref", "refs/heads/rebuilt").strip())
        self.overlay = self.commit(self.overlay, {"README.md": ("100644", b"no markers\n")})
        with self.assertRaisesRegex(RebuildError, "exactly one"):
            self.rebuild()

    def test_overlay_deletion_preserved(self):
        self.overlay = self.commit(self.overlay, {"README.md": None})
        result = self.rebuild()
        self.assertNotIn(b"README.md", self.git.tree(result))


if __name__ == "__main__":
    unittest.main()
