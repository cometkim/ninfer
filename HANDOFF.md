## v3 artifacts made canonical on HF (2026-09-16)

The validated v3 bytes now live under the canonical filenames in both HF
repos via server-side copy; remote SHA-256 values were verified to equal the
recorded v3 checksums (ac98cd39... full, 8b86901a... qat) before the .v3.ninfer
duplicates and the stale v2-era conversion reports (they exposed local paths)
were deleted. Both cards lead with the v3 artifact table under the canonical
name and demote the v2 release to a superseded subsection; all commands use the
canonical filename. Card commits ac8e0b75/c3d6aee7 fast-forwarded the owner's
re-signed branches and are live on HF. Local models/ files renamed to canonical
names; temp worktrees and scripts removed. The earlier card commits 026ec899/
9ce6997d were superseded by the owner's restructure to e21f8b53/36697e5e.
All prior HANDOFF bytes follow unchanged.

## Upstream-rebase stack published to origin (2026-09-16)

Following the fingerprint check below, all sixteen refs were published
atomically to origin with exact per-ref force-with-lease values captured from a
live ls-remote that first matched every pre-rebase tracking tip: master
6cc95cc5->1d8587bc (fast-forward), cometkim/dev 5f6a787e->ca07c1b3, and the
fourteen rebased feat/* branches (msvc-test-constexpr bd401a25, windows-port
60b22bdc, webui bbcaa5d3, mtp7 c738401c, hyperquant d2126028, 1m-context
2cc56b5d, dflash2 246255e9, kernel-perf 4b99057d, build-speed-integration
c64f606c, build-speed ba5cafef, nvfp4-dflash2 f9c333e8, qwen3.8-profile-base
af0a1418, qwen3.8-nvfp4full e21f8b53, qwen3.8-nvfp4qat 36697e5e). Post-push
ls-remote confirmed every remote tip equals the local ref. Nothing was pushed
to upstream or natpate; feat/build-speed-upstream remains untouched (its
origin copy was already deleted remotely before this push). This HANDOFF-only
commit records the publication. All prior bytes follow unchanged.

## Upstream rebase onto 1d8587bc — stack, dev rebuilt and validated (2026-09-16)

Upstream master moved 6cc95cc5 -> 1d8587bc with three commits: the chat-template
literal-content frontend fix (8eaed538), the q4 4096x5120 linear dispatch tune
(bb844c43), and the W4A4 activation-scale one-tile-per-TMA-request rework
(1d8587bc). Local master fast-forwarded; nothing was pushed anywhere.

All fourteen active feature branches were rebased in stack order. Exactly one
conflict needed a semantic merge: the Windows descriptor-block commit in
feat/windows-port collided with the new tile-contiguous a_scales TMA addressing
in `nvfp4_w4a4_tma.cuh` and `nvfp4_linear_swiglu_w4a4_tma.cuh`. Resolution keeps
upstream's scale_tile computation (comment, kScaleTilesPerPlane, `(0,
scale_tile * 16)` box address) routed through the fork's device-resident
`descriptor_block->a_scales` instead of the by-value `descriptors.a_scales`.
Every other branch delta is patch-id identical to its pre-rebase delta; the
windows-port delta differs only in those two scale-load hunks. The historical
feat/build-speed-upstream and the unrelated worktrees (ninfer-1m-report,
ninfer-kp, the dirty sync3 stack) were untouched.

Dev was rebuilt from the new stack with tools/maintainer/rebuild_dev.py
(base 1d8587bc, overlay authority = pre-rebuild dev 5f6a787e, the same nine
cumulative features) and adopted as cometkim/dev ebf6cd43 after the tool's
exact feature/overlay parity verification. The old-dev -> new-dev content diff
equals the combined three-commit upstream delta, with only the two intended
TMA-header adaptations and hunk-offset shifts; README shell, feature table and
all overlay paths are unchanged. Pre-rebase refs are under
refs/backup/upstream-rebase-20260916/.

Validation on the rebuilt dev (RTX 5090, MSVC, sm_120a, build-ninja wrapper):
full build 422/422 targets, and the ten affected GPU/CPU tests all pass —
jinja, qwen3_5 frontend, q4 a16, nvfp4 a16/a4, attn/gdn input projection,
linear_add and linear_swiglu nvfp4 (which exercise the merged W4A4 TMA scale
route), and dflash2 nvfp4 routes. No full-model inference smoke or throughput
measurement was run for this documentation-level sync; no push to origin,
upstream or natpate occurred, and rebased feat/* plus dev remain unpublished.
All prior HANDOFF bytes follow unchanged.

## v3 profile artifacts validated and published to HF (2026-09-16)

Both newly converted v3 profiles were validated on the RTX 5090 fork build
(Windows, INT8 group-64 KV, 8192 context) across all four lanes: greedy text,
MTP3, DFlash2 draft-window 7 and Vision image input. Full answered coherently in
every lane (MTP3 30.3%, DFlash2 10.4% on the 128-token greedy smoke); QAT passed
plain/MTP3/Vision and initially failed DFlash2 startup because its artifact
predates the codebook-Use emission. QAT was regenerated through the standard
tools/upgrade_ninfer_v2_to_v3.py from the v2 release (weight bytes preserved);
the corrected artifact carries both selector-codebook Uses and its DFlash2 lane
passes (22.4%). Smoke acceptance rates are not quality claims; the v2 evaluation
tables remain the weights evidence since payload bytes are preserved.

Release facts recorded in both cards (commits 026ec899 full, 9ce6997d qat,
force-with-lease pushed): full.v3 19,407,229,188 bytes SHA-256 ac98cd392c84a04b2a21c2f5c3988dece88d20a697ba1de663fb32d5998b8ee9;
qat.v3 18,638,510,576 bytes SHA-256 8b86901a8cd2a297a3d737e470c793b67e5ce65b49131c48c2f2f0b346fd943c.
Updated cards are live on HF (commits 9937b014 full, 8ca9355c qat); both v3
artifact files uploaded and verified remotely (qat commit 614848bf; full commit
in the upload log). The defective
pre-fix qat.v3 copy is retained in tmp/ninfer-hf-release-20260916. All prior
HANDOFF bytes follow unchanged.

## Pre-sync-matched overlay and profile stacks published (2026-09-16)

Published seven refs atomically to origin with exact per-ref force-with-lease
values, all verified against both tracking refs and live ls-remote first:
cometkim/dev aca89e60, feat/1m-context ff85a086, feat/dflash2 e8e1c92c,
feat/kernel-perf 0980b6b1, feat/build-speed-integration 50154325,
feat/qwen3.8-nvfp4full 092483a9, feat/qwen3.8-nvfp4qat 3a829323.
Post-push ls-remote confirmed every remote tip equals the local ref. Observed
pre-publication remote values are kept under refs/backup/publish-overlay-presync-20260916/remotes/.
This carries the clean one-commit profile cards, their dev re-application, and the
fork-base overlay matched to the pre-sync inventory (wrappers back in the base,
1m-context relieved of machine-specific files). Remaining refs (master, the other
seven feat branches, historical feat/build-speed-upstream) were already current
and untouched. Nothing was pushed to upstream or natpate; no HF upload occurred.
This HANDOFF-only commit records the publication; all prior bytes follow.

## Fork-base overlay matched to the pre-sync inventory (2026-09-16)

The Ninja wrappers (build-ninja.ps1, configure-ninja.ps1) and vcpkg.json are
fork-base files again: identical pre-sync blobs now enter through the bottom
overlay, and feat/1m-context no longer carries them, restoring the rule that
machine-specific files never ride a feat branch. The manifest baseline is the
pre-sync fork base 83ddf726, so the rebuild audit classifies exactly that
inventory. CMakeLists.txt and src/CMakeLists.txt remain with feat/windows-port,
which absorbed the portable MSVC build layer during the v3 sync; they are
explicitly excluded from overlay copying. HANDOFF.md stays overlay-owned for the
rebuild design.

All four affected cumulative features were replayed with preserved messages,
authors and order; dev is rebuilt from the corrected overlay and the same nine
snapshots. The final dev tree changes only in HANDOFF.md and the manifest; the
wrapper blobs equal the pre-sync commits byte for byte. Prior refs are under
refs/backup/overlay-presync-20260916/. No build, conversion, GPU run or push.

## Clean profile cards re-applied to dev (2026-09-16)

Dev\'s cumulative dflash2/kernel-perf/build-speed-integration snapshots now carry
the rebuilt self-contained cards from the standalone profile branches (full
092483a9, qat 3a829323): existing sections edited in place for v3, reproduction
guides inlined into Reproduce, and both reproduction.md copies removed from the
tree. Commit messages, authors and stack order are preserved; dev is rebuilt from
the same empty-table fork overlay and nine ordered feature snapshots.

Exact comparison limits the dev tree change to the two cards, the two removed
guides and this HANDOFF update; product code, README rows and all other paths
are unchanged. Prior refs are backed up under refs/backup/dev-reapply-cards-20260916/.
No build, conversion, GPU run or push was performed.

## Profile cards rebuilt clean, one commit per branch (2026-09-16)

Dropped reproduction.md and the artifact-version note entirely; each card is
self-contained with its original section set, existing sections edited in place
for v3, and the v3 converter command inline in Reproduce. Each branch is now a
single clean commit over feat/qwen3.8-profile-base: full 092483a9, qat 3a829323.
Relative to the validated 49f0c22e/5470ccd3 tips only the card text changed and
reproduction.md was removed; recipe/test code is byte-identical, so the passing
CPU checks remain valid without a rerun. One diff --check trailing-whitespace
hit is original pre-sync card content preserved verbatim. Worktrees are clean.
Remote profile branches still hold superseded eec74d9a/8fe684de; publishing the
rebuild needs force-with-lease. Dev's cumulative squashes still carry the older
restored-card form and reproduction.md copies; align them at the next dev rebuild.
All prior HANDOFF bytes follow unchanged.

## Model-card v2 sections rewritten for v3 (2026-09-16)

Both profile cards now edit the existing sections in place instead of leaving
v2-era text beside new v3 guidance. Engine support describes v3 configuration/
bindings selection, upstream-supported Text/MTP routes, and the fork runtime
needed only for an NVFP4-encoded DFlash2 companion; branch topology reflects
the sibling recipes over profile-base. Run with NInfer uses current build paths,
the v2-to-v3 offline upgrade for the published file, and clarified companion
runtime needs. Reproduce points at the maintained recipe and reproduction guide
instead of removed v2 converter modules, and keeps the historical eval campaigns
as release evidence. The version note no longer calls those sections legacy.

Amended the documentation commits into 49f0c22e (full) and 5470ccd3 (qat);
the previously published eec74d9a/8fe684de are superseded locally and need a
force-with-lease push. Headings, structure and relative links verified; stale
registry/natpate/build-ninja/module-command references are gone. Whitespace
checks pass. No code or recipe change in this fix; tests were not rerun.
All prior HANDOFF bytes follow unchanged.

## Full/QAT profile branches published (2026-09-16)

Published only feat/qwen3.8-nvfp4full at eec74d9a and feat/qwen3.8-nvfp4qat
at 8fe684de to origin, atomically with explicit exact force-with-lease values.
Remote tips matched the prior tracking refs before publication and the local
tips afterward. Includes restored cards and appended v3 recipe updates.
Dev, profile-base and other branches were not pushed. No HF upload occurred.
This HANDOFF update remains uncommitted; all prior bytes follow unchanged.

## Independent Full/QAT v3 recipe updates appended (2026-09-16)

Appended eec74d9a on feat/qwen3.8-nvfp4full and 8fe684d on
feat/qwen3.8-nvfp4qat. Each profile owns its conversion rules; profile-base
remains unchanged at 790545eb. No prior profile commits were rewritten.
Full explicitly classifies imports/local encoding/BF16 exceptions and validates
parent calibration once; QAT imports projections and decodes its own controls.
Cards retain original release sections and add v3 conversion/component guidance.
The optional BF16-source W8 companion is not relabeled as the released NVFP4 module.

Seven existing CPU checks and both actual CLI recipe-loading paths pass in the
existing managed Python 3.11.16 environment. Separate Windows artifact IO was
supplied for validation, not embedded in the profile branches. Python compilation
and whitespace checks pass. No full conversion, model download, GPU qualification,
new artifact checksum, Git push or HF upload occurred. Dev code/history remain
unchanged; this prepend records the branch work only. All prior bytes follow.

## Full/QAT release model cards restored (2026-09-16)

Restored both pre-sync release cards at their original model-cards paths, including
YAML metadata, artifact hashes, quality tables, section order and release provenance.
Only an explicit v2/v3 interpretation note and pinned historical contract links alter
the original cards. The newer v3 instructions remain beside each as reproduction.md.
No release metric or checksum is relabeled as a v3 qualification.

Each standalone profile owns its restored card and guide. The cumulative dflash2
conversion commit carries the same documents beside its existing full/QAT recipes;
kernel-perf and build-speed-integration inherit them. README profile links now resolve
to these in-tree cards. Dev is reconstructed from the empty-table fork base and the
same nine ordered feature snapshots; cards are not placed in the fork overlay.
Prior refs are backed up under refs/backup/model-card-restoration-20260916/.
Exact comparisons protect original card content, existing product code and the other
five model-card directories. HANDOFF history and unrelated worktrees remain intact.
No model conversion, build, GPU validation or push was performed.

## Rebuilt feature documentation published (2026-09-16)

Successfully published 16 refs atomically to origin (cometkim/ninfer): cometkim/dev
at aa0933ea, all fourteen active feature branches listed below, and master at
6cc95cc5. Master was a verified fast-forward from origin and matched the recorded
upstream/master pin; no local master change or upstream synchronization was made.

Published feat/*: msvc-test-constexpr, windows-port, webui, mtp7, hyperquant,
1m-context, dflash2, kernel-perf, build-speed-integration, build-speed,
nvfp4-dflash2, qwen3.8-profile-base, qwen3.8-nvfp4full, qwen3.8-nvfp4qat.
Every existing destination matched its pre-publication origin-tracking value;
explicit exact per-ref leases protected the atomic push, with empty leases for
the two new branches (nvfp4-dflash2 and qwen3.8-profile-base). A subsequent
ls-remote verified exact remote/local equality for all 16 refs and no changes
to unrelated remote branches. Historical feat/build-speed-upstream was not pushed.
Observed existing remote tips are retained locally under
refs/backup/publish-readme-20260916/062654967869/.

This HANDOFF-only follow-up records the successful stack publication; its own
publication is verified separately after committing. No product code, README,
other worktrees or runtime priorities changed. Prior pending runtime work and
cautions below remain applicable. No new build/GPU tests, installs, downloads,
or pushes to upstream/natpate were performed.

Every pre-existing HANDOFF byte follows unchanged.

## Full/QAT profile directory restored (2026-09-16)

The four-column README feature directory now includes profile-base and the sibling
full/QAT recipe branches, linking their branch-owned guides/model cards through
stable origin GitHub branch URLs. Their squash column is an em dash: these extracted
branches are not separate dev squashes. Earlier full/QAT conversion recipes already
exist in cumulative dflash2; directory visibility does not integrate candidate code.
The cumulative dflash2 feature owns these directory links alongside its converter
entry; kernel-perf and build-speed-integration inherit them. The independent profile
branches retain ownership of the linked documentation and remain unchanged siblings
on profile-base over master. The bottom overlay keeps an empty feature table.

Only README and this HANDOFF prepend change on dev. Existing table rows, product
text, minimal natpate attribution, all previous HANDOFF bytes, feature/dev commit
messages and counts are preserved. Exact source-tree parity, local branch/document
link targets, four-column composition, and the empty base table are checked.
Other refs/worktrees, including dirty sync3, are preserved; main is clean.
No runtime/profile code integration, build, GPU run, download or push. GitHub URLs
are checked against local target refs/documents, not claimed remotely published.
Backups: refs/backup/profile-directory-20260916/.
Ref mapping/checks: C:/Users/comet/Workspace/tmp/ninfer-profile-directory-20260916/result.json.

Every pre-existing HANDOFF byte follows unchanged.

## Kernel runtime-performance guide corrected (2026-09-16)

kernel-perf now owns docs/features/kernel-perf.md and its four-column README row:
runtime QK/RoPE and attention-gate fusion, PDL/GDN chains, Small-T cache routes,
INT8 prompt split reduction and benchmark attribution controls. Build-speed guides
are unchanged; no new GPU measurements or end-to-end gains are claimed.
The correction is folded into kernel-perf's last commit, with only its
build-speed-integration descendant replayed. Dev retains nine feature snapshots;
the bottom committed overlay owns this prepend and an empty feature table.
All prior HANDOFF bytes, other README rows/body, natpate provenance, feature
messages and the preceding feature chain are preserved. Exact tree checks allow
only README, the new guide and HANDOFF; unrelated refs/worktrees are unchanged.
No code/tool changes, build, GPU run or push. Pending runtime work remains as below.
Backups: refs/backup/kernel-perf-docs-20260916/.
Ref mapping/checks: C:/Users/comet/Workspace/tmp/ninfer-kernel-perf-docs-20260916/result.json.

Every pre-existing HANDOFF byte follows unchanged.

## Four-column feature table restored (2026-09-16)

Supersedes only the prior two-column table guidance. Feature-owned four-column rows and
meaningful descriptions are restored from pre-compaction history; the brief natpate
Windows-port/WebUI provenance, current README product sections and clean feature guides
remain unchanged. The bottom overlay owns the empty table and this correction; kernel-perf
owns the restored validator/tests. Fourteen feature histories and the nine-snapshot dev
stack retain feature messages/counts/topology and exact product trees, with no top patch.
Seven rebuild tests, relative links/anchors and whitespace checks pass. Other worktrees,
dirty bytes, sync3 and historical build-speed-upstream are preserved. No push.
Backups: refs/backup/readme-table-restored-20260916/ (dev and feat/*).
Ref mapping and verification: C:/Users/comet/Workspace/tmp/ninfer-readme-table-restored-20260916/result.json.

Every pre-existing HANDOFF byte follows unchanged.

## Minimal README correction completed (2026-09-16)

Supersedes the prior four-column README guidance without changing its historical bytes.
README now has a brief personal-fork introduction and two-column Feature/Documentation
entries only. natpate is credited once in Upstream lineage for Windows-port/WebUI provenance.
The pre-sync heading order and product sections remain; operational branch/remotes/rebuild
and ownership prose is removed. The bottom fork overlay has an empty table; all fourteen
active feature histories own their short entries and linked documentation. Tooling belongs
to kernel-perf and is inherited by build-speed-integration. No top documentation patch.

Reconstruction preserves feature commit counts, messages and parent topology, and exact
product trees at every rewritten commit. Seven rebuild-tool tests pass, including malformed
row rejection, repeated rebuilds and dirty checkout preservation. Relative documentation
links/anchors and whitespace checks pass. The four active PR worktrees and dev are clean;
other worktree HEADs, indexes, diffs and dirty/untracked file bytes are preserved, including
sync3. Historical feat/build-speed-upstream is unchanged. No push, dependency installation,
GPU build, inference run or candidate-runtime integration was performed.

Original refs: refs/backup/readme-minimal-20260916/ (dev and feat/*).
Evidence and ref mapping: C:/Users/comet/Workspace/tmp/ninfer-readme-minimal-20260916/result.json.
Local candidate observations moved out of feature guides follow. These describe the
pre-correction state; this documentation change does not integrate the pending runtime fixes.

docs/features/nvfp4-dflash2.md:
It is separate from the cumulative `feat/dflash2` branch; its corrections are not implied there.

model-cards/Qwen3.8-27B-nvfp4full-NInfer/README.md:
Vision/MTP use upstream optional allocation from the BF16 base. A requested DFlash2
component uses the saved converter's **W8** allocation by default. Existing installed
full/QAT artifacts instead carry **34 NVFP4 module objects**. Base conversion below
reproduces the Text profile, not those artifacts byte-for-byte. The earlier 31-object
override is incomplete and must not be presented as faithful reproduction.

The retained broader `feat/custom-weight-profiles` candidate at `766a1423` owns the
34-object override, module-only W8 splice and v2 migration; it is not a prerequisite
for these base recipes. NVFP4 DFlash execution is separately owned by
`feat/nvfp4-dflash2` at `09d69a07`. Installed NVFP4-module artifacts may require that
binding support even when speculation is disabled; do not claim bare upstream can
admit every installed artifact. Text-only and Text/MTP artifacts produced here use
existing upstream runtime routes structurally; no GPU qualification is claimed.


model-cards/Qwen3.8-27B-nvfp4full-NInfer/README.md:
Windows validation used existing managed Python 3.11 with separately supplied platform IO.
No full checkpoint conversion, calibration run, model download, GPU build, inference
quality or new performance measurement was performed for this packaging change.

model-cards/Qwen3.8-27B-nvfp4qat-NInfer/README.md:
Vision/MTP use upstream optional allocation from the BF16 base. A requested DFlash2
component uses the saved converter's **W8** allocation by default. Existing installed
full/QAT artifacts instead carry **34 NVFP4 module objects**. Base conversion below
reproduces the Text profile, not those artifacts byte-for-byte. The earlier 31-object
override is incomplete and must not be presented as faithful reproduction.

The retained broader `feat/custom-weight-profiles` candidate at `766a1423` owns the
34-object override, module-only W8 splice and v2 migration; it is not a prerequisite
for these base recipes. NVFP4 DFlash execution is separately owned by
`feat/nvfp4-dflash2` at `09d69a07`. Installed NVFP4-module artifacts may require that
binding support even when speculation is disabled; do not claim bare upstream can
admit every installed artifact. Text-only and Text/MTP artifacts produced here use
existing upstream runtime routes structurally; no GPU qualification is claimed.


model-cards/Qwen3.8-27B-nvfp4qat-NInfer/README.md:
Windows validation used existing managed Python 3.11 with separately supplied platform IO.
No full checkpoint conversion, calibration run, model download, GPU build, inference
quality or new performance measurement was performed for this packaging change.

Every pre-existing HANDOFF byte follows unchanged.

## Feature-owned README entries and pre-sync structure restored (2026-09-16)

Supersedes the earlier whole-file README ownership interpretation. Fork-base retains the
pre-sync section order and four-column feature-table placeholder. Each active feature
branch now contributes its own linked row; cumulative branches inherit preceding rows.
Upstream-PR branches carry only the marked table and their own documentation, not the
fork-base prose. The reconstruction tool composes the base shell and feature table with
explicit validation, including repeated rebuilds and rejection of edits outside the table.

All fourteen active feature refs were reconstructed without changing product code; the
maintainer tool/tests are owned by kernel-perf and inherited by build-speed-integration.
Standalone full/QAT remain siblings over profile-base; runtime and build-speed remain
independent. Historical build-speed-upstream and experimental worktrees are untouched.
Dev includes the nine cumulative entries only, not unintegrated standalone candidates.

Prior refs are retained under refs/backup/feature-readme-20260916/; reconstruction evidence
and original working documentation are in Workspace/tmp/ninfer-feature-readme-rebuild-20260916.
Every existing HANDOFF byte follows unchanged. Product trees are compared exactly outside
README, new feature guides and the scoped rebuild tooling; links and documentation structure
are checked. No numerical behavior changed, so no GPU builds or inference runs are needed.
No push or new candidate-runtime integration was performed.

## Fork-base documentation ownership applied (2026-09-16)

The corrected stable README, durable AGENTS rules, and complete HANDOFF history are
folded into the bottom fork overlay above pinned upstream 6cc95cc5. The nine cumulative
feature deltas are rebuilt unchanged; no top-of-dev README patch remains. README keeps
only Windows-port/WebUI provenance for natpate and an empty feature-table placeholder.
Feature details belong to their branches; local state remains here.

Pre-rebuild dev is preserved at refs/backup/readme-fork-base-20260916/dev; committed
overlay input is retained alongside it at /overlay-source. Original working documentation
bytes and patch are saved in C:/Users/comet/Workspace/tmp/ninfer-readme-fork-base-20260916.
Every pre-existing HANDOFF byte is preserved below. Exact final-tree comparison allows
only README, AGENTS and this HANDOFF update; feature refs and non-overlay source content
are unchanged. No build/runtime tests were needed for this documentation-only change.
No remote push, candidate-runtime integration, or other worktree change was performed.

## README ownership correction (2026-09-16)

Supersedes the README-refresh guidance below: root README is stable fork-base documentation,
owned by the bottom fork-base commit and explicit overlay, not feature patches. Its feature
table is an empty placeholder; each feature branch owns its own documentation. Local branch
state, measurements and pending work belong only here. natpate is Windows-port/WebUI provenance,
not an actively followed upstream. README retains product usage/reference material and links
Windows setup to its owning guide; patch inventories and local status have been removed.
AGENTS now records these ownership rules; fork_overlay.json already includes README.md.

Documentation changes remain uncommitted; no history, branches, refs or feature documents were
changed. No builds or runtime tests ran. All earlier HANDOFF bytes are preserved below.

# HANDOFF — cross-session work state

## README current-state refresh (2026-09-16)

Documentation only: README now separates the local pinned Jinja integration, standalone
upstream runtime/build-speed candidates, and sibling full/QAT profile patches. It records
unpublished status, retired branch preservation, pending candidate-only fixes, v3 migration,
explicit Windows preset paths and scoped verification. Upstream performance excerpts were
aligned with the pinned upstream README, not remeasured. Existing downloaded/installed
artifacts were not changed; no new conversion, GPU test, build, fetch, push or commit ran.

Checks: README relative links/anchors and git diff --check pass. Only README and this
prepend-only HANDOFF entry were changed in this task. Every pre-existing HANDOFF byte,
including the uncommitted additions and restored historical sync section, is preserved.
Next implementation work remains integrating the candidate-only runtime/module corrections
through their feature owners before any dev rebuild; do not treat profile preservation as
fresh full conversion or model-quality qualification. Historical entries below are unchanged.

## Latest status — obsolete branch removed; historical entries preserved (2026-09-16)

At the owner's request, the local feat/custom-weight-profiles branch was deleted. Its
766a1423 tip remains at refs/backup/profile-lineages-20260916/heads/feat/custom-weight-profiles,
and its existing worktree is detached. PR-ready profile/runtime branches and remotes were
not changed. Older entries below describe their state at the time, not current branch status.

HANDOFF updates must prepend new entries without deleting or rewriting earlier sections.
The original upstream-v3 sync section removed during earlier summarization is restored below
from a63a9518. Subsequent validation and publication entries supersede its pending-work status.

## Full/QAT lineage correction — scoped sibling recipes (2026-09-16)

The generic extraction below is superseded as the profile packaging decision, not deleted.
Existing lineage names are now semantic rebuilds on upstream 6cc95cc5:
- feat/qwen3.8-profile-base b56b2e74: one 22-line shared allocation/grouping helper.
- feat/qwen3.8-nvfp4full f7065dbd: full recipe, local encoder, calibration-path input,
  focused CPU tests and owning reproduction/model card (five files above shared base).
- feat/qwen3.8-nvfp4qat 5f0c43c5: independent QAT recipe, CPU tests and owning card
  (three files above shared base; no import of full, encoder or calibration dependency).

Full preserves 112 imported / 135 local NVFP4 / nine BF16 parent exceptions and Q8
vocabulary. QAT preserves 256 imported NVFP4 parents, 48 BF16-decoded control groups,
Q8 vocabulary and source divisors. Both cards distinguish official source, full and QAT;
provide explicit local-source conversion/plain-decode/MTP commands; link immutable old
cards/artifact authorities; and disclaim historical quality/performance as new v3 evidence.
Full consumes the saved 135-site calibration JSON as an explicit local prerequisite.
Historical calibration implementation/corpus remain in the preserved old history.

All pre-task branch and remote refs were backed up under
refs/backup/profile-lineages-20260916/{heads,remotes}/* before changing old lineage refs.
Original full55152a4f and qat91f8a447 model-card/profile provenance remains there intact.
Broader feat/custom-weight-profiles766a1423 remains unchanged as a superseded extraction
and optional tooling candidate (34-object NVFP4 module override, W8 splice, migration).
Runtime feat/nvfp4-dflash2 remains09d69a07, separate and optional. Neither is a dependency
of base Text/MTP recipes. Installed 34-object NVFP4-module artifacts are NOT the default
W8-module recipe output; bare upstream admission of those artifacts is not asserted.
No obsolete registry/inventory, Windows/HQ/1M/kernel/build-speed runtime delta was included.

Verification: seven CPU tests pass in managed Python3.11.16/Torch CPU, including actual
logical preparation at model shapes, exact imports/control decode and full encoder oracle
checks. Both documented module test entry points and converter --help work with separate
Windows IO support supplied from the existing validation checkout; bare upstream fails
on Windows os.sysconf, so no platform fix is hidden in these patches. py_compile and both
upstream-to-tip diff checks pass. No GPU/build, large conversion, download or quality run.
Profile/base worktrees are clean under C:/Users/comet/Workspace/tmp/ninfer-profile-{full,qat,base}.
Main remains e32dd6eb with only this uncommitted HANDOFF change; cumulative refs and runtime
branch are unchanged. No push, PR or dev rebuild was performed.

## Independent upstream PR extraction — local branches ready (2026-09-16)

Both branches are direct children of pinned upstream master 6cc95cc5:
- feat/nvfp4-dflash2 09d69a07: execution/binding/Ops and independent tests for all 34
  installed NVFP4 DFlash2 module objects; no conversion, Windows, HQ, 1M or kernel-perf delta.
- feat/custom-weight-profiles 766a1423: full/QAT recipes, encoder/calibration input,
  34-object module override, W8 splice, migration admission and CPU tests; no runtime delta.

Fresh disposable Windows validation integrated both with platform/MSVC fixes only. Full
build passed; 11 GPU oracle tests plus corrected artifact-reader regression passed, and eight
CPU conversion tests passed. Real full.v3 DFlash2 K7 tg128/int8/8192 CUDA Graph smoke succeeded
(60/471 accepted, 68 rounds). This is not the cumulative fork's expected acceptance fingerprint
or a throughput comparison. Linux build and full model-quality qualification were not run.

Extraction exposed inherited issues fixed in the candidates ONLY: native NVFP4 subview scale
pointer offset was overwritten by the parent pointer; lookup codebooks incorrectly required
activation Use metadata absent from upstream conversion; the old override encoded only 31
rather than the installed 34 NVFP4 objects. These corrections are NOT yet integrated into the
existing cumulative feat/dflash2/dev tree. Do not claim that tree has the candidate fixes.

No pushes or GitHub PRs. Main source and existing cumulative refs remain at e32dd6eb; this
handoff update is uncommitted. PR descriptions and validation evidence are retained under
C:/Users/comet/Workspace/tmp/ninfer-pr-*. Runtime and tooling worktrees are respectively
ninfer-pr-nvfp4-dflash2 and ninfer-pr-custom-weight-profiles under that temporary directory.

## Resumed pinned Jinja sync — validated local adoption (2026-09-16)

Target is explicitly pinned upstream `6cc95cc5023bb9a69f354960b64afc0c70f9771f`.
Original local refs are preserved at `refs/backup/resumed-jinja-20260916/*`; prior deferred
candidates and historical worktrees remain intact, including the original sync3 five-file patch.
Saved semantic rebases were reused, then owner fixes were replayed through descendants.
No push is authorized or performed in this continuation.

Windows owns LF checkout of maintained `tools/chat_templates/*.jinja` resources. Upstream's
new generic frontend removes the old digest-qualified fixtures; do not retain a rule for those
deleted files or weaken production digest checks. Existing checkouts may need their unchanged
template files refreshed through Git after the attribute is introduced. The closure-selection
test no longer injects a 1ms sleep (about 16ms on this Windows host); its two-action, non-eviction
assertions and production budget policy remain unchanged, as does the separate delay-budget test.
DFlash owns full/QAT admission in the standard `tools/upgrade_ninfer_v2_to_v3.py`; the broken
fork wrapper was removed. All six fork presets explicitly select the existing full `.v3.ninfer`.

Production build succeeded using unlimited parallelism, Windows/MSVC 14.44, CUDA 13.3, sm_120a.
Actual production frontend, C++ Jinja, and resource-manager tests pass. Six maintained-template
Python/C++ parity tests, eight conversion/IO/splice checks, and five rebuild-tool tests pass.
The CMake cache initially selected global Python 3.14 without Jinja2; corrected locally to existing
`C:/Users/comet/Workspace/ninfer-src/cvenv/Scripts/python.exe` (managed Python 3.11), and the actual
chat-template CTest entry passes. No dependencies were installed. Real full and QAT v2 metadata
map successfully (1325/1334 objects, 1513 bindings); direct CLI execution outside the repository
reaches overwrite refusal on an existing sentinel, preserving it without copying model payloads.
No artifact was regenerated or downloaded, and no v2 runtime fallback was introduced.

Final product candidate `resume/jinja-dev-final` (950410a6 before overlay documentation) was
create-only rebuilt with exact feature/overlay parity. Product `src/ops` is byte-identical to
the rolled-back validated baseline. Complete serial CTest: 126 entries in 779.21 seconds,
118 passes, seven prerequisite-dependent skips, one Python-environment failure; after correcting
the interpreter, `ctest --rerun-failed` passes that final test (0.19s). Resolved result is
119 passes, seven skips, zero remaining failures. No numerical criteria were changed.
Live existing full.v3 artifact, int8 KV, 8192 context, greedy/no-thinking, max-new128,
"Write a haiku about a GPU.": embedded vs maintained templates render byte-identical prompts
through the production C++ renderer and yield identical 86-byte CLI outputs, 20 prompt tokens
and 18 generated tokens. This is a representative template regression check, not a new
performance claim or a claim that all intentionally changed template behavior is identical.

Locally adopted master is pinned6cc95cc5. Cumulative features: msvc c8fa75b9,
Windows ff79165d, WebUI 1ace5fb4, MTP7 2503d751, HQ 5b72d809, 1M 20130ab3,
DFlash 2787eb14, kernel 4d9c1e10, build-speed integration 37a34aac.
Standalone build-speed remains upstream-only950e0552 (no Windows/HQ changes).
Dev contains the validated candidate plus overlay documentation only. No remaining in-scope
implementation blocker. Linux build, skipped real-model CTest prerequisites and a full offline
weight rewrite were not exercised; the existing artifact was neither regenerated nor replaced.
Evidence directory: `C:/Users/comet/Workspace/tmp/ninfer-whole-sync-regression-20260916/runtime/`.
The historical sections below describe earlier states; their two known failures are superseded
by the actual frontend and resource-manager passes above, not waived.

## Build-speed feature rebuilt — validated and published to origin (2026-09-16)

Standalone upstream review branch `feat/build-speed` is fe815297 on agreed master e360c4c0.
It contains only prompt BF16/INT8 dtype TUs and NVFP4 non-RDC H24/H16 geometry TUs; no
Windows changes, fork signatures, HQ, TMA edits or machine paths. Live upstream/master was
6cc95cc5 when checked; it was not fetched or silently integrated. Historical feat/build-speed
195c5dc7 and feat/build-speed-upstream d41d617f are backed under
`refs/backup/build-speed-20260916/{feat-build-speed,feat-build-speed-upstream}`; the latter
historical public ref remains unchanged. Pre-task dev fdb7dbc9 is backed there as `dev`.

Cumulative owner `feat/build-speed-integration` is a9c741d8 on feat/kernel-perf ea781da8.
It owns the semantic BF16/INT8 adaptation to fork split reduction, identical NVFP4 geometry
split, and fork-only HQ geometry split. Existing cumulative feature refs remain untouched;
no integration fix belongs only to dev. The separate sync3/kernel-perf worktree's original
five-file dirty patch remains untouched. Main original dirty sources and tracked diff were
also preserved in `C:/Users/comet/Workspace/tmp/ninfer-build-speed-20260916/`.

Bounded sampling used exact production NVCC flags, unlimited parallelism, Windows/MSVC14.44,
CUDA13.3.73, sm_120a. Compile-only Ninja wall times (seconds):
- NVFP4 non-RDC: baseline 7.043/7.021; geometry split 6.086/6.182 (~12.8% lower).
- HQ: baseline 14.325/14.095; geometry split 9.875/9.964 (~30.2% lower).
- FP8 sampled at 6.154; no further split pursued. K8V4/metadata/family over-splitting was
  not pursued after these bounded candidates resolved the useful scope.
These are affected-TU compilation claims, not clean-build or inference-speed claims.
BF16/INT8 prior compile+link evidence below remains valid; it was not rerun or generalized
to the new geometry changes. Kernel math, launch shapes, metadata variants, codec and carry
boundaries are unchanged. All standalone upstream changed CUDA TUs compile against upstream
headers with exact production flags. Production integrated softmax/HQ targets build.
Serial RTX 5090 production oracle tests passed at unchanged criteria: general softmax
163.29s (BF16/INT8 and represented packed-cache mathematical oracle), HQ 7.80s (codec,
residual/carry and both geometries), NVFP4-only 11.62s. No test was weakened or replaced.
Full Linux build was not available. Evidence/scripts/logs reside in the temporary directory
above; no benchmark harness or machine settings were added to the feature.

Dev product snapshot is 67a43c4b, reconstructed with the existing create-only tool and
explicit overlay plus nine adjacent cumulative feature squashes ending in build-speed-integration.
Exact tested full-tree parity against preserved a7a61066 was verified before atomic dev ref
adoption; checkout/index were not reset. Subsequent dev documentation only records completion.
The tool's historical-overlay ancestry precondition rejects already-rebuilt dev, so
`refs/backup/build-speed-20260916/overlay-provenance` (83e8bc6c) anchors the exact tested tree
to both a7a61066 and the preserved historical baseline. This changes no product bytes and does
not bypass the inventory/parity checks. README records the usable command and both branch forms.
At the owner's request, feat/build-speed fe815297, feat/build-speed-integration a9c741d8,
and dev d5c21f36 were published atomically to origin with explicit force-with-lease expectations.
The following dev documentation commit records publication only. No upstream/natpate push or
PR was made; feat/build-speed-upstream remains unchanged. No unrelated feature rewrite or
deletion of historical worktrees/builds/dirty sync assets occurred. No remaining implementation
or numerical blocker is known; Linux full-build validation remains unperformed.

## BF16/INT8 prefill TU split — prior qualification, now owned by build-speed integration (2026-09-16)

Restored the measured dtype-specific prompt launch translation units under
`src/ops/softmax_attention/dense/causal_cache`: `prompt_bf16.cu`, `prompt_i8.cu`, and the
private `prompt_routes.h` boundary, registered by `src/ops/softmax_attention/sources.cmake`.
`prompt.cu` retains dispatch and wave-fill policy; both geometries and direct/unmasked/masked
metadata instantiations remain covered. Kernel bodies, launch grids, constants, cache formats,
math and split reduction are unchanged. Dynamic-smem attributes now initialize only the selected
dtype. Final code matches the measured candidate tokens (formatting/comments only).

Validation: production `build-ninja.ps1 -Target ninfer_softmax_attention_test` succeeded with
unlimited build parallelism. Serial CTest `-R '^ninfer_softmax_attention_test$'` passed in
156.53 seconds on RTX 5090 selected by UUID (Windows 11, MSVC 14.44, CUDA 13.3.73, sm_120a).
The existing independent FP64 oracle checks represented BF16/FP16 and signed INT8 codes with
stored FP16 scales, causal attention for D256/H24/KV4 and D256/H16/KV2, cached/append prefill,
paged mappings, masked/ragged batches, and output/cache guards at unchanged criteria.
This production-target pass resolves the numerical qualification blocker; the isolated custom
harness's 0xc0000409 root cause remains undiagnosed and its runs are not validation evidence.

Prior uncontended measured affected-TU build results remain valid and were not repeated:
compile 12.043 -> 8.944 seconds; compile plus device link/archive/test host link
48.617 -> 45.240 seconds (6.9% improvement). Not a clean-build or inference-speed claim.
Evidence and owned logs: `C:/Users/comet/Workspace/tmp/ninfer-prefill-split-20260916/`
(`evidence.json`, `production-build-final.log`, `production-oracle.log`, `verified-feature.patch`).

The original five-file patch remains UNCOMMITTED in
`C:/Users/comet/Workspace/tmp/ninfer-sync-stack-20260916`, on `sync3/kernel-perf` ea781da8;
do not discard it during unrelated work. It is now committed semantically through the separate
build-speed integration owner described above, not by rewriting kernel-perf. The earlier
uncommitted-main ownership and deferred-authorization notes are superseded by this rebuild.
The prior valid dtype timing and production numerical qualification remain evidence; the
failed custom harness has not been substituted for production validation.

## Upstream v3/custom-weights sync — reconciled, validated and published (2026-09-16)

Upstream baseline is e360c4c0 (v3 custom weights, shape-keyed Ops, src/models/qwen3_5).
The initial a63a9518 integration was not lossless: DFlash file replacements removed PDL,
route tests and recipe capabilities. Do not use it or 77027734 as the final sync result.
Pre-sync refs/backup/pre-sync-20260916/* and backup/20260916/all-refs.bundle are preserved.

### Ownership and continuous rebasing

The curated stack is now cumulative: msvc-test-constexpr -> windows-port -> webui -> mtp7
-> hyperquant -> 1m-context -> dflash2 -> kernel-perf. Review each against its predecessor.
Kernel-perf owns frequency-aware Q/K fusion, gated attention, HQ PDL, NVFP4 GEMV/SIMT and
DFlash launch dependency contracts, measurement controls and request-error isolation.
It now inherits the Windows TMA fix; the old master-only bb7c1f8c snapshot remains recoverable
but is not the final curated tip. No dev-only semantic conflict-fix layer is permitted.

New sync3/* refs are reconstructed in the isolated worktree
C:/Users/comet/Workspace/tmp/ninfer-sync-stack-20260916. Their interim tips are backed up at
refs/backup/final-validation-20260916/sync3/*. Final public refs are updated only after the
create-only rebuild verifies exact ownership parity. The maintenance tool is
`tools/maintainer/rebuild_dev.py`; its explicit `fork_overlay.json` preserves the actual
fork environment, instructions, evaluation resources, presets and fixtures. It reads only
committed inputs, never stages or resets the checkout, and refuses an existing output ref.

### Preserved behavior and justified replacements

- Restored all twelve affected NVFP4 PDL files, including device waits/end publication and
  DFlash QKV launches; retained integrated frequency-table/YaRN and HQ publish-at-end forms.
- Restored DFlash independent oracle test/registration, five A16 shapes, context-materialize
  replay ordering and complete positive-T A16 contract documentation.
- Full/QAT allocations are real v3 recipes, not aliases of the official mixed-FP8 recipe.
  Full keeps 112 imported MLP parents, 135 calibrated local parents, nine BF16 exceptions
  and Q8 vocabulary. QAT imports its own packed words/divisors and BF16-decoded controls.
  The saved converter default is a W8 DFlash module; the explicit override preserves the
  shipped 31-parent NVFP4 module. Module-only W8 splice retains non-module payloads exactly.
- Windows positional Python reads are restored in the IO owner and preserve binary data,
  file position and EOF. Source/runtime paths follow v3, not deleted target inventories.
- Dropped only the obsolete W8 small-T TU split (upstream per-shape Q8 already owns it),
  duplicated Windows TMA implementation and obsolete fixed-profile registration/inventories.
  The owner-reverted NVIDIA all-64-layer MLP experiment is not reintroduced; it is not the
  saved fork baseline. Full/QAT profile functionality is retained, not dropped with wrappers.

### Verification

Windows 11, RTX 5090, sm_120a, MSVC 14.44, CUDA toolkit 13.3.73; runtime reports CUDA 13.4.
Build wrapper completed with zero errors after the complete PDL restoration.
- Final serial CTest: 115 passed, seven skipped, two known failures out of 124, 816 seconds.
  Resource-manager failure remains candidate-stratified reuse closure. Frontend remains
  0xc0000409; it also fails standalone in this session, so do not repeat the older claim
  that it currently passes standalone. No additional test failures. Log: ctest-reconcile-full.log.
- Nine focused restored/integrated numerical tests passed, including DFlash route oracle,
  A16 real shapes, attention/GDN, Q/K fusion, HQ and RoPE scaling (ctest-reconcile.log).
- Existing models/qwen3_8_27b_nvfp4full.v3.ninfer: tg128, int8 KV, 8192 context, one warmup
  and one measured repetition. MTP3 = 71/170 accepted (41.76470588%), 57 rounds;
  DFlash2-K7 = 63/450 (14.0%), 65 rounds, 1.969230769 tokens/round. Both match saved fingerprints.
  Evidence: profiles/bench/sync-20260916/reconcile-{mtp3,dflash7}.json.
- Greedy raw-output CLI, haiku prompt, int8 KV, max-context 8192, max-new 128: fusion/PDL
  0/0 versus 1/1 outputs byte-identical. Retained /tmp/reconcile-ab{00,11}.{txt,log}.
- Eight focused CPU synthetic conversion/IO/splice checks passed with existing uv-managed
  Python 3.11.16/Torch CPU; pytest absent, direct test functions used. Exact imports/divisors,
  encoder ties/chunk invariance, full/QAT preparation and 31-parent override covered.
- Four isolated rebuild-tool tests pass, including exact blob/mode transformations,
  untouched dirty worktree/index, ownership/ancestry errors and overwrite refusal.

No full checkpoint conversion, new quality campaign or performance-gain measurement was run.
Old throughput tables are not new v3 measurements. BF16 KV graph-update limitation is unchanged;
use int8 for the matched A/B workload. GPU jobs must remain serial.

### Publication and scaffolding

The cumulative feature stack and clean dev rebuild were pushed atomically to origin with
explicit force-with-lease expectations. Published feature tips: msvc-test-constexpr 5160036b,
windows-port c86db216, webui d97bb201, mtp7 691bab10, hyperquant 2d2232e9,
1m-context f70c9de6, dflash2 ff275d43, kernel-perf ea781da8. The clean dev product snapshot
is 4840e892 (one overlay + eight feature squashes); subsequent changes only record publication
in this fork-owned handoff. Local feat/* refs and cometkim/dev are aligned with these results;
old local refs are backed up under refs/backup/final-validation-20260916/local/*.

Exact feature/overlay parity was verified by the rebuild tool. Compared with the tested input,
the final product snapshot differs only in restored DFlash public-contract comments. No code
was changed after numerical/inference validation. Full/QAT standalone historical branch refs
remain retained; their preserved v3 implementations are now part of feat/dflash2.

Do not push tmp/kp-validate. ../ninfer-kp and the sync worktree/builds are retained intentionally
as validation scaffolding, not public feature branches. Local build wrappers, vcpkg.json and
logs remain excluded and unstaged. Earlier rebased/*, sync3/* and dev-rebuild2 refs remain for
recovery; no backup history or local build assets were deleted. Optional future work is a fresh
paired v3 performance campaign, not an unfinished part of this preservation sync.

## Upstream v3/custom-weights sync — COMPLETE except kernel-perf (2026-09-16)

Upstream moved to e360c4c0: the v3 container format, model/weight decoupling (issue #245:
custom weight recipes, no checkpoint-specific engine registration), the shape-keyed op tree
(src/ops/linear/<fmt>/shapes), and the consolidated src/models/qwen3_5 runtime replacing
src/targets/*. The whole feat stack was clean-rebuilt on it; all work happened on rebased/*
branches and cometkim/dev was rebuilt from a fresh fork base. Backups: refs/backup/
pre-sync-20260916/* and backup/20260916/all-refs.bundle.

Branch state (rebased/* = the new tips; push as feat/*):
- rebased/msvc-test-constexpr (5160036b): constexpr-sqrt + <array> CTAD test fixes.
- rebased/windows-port (ee6e721d): full port onto v3 — InputFile unbuffered overlapped IO,
  Uint128 for the new cost/planner sites, portable logging/Winsock/CRT call sites, MSVC
  global flags, UTF8PROC_STATIC, TMA descriptor blocks, mkdtemp-fixture and GNU-ld-wrap test
  guards, share-write/delete artifact handles, Python artifact IO portability (pwrite/
  fdatasync/fadvise/sysconf). Suite green except two upstream-observed flakes:
  ninfer_resource_manager_test "candidate-stratified reuse closure" (planner expectation,
  host-order dependent) and ninfer_qwen3_5_frontend_test under ctest only (passes
  standalone; upstream's own commit notes ctest/direct disagreement).
- rebased/webui (3034fa8a), rebased/mtp7 (7a761184): rebased clean.
- rebased/hyperquant (b2aa8ec2): full port; serial ctest 110/120 (the two flakes above).
- rebased/1m-context (80bc4fb1): YaRN/rope-frequencies ported into src/models/qwen3_5
  (ExecutionCore rope_frequencies, TextContext::set_rope_frequencies, side-row lifecycle in
  program/storage, fork-side-row epochs in kv_store.h, residual window in
  state/decoder_state.*, envelope validation in planning/startup.cpp). BUILD PENDING — the
  worktree build COMPLETE and green: rope-scaling tables, hq codec, hq retrieval,
  hq attention and all softmax suites pass (12/12 targeted tests).
- rebased/dflash2 (9bc4392f): NVFP4 draft-module ops onto the shape-keyed tree (shapes/
  dflash2.cu registering the five drafter matrices, attn-input simt port, conv prepare/
  finish, materialize, selector codebooks), v3 loading (NVFP4 QKV parent, 128-row-aligned
  sub-range native weights for the m128x4 swizzle, codebook uses), nvfp4 swiglu A16
  decomposition, the v2->v3 upgrade fork extension (nvfp4full/nvfp4qat inventories) and the
  two profile recipes. dflash2_nvfp4_routes + nvfp4_a16 tests PASS.
- feat/kernel-perf: NOT ported this sync (largest remaining item; fusions/PDL/TU splits over
  the reorganized op tree — the old branch and its dev-only companions 494b8943/f95c86d1
  stay recoverable from refs/backup/pre-sync-20260916 and the pre-rebuild dev). Port it
  next session; remember to drop its duplicated Windows TMA fix (already in windows-port).

IMPORTANT FIX this sync: the W4A4 TMA launchers' Windows descriptor block copied from the
LAUNCHER STACK via cudaMemcpyAsync — under CUDA graph capture the graph bakes the host
pointer and replays read a dead frame (illegal-instruction on every NVFP4 shape). Replaced
with a one-thread kernel carrying the descriptors as a by-value uint4 payload (graph nodes
snapshot parameters). Also InputFile must complete ERROR_IO_PENDING reads via
GetOverlappedResult, and artifact handles must share FILE_SHARE_WRITE|DELETE (tests rewrite
fixtures while a Reader is live).

Artifacts: all local v2 files upgraded to v3 (payload word-identical) via
tools/upgrade_ninfer_v2_to_v3_fork.py (fork wrapper; also patched upstream's tool for
Windows IO + weight-only NVFP4 policies + codebook uses): models/*.v3.ninfer for
groupwise-int, nvfp4, nvfp4full, nvfp4qat. VERIFIED on the v3 engine (rebased/dflash2
build): official groupwise-int MTP3 29.2%; nvfp4full MTP3 41.76% — the EXACT v2
fingerprint; nvfp4full DFlash2-K7 with the NVFP4 module deterministic at 14.0% / 1.97
tokens-per-round; qat plain decode runs; nvfp4full text route answers correctly. NOTE: the
v2 originals remain beside the .v3 files; repoint presets/cards after soak. The old
v2-fingerprint campaigns (modelcard/paired baselines) need fresh runs before being quoted
as v3 results.

Dev rebuild: fresh fork base (f82b6468) + squashes msvc-test-constexpr, windows-port, webui,
mtp7, hyperquant, 1m-context, dflash2 (+presets cherry-pick). kernel-perf and its dev-only
companions intentionally absent this cycle. Dev BUILT and the serial suite is green:
113/122 with only the known ninfer_resource_manager_test planner flake failing (the
frontend-under-ctest flake passed this run). Tip d79d25b8 (+ squash(dev) hq-codec
alignment over c1ac1521).

## DFlash2 drafter campaign — CLOSED by owner (2026-09-15)

Owner decision: close with v3 stock selected (no artifact adopted); the W8 ceiling
finding is the final deliverable; the designated continuation experiment is the
full-module calibrated 4-bit re-encode (all 31 NVFP4 sites + codebooks from BF16,
coherently — owner: "whether a calibrated 4-bit re-encode can hold most of it at 1
GiB is exactly what that project would answer"). Final numbers: K1 43.82 stock vs
58.75 W8-ceiling (>=55 goal bar met by measurement, not adopted); QAT rounds r1-r4 +
checkpoint sweep all failed the gate (co-adaptation; K1/K7 opposition). Full record in
the campaign doc's closure block; the corpus' propose/verify/context streams, the
replica/trainer tooling, and all A/B evidence are retained; the per-layer bisection
dumps, the walk feature cache, and the round masters were deleted at owner cleanup
(the cache is regenerable from the corpus streams with the retained tooling; the
master weights were failed-round artifacts). The working tree is clean — the owner's
pre-session converter edits were reverted at owner direction, preserved at
~/Workspace/ninfer-src/converter-v3-edits-backup.patch.

## DFlash2 drafter campaign — MEASUREMENT AGENDA COMPLETE (2026-09-15)

FINAL FINDING: the existing W8-module lane (models/qwen3_8_27b_nvfp4full_w8.ninfer)
MEETS the acceptance goal bar — K1 58.75% (>=55, beats the 58.02 stretch; NVFP4 module
costs ~15pp at K1), with idle-gated paired end-to-end wins at K1 (+6.0%) and K2
(+12.0%, acceptance +10.28pp); K7 acceptance/t-r up (12.90/1.882), end-to-end neutral.
Adoption is an owner call (module-encoding change, +1.01 GiB device weights); the QAT
alternative (full-module re-encoding, all 31 sites + codebooks) now has a proven
ceiling and a quantified 15pp target at K1. All four QAT training rounds + the
checkpoint sweep remain negative (below). Evidence: quiet_w8_p{1,2,3}/ + w8ceiling_*.

## ACTIVE — DFlash2 drafter training campaign (owner-authorized 2026-09-15)

Goal: raise DFlash2 acceptance to the MTP lane's level (K1 43.82% -> >=55%, stretch 58.02;
best tokens/round -> ~2.03), via NVFP4 QAT (train the deployed function; STE on the exact
nvfp4_encode grid), on-policy data labeled by the VERIFY-PASS argmax, engine acceptance as
the only gate metric (matched A/B, acceptance AND end-to-end, module ships as a
word-splice revision). Charter and phase plan: the NEW CAMPAIGN section of
docs/maintainer/mtp-drafter-training.md.

Phase 0 EXECUTED on the gate corpus (v3 K7): the binder is CONTAINMENT — the full proposal
head's top-16 contains the verifier token only 31.3% pooled (52.9% at position 1, 20-35%
at 2-7), while the walk's selection-given-contained is already 47-92% and a perfect walk
adds only ~0.14 tokens/round. Phase-1 scope settled: QAT-fine-tune the 20 draft-layer NVFP4
parents (the representation feeding linear_topk), selector secondary. Acceptance semantics
recorded: greedy sparse acceptance compares draft j vs verify argmax at COLUMN j. New
env-gated instrumentation `NINFER_DFLASH_DUMP_DIR` (dflash_propose.bin / dflash_verify.bin
per round; --no-cuda-graph; removed from the working tree at campaign closure — the dump
format is documented here and in the campaign doc for re-implementation).
Evidence + analyzer: profiles/bench/dflash2-campaign-20260915/. Next: phase-1 torch
replica + NVFP4 STE harness (capture format above), then word-splice A/B via the
splice_rtncal_module.py pattern.

OWNER DIRECTIVES for phase 1 (campaign PAUSED before execution, 2026-09-15): training data
reuses the AI Hub corpus locally at `C:\Users\comet\Downloads\3.개방데이터\1.데이터`
(71903 multiturn + 71902 synthetic/instruction, each 원천데이터.zip — no re-download);
the learning process must reflect full real-world inference generation INCLUDING thinking
(captures and training rounds spanning entire generations across the full token range, not
overfit to early tokens). Recorded in the campaign doc's phase-1 directives.

Phase-1 ROUND 2 COMPLETE (2026-09-15): the round-1 replica misalignment was a POSITIONS
DUMP TRUNCATION (k of W block positions -> every rope angle off by one; fixed in engine +
dual-format parsers; replica hidden fidelity 10.9% -> 2.2%, q/k 29% -> 0.23%). Round-2
training on the corrected replica delivered the campaign's FIRST ENGINE GAIN: K1
43.82 -> 45.98% (+2.16pp); K7 regressed 12.05 -> 9.09 because the walk's
selection-given-contained collapsed (containment flat) — CE-through-the-head shifted
hidden out of the untrained selector machinery's calibration. ROUND 4 (walk-surrogate co-training, option iv) EXECUTED AND FAILED: engine-exact
walk math transcribed (edge = unary + <pred_emb(prev).(hidden@W_sel), succ_emb(cand)>,
greedy argmax), W_sel-only training on cached frozen features (walk_cache.pt) lifted
in-sample pick accuracy 53.5->71.3% but the engine gate went K7 12.05->10.12, K1
43.82->43.82 (zero transfer; 35.9% W_sel movement overfit AI Hub + recorded chains;
bench-gate distribution shift). ENGINE-IN-THE-LOOP CHECKPOINT SWEEP (epochs
1/2/5/10/20/40 spliced + benched, e2e_checkpoints/): decisive negative — epochs
1-2 re-quantize to stock words exactly; from epoch 5, K7 collapses ~4pp whenever
K1 gains anything; no checkpoint passes the gate (K1/K7 opposition = structural
co-adaptation). RECOMMENDATION: close the DFlash2 drafter campaign with v3 stock
selected; owner alternatives: large-corpus lever (low prior) or full-module QAT
from the z-lab BF16 source (new project). ROUND SERIES CLOSED at four failed
gates (r1-r4);
mechanisms: walk co-adaptation, walk-only overfit, gate-corpus mismatch. Further work
needs a large diverse on-policy corpus (CLI captures at scale) or engine-in-the-loop
selection — owner scope. No artifact adopted; v3 stock remains selected.
ROUND 3 EXECUTED (aligned data + drift penalty lambda=1000): FAILED BOTH BARS —
K7 6.93%, K1 44.32% (worse than r2's 9.09/45.98; penalty gradient war under Adam,
selector drift 1.06). Also discovered rounds 1-2 trained on rope-shifted data (the
trainer's position fix never applied — indentation variant missed). Round series:
stock 43.82/12.05; r1 42.22/6.86; r2 45.98/9.09 (best, masters overwritten — save
per-round masters); r3 44.32/6.93. No round passed the gate; mechanism = selector
co-adaptation dominates head-path CE gains. OWNER DECISION POINT for round 4:
walk-surrogate co-training (iv), SGD+lambda retune (v, low prior), position-1-only
aligned (vi). Evidence:
profiles/bench/dflash2-campaign-20260915/{e2e,phase0_qat_r2,qat_words,train_corpus_r2.log}.

Phase-1 context: the torch replica is BUILT and
near-validated (embedding bit-exact, per-layer 2-5% rel — an arithmetic floor; top-16
overlap 0.45), the QAT TRAINER is operational, the AI HUB CORPUS IS CAPTURED (100
conversations via the production CLI, thinking ON, full natural generations, DFlash2 K7:
37,207 rounds / 260,449 draft positions under aihub_capture/), and TRAINING ROUND 1 RAN
AND FAILED THE ENGINE GATE: replica CE 7.48->5.80 and replica agreement 5.19->8.18%, but
the engine regressed (K7 12.05->6.86%, K1 43.82->42.22%; masters moved only 0.6-1.6%
rel-L2). Diagnosis: the replica at 0.45 top-16 overlap is a misaligned surrogate — the
MTP proxy-misalignment mechanism one level down. ROUND-2 GO/NO-GO = replica fidelity:
either close the 2-5%/layer attention-path semantic gap (target top-16 overlap ~1.0) or
switch to engine-in-the-loop word selection; do NOT retrain against the current
surrogate. Artifacts: corpus + qat_words/masters.pt + train_corpus.log + e2e/qat_k*.json
(all under profiles/bench/dflash2-campaign-20260915/); the 19.4GB test artifact deleted.
Capture gotchas: sanitize invalid-UTF-8 rows + NFC; prompts via --messages JSON (Korean
through argv breaks the CLI's NFC). Tooling: capture_aihub.py / dflash2_{load,replica,
train}.py / splice_qat_module.py; replay discipline in the campaign doc.

## MTP drafter training — CLOSED (2026-09-15, engine-alignment gate failed)

[Experiment state and history](docs/maintainer/mtp-drafter-training.md) — read the
"EXECUTED AND CLOSED" block at the top; it supersedes the earlier W8-file localization.
Summary: the verify-vs-decode divergence for nvfp4full v2/v3 lives in the NVFP4 GDN input
record route (A16 gemv decode vs A16 small-T width 2..3 vs W4A4-materialized width >=4 —
NOT the W8 file the earlier session cited). Bit-exact per-column alignment was implemented
and dump-verified (`nvfp4_gemv_columns_kernel`, env `NINFER_GDN_VERIFY_COLUMN_ROUTE=1`,
single launch per GDN layer per round; plus an `NINFER_GDN_VERIFY_RECORD_A16=1` policy
variant, measured negative). Deterministic acceptance: v3 K2 +6.54pp / K3 +4.07pp, but
v2 K2 **-5.24pp** and v2 K3 pays ~7% round-time overhead — no setting improves v3 without
regressing v2, so the predeclared gate failed and the campaign closed with v3 as-is.
Evidence: `profiles/bench/verify-align-20260915/`. The prototype is env-gated in the
working tree (unset = stock; stock baselines 34.57/41.76/58.02/40.43/67.11/58.47 reproduce
exactly). Dump comparisons require `NINFER_BENCH_FUSIONS=0` (the prior session's runs had
fusions off; fusions-on changes the swiglu numerics and adds one-token warmup prefills).
NOT committed. Throughput cells from this session are NOT quiet-measured (desktop active).

DFlash2 community checkpoint assessed same day (owner request,
maurienne-ai/Qwen3.8-27B-DFlash2-NVFP4-RTNcal @ bd7a9342): NO import value for this engine —
its NVFP4 words are statistically identical to our own RTN module encoding (rel-L2 vs BF16
0.0949/0.0951/0.0959 on both sides across all 35 quantized sites) and its measured gain is
calibrated W4A4 activation scales, which our weight-only draft execution does not consume.
Applies equally to nvfp4full v2/v3 and nvfp4qat (same module encoding). The owner then
requested the real e2e test: `splice_rtncal_module.py` spliced their 20 per-layer module
parents bit-for-bit into a v3 copy (layout verified against the stock artifact words), and
the DFlash2 A/B measured K7 12.05 -> 11.67% (-0.38pp, tok/s -1.4%) / K1 43.82 -> 45.45%
(+1.63pp, tok/s +1.2%) — opposite-signed RTN-scale noise, slightly worse at the K7
operating point. NO ADOPTION. Evidence: `profiles/bench/dflash2-rtncal-20260915/`
(rel_l2.json + e2e/); test artifact and checkpoint download removed, splice tool kept in
tools/convert (uncommitted).

Closed-campaign disk cleanup executed (124 GB reclaimed, C: now ~149 GiB free): removed
session5/cocalibration/captures (47 GB), the six thinking_* features corpora and their
corpus safetensors, pairs/ (10 GB), all rejected .pt checkpoints (mtp_finetuned*, session3/4,
scale5504), and this session's diagnostic probe dumps. Preserved: all gate/evaluation
summaries, sequence_eval/acceptance_eval reports, capture logs and contracts, layerdump
evidence, splice scripts, rel_l2 evidence, verify-align-20260915, gates/, throughput/.

Read with AGENTS.md before planning work. Earlier records remain recoverable from
refs/backup/pre-sync-20260908/*, backup/20260908/all-refs.bundle, and
refs/backup/post-rebase-20260909/{dev,1m-context}. The branch tips before these checks are retained in
refs/backup/populated-20260909/{dev,1m-context}.

## nvfp4full v3 (MLP swap) local evaluation build (2026-09-11, NOT released)

Owner approved "MLP 교체 진행 + 품질 게이트 평가, 릴리즈 금지, 변경사항 로컬 유지". State:
working tree carries the converter re-source (recipe_nvfp4full.py NVIDIA ModelOpt adapter with
multiplier->divisor inversion `d=fl(1/scale_2)`, gate/up shared-scalar asserts, new preflight
proving the official allocation incl. lm_head NVFP4 + 208 scalar-FP8 sites; inventory
SOURCE_NVFP4_MLP_LAYERS=range(64) → 128 source/119 local; LocalDivisorTable tolerates the 16
retired MLP entries in the fixed calibration JSON; convert docstring/report note updated;
verify_nvfp4full needed no changes). NOT committed, NOT published; doc Section 14 and model
cards intentionally not rewritten (release-gated).

Critical discovery during the build: the HEAD converter produces the **W8-module** DFlash2 form
(recipe -v3, 20,550,864,896 B) while the published v2 artifact carries the **NVFP4 module**
(19,406,942,468 B, pre-rebase form with renamed directories). The gate needed MLP-only change
vs v2, so v3 was produced by SPLICING: copy v2, overwrite the 256 MLP objects from the fresh
build (`profiles/bench/nvidia-import-20260911/splice_v3_mlp.py`). The splice's integrity check
passed: all 1059 non-MLP non-DFlash2 objects in the fresh build are byte-identical to v2 —
the modified converter reproduces v2's text side exactly. Artifacts (all local):
`models/qwen3_8_27b_nvfp4full_v3.ninfer` (codes+NVIDIA d_x), `_v3b.ninfer` (codes+v2 d_x,
discriminating variant), `_v3_w8module.ninfer` (full converter output, kept for provenance).
Verification: 128 parents + 128 divisors word-exact vs NVIDIA source; format counts match v2.

GATES (evidence: profiles/bench/nvidia-import-20260911/gates/ incl. README):
- Acceptance (tg128 int8 8192, deterministic; v2 reproduces modelcard baselines exactly):
  MTP3 41.76% -> 34.57% (v3) — d_x-independent (v3b identical 195/564); DFlash2-K7
  12.74% -> 12.05% (v3), 8.61% (v3b — codes are co-calibrated with their d_x, import whole).
- Perplexity (fixed corpus quick, 261k tokens, int8): overall 4.7840 (v2) -> **4.7196 (v3)**
  = -1.35%, chinese/english_ref/code all better; v3b 4.7810.
- VERDICT: function quality UP, both speculative lanes DOWN (MTP3 ~= -9% decode throughput).
  Adoption unresolved — quality-vs-spec tradeoff for the owner; GPQA/LBV2 not run at this stage.
  END-TO-END THROUGHPUT (same day, quiet-discipline campaign, `run_throughput.ps1` +
  5-cell fixup with unique names; idle gate 3x<=5% every cell, DWM p95 <= 2.1%):
  MTP3 140.44/141.17 -> 127.94/127.51 (-8.90/-9.67%), DFlash2-K7 126.10 -> 120.97/120.75
  (-4.06/-4.24%), MTP0 control 88.03/88.07 -> 88.02/88.21 (-0.00/+0.15%). v2 MTP3/DFlash2
  absolutes match the modelcard-20260910 baselines (141.3/126.2). One contaminated v2
  DFlash2 cell (DWM 19.7%, 77.87 tok/s) was excluded and re-measured; the first campaign
  pass also overwrote pair-1 JSONs by filename (pair-2 survived) — fixup cells carry _p2
  suffixes. Tables in gates/README.md. MTP drafter check (owner question): the NVIDIA
  checkpoint's 15 mtp tensors are BIT-IDENTICAL to the official Qwen3.8-27B MTP (BF16,
  quantization-ignored) — no NVIDIA drafter tuning exists; our W8 MTP is the same weights.
  v3 MTP width sweep (acceptance deterministic; K cells' throughput not quiet-measured):
  K1 58.02% (v2 67.11), K2 40.43% (v2 58.47), K3 34.57% (v2 41.76); tokens/round math keeps
  K3 optimal for v3 (≈2.04 vs 1.81/1.58), so no width lever recovers the MTP loss. A BF16-MTP
  variant would be drafter-side second-order (W8 error ~0.4%) and needs binder work (MTP
  contract is W8) — low expected value, not attempted.
Converter run note: Windows mmap — safetensors single-part parents must be `.clone()`d before
the divisor read switches shards (fixed in materialize_source_nvfp4_weight); earlier segfaults
(139) were this. Throughput numbers in the gate cells are NOT quiet-measured (desktop busy,
stddev up to 20) — acceptance/PPL only.

## nvidia/Qwen3.8-27B-NVFP4 assessment (2026-09-11, pending owner decision)

NVIDIA published the official first-party ModelOpt NVFP4 checkpoint (released 09-08, main sha
`dbb8f445`, Apache-2.0, 3 shards ≈18.16 GB, producer modelopt `0.47.0.dev80+g913f5e224` per
hf_quant_config while the card says v0.48.0). Verified from metadata, quant summary, and safetensors
headers/4-byte scalar peeks (no full download yet):

- Recipe: NVFP4 on all 64 layers' MLP gate/up/down plus `lm_head` (193 sites), static per-tensor
  FP8 on all self-attn q/k/v/o and GDN in_proj_qkv/in_proj_z/out_proj (208 sites); embedding, MTP,
  Vision, GDN conv/A-B, norms left BF16 (`ignore: mtp*`). NVFP4 weights use `NVFP4MSECalibrator`
  (Local-Hessian calibration, 2,048 Nemotron-Post-Training-v3 samples); inputs are dynamic block-16
  E4M3 with a static per-tensor global scale — same execution semantics as our NVFP4 sites.
- Card quality vs BF16: GPQA-D 88.01/88.92, Terminal-Bench 74.02/75.56, AA-LCR 73.38/72.63,
  MMMU-Pro 74.86/75.14, SciCode 48.41/47.93, IFBench 78.93/80.07 (vLLM on GB300, W8A8 static
  activation quant at FP8 sites — our engine is weight-only there, so not directly transferable).
- Export naming is ModelOpt-native, not compressed-tensors: `weight` (U8 packed / F8_E4M3),
  `weight_scale` (E4M3 [N,K/16] natural block scales, or scalar F32 for FP8 sites),
  `weight_scale_2` (F32 scalar = amax/2688 multiplier), `input_scale` (F32 scalar multiplier).
  Ingestion deltas vs our contracts: scalar inversion d_w=fl(1/scale_2), d_x=fl(1/input_scale)
  (not word copies; ~1-ulp defined transform), per-tensor FP8 needs a new scalar-scale format
  (broadcast to BF16 row scales is lossy and violates preserve-words), NVFP4 `text/output_head`
  site + MTP/DFlash2 head aliases, MLP layers 56..63 FP8→NVFP4 range extension. gate/up share
  bit-identical `weight_scale_2`/`input_scale` (verified layer 0), so the fused gate_up parent and
  shared-divisor checks hold unchanged. FP8 sites also carry activation `input_scale` fields we
  would exclude, same as today's k_scale/v_scale.
- Estimated device weights ≈16.9 GiB (official nvfp4 18.98, nvfp4full 16.03): all-MLP FP4 + FP4
  head saves ≈2.1 GiB vs the registered profile, keeps FP8 attention/GDN (better per-site
  precision than nvfp4full's FP4 attention).
- Upstream master has not reacted (their recent work is MoE/W4A4 perf).

Recommendation delivered: evaluate as a fork-local additional profile (converter adapter + per-tensor
FP8 format + NVFP4 head site), not a rebuild of existing profiles; nvfp4full/nvfp4qat are different
designs and stay. Strong extra motivation: the registered profile's documented source
`unsloth/Qwen3.8-27B-NVFP4 @ 60e813d4` was force-pushed out of that repo (already recorded in
Section 14.2), so the registered `nvfp4` recipe currently has no recoverable pinned source;
NVIDIA's checkpoint is the stable first-party replacement candidate. Switching the *registered*
profile's source is upstream's call — propose with evidence after the fork-local evaluation. Card
copies of the raw metadata are under %TEMP%/nvfp4-card (throwaway).

Owner redirect (same session): provenance is a separate fixable issue; the live question is what
parts can be imported to improve nvfp4full. Analysis conclusion: (1) QUALITY — import all 64
layers' MLP NVFP4 codes/block-scales/site-scalars (Hessian/MSE-calibrated; replaces the unsloth
codes on 0..55, our worst-measured tensors at 0.107–0.126 rel-L2, and our local amax codes on
56..63 at ≤0.0951); zero contract/size change (same 247-parent inventory), pre-checkable offline
by per-tensor rel-L2 vs BF16 before any artifact build. (2) SIZE — the only real import is the
NVFP4 lm_head (−0.59 GiB, 16.02 → ~15.43 GiB device weights): needs NVFP4 head-site execution +
MTP/DFlash2 full-head alias updates; W8→FP4 precision drop at the head is a quality risk to gate
(GPQA/LBV2 lanes + deterministic MTP3/DFlash2 acceptance vs 41.76%/12.74%). (3) SECONDARY — their
2k-sample activation amax per site (amax = 448·input_scale for FP8 sites, 2688·input_scale for
FP4 sites) can cross-check/improve our 135 local d_x values (our corpus is only 10 docs / 2,690
tokens). NOT useful: per-tensor FP8 attention/GDN codes (bigger than our NVFP4, coarser than our
BF16 exceptions), embedding/MTP/Vision (left BF16 by NVIDIA). Remaining size levers are local,
not imports: 9 BF16 exceptions → NVFP4 would save another 0.73 GiB. If imports land: rebuild
nvfp4full v3 (third snapshot), rerun quality+paired bench, refresh HF cards (~19 GB republish).

MEASURED (2026-09-11, owner said "go measure it"): offline weight-space campaign complete.
Evidence: `profiles/bench/nvidia-import-20260911/` (rel_l2.json per-tensor + summary;
measure_rel_l2.py two-phase mlp/head, unbuffered, reuses the nvfp4_encode oracle and
quantize_matrix; decode convention validated: low-nibble-first fits 0.0935 vs 1.41 swapped).
NVIDIA source downloaded to `~/Workspace/ninfer-src/Qwen3.8-27B-NVFP4-nvidia` (21.9 GB, 3 shards).
MLP rel-L2 vs BF16, 192 tensors (64 layers x gate/up/down), metric identical to the Section 14.3
anchors:

| source | min | median | max |
|---|---:|---:|---:|
| nvidia Hessian/MSE | 0.0833 | 0.0861 | 0.1008 |
| our local amax profile | 0.0942 | 0.0949 | 0.0951 |
| unsloth (L0..55) | 0.1040 | 0.1073 | 0.1535 |

nvidia beats local on 190/192 (only L0/L63 down_proj lose, worst +6.8% relative) and unsloth on
168/168; median −9.3% vs local, −19.8% vs unsloth. Local MLP band 0.0942–0.0951 and unsloth
min/median reproduce the documented anchors; the unsloth max 0.1535 (L0 down_proj) exceeds the
doc's 0.107–0.126 band — that band likely reflected fused-parent or spot-checked tensors, not all
168 matrices. **MLP-code import verdict: green light** — replace all 64 layers' MLP codes+block
scales+site scalars. lm_head: nvidia FP4 0.0848 vs our W8 0.0055 (15x weight-space error for
−0.59 GiB): well-calibrated for an FP4 head but a pure size/quality trade — gate end-to-end
(GPQA/LBV2 lanes + deterministic MTP3/DFlash2 acceptance) before adopting. Next steps if owner
approves: converter source adapter (ModelOpt naming, scalar inversion d=fl(1/scale), same
blockscale permutation), nvfp4full v3 (third snapshot) build with MLP swap only, quality+paired-bench campaign;
head variant evaluated separately. Windows notes: safetensors get_tensor returns mmap views —
clone before handle close; holding all shard handles open hits paging-file error 1455; the head
must be measured in row chunks (248k rows OOM the 5090 in one decode).

QAT comparison added (same session, owner asked): measured the qat artifact's MLP parents directly
(`models/qwen3_8_27b_nvfp4qat.ninfer` — payloads are verify-proven word-for-word QUASAR source
copies; the HF cache snapshot is gone, no redownload needed). Fourth column, 192 tensors:
quasar_qat min/median/mean/max = 0.0838/0.0850/0.0854/0.1052 — beats nvidia PTQ on 117/192,
same band (median 1.3% lower); both dominate local amax and unsloth. Interpretation recorded:
rel-L2-to-BF16 is a proximity-to-teacher metric — valid quality proxy for PTQ (frozen rest of
network) but for QAT it measures training drift, not quality; QUASAR stayed teacher-close, so
weight space cannot rank QAT vs nvidia-PTQ (end-to-end lanes own that ranking), and QAT codes
must NOT be mixed with PTQ codes (co-adapted system). The qat profile's end-to-end deficit vs
nvfp4full is therefore attributable to its full-stack quantization design (all linears FP4, no
BF16 exceptions), not worse MLP codes. Unsloth-vs-local mechanism (throwaway diagnostic, script removed):
scale words follow the same RNE(blockmax/6) rule (98%+ match), saturation/code-at-max stats
identical (clipping ~1% of error energy — not the cause); differences are (a) unsloth d_w comes
from the FUSED gate+up amax, snapped to a coarse ~0.4% grid (stored 2752/6400/11584-class values;
L1 gate and up both store 11584 ≈ 2688/max(gate,up amax)=11614), and (b) the E2M1 code words match
an RNE reconstruction on the stored grid only ~50% with a systematic +0.6% recon-norm overshoot —
their code-selection procedure (or the exact source values quantized) differs from the
self-consistent RNE rule; this is where the extra error (0.107–0.1535 vs our 0.094–0.095) lives.
Our local profile's uniform band follows from block-max-anchored scales + self-consistent RNE:
error reduces to E2M1's intrinsic relative step, distribution-independent. Clarifier: the
local_amax column is the actual nvfp4full payload only on layers 56..63 (fused-parent encoding;
the per-half measurement is equivalent, same band); layers 0..55 actually carry unsloth words, so
the artifact's true MLP error is the mixed set: min/median/max = 0.0942/0.1070/0.1535, and the
nvidia swap improves 191/192 tensors against the artifact's actual payload (sole exception
L63 down_proj). Added at owner request: unsloth's FP8 row-scaled MLP 56..63 measured with
the same oracle (24 tensors, 0.0266 median — the registered-nvfp4-style tail), and the
three fork-lineage configuration rows now in the campaign README: unsloth-source
(NVFP4 0.1073 + FP8 0.0266, ~11.27 GB MLP), full-local NVFP4 (0.0949, ~9.63 GB), nvfp4full
v2 actual (0.0942/0.1070/0.1535, ~9.63 GB); NVIDIA dominates all three at the 9.63 GB
all-NVFP4 price point.

## Model-card rewrite and HF publish state (2026-09-10, later session)

Both model-card sources (`model-cards/Qwen3.8-27B-nvfp4full-NInfer/README.md` and the qat
one) were fully rewritten in the working tree (uncommitted) for DFlash2 support: a dedicated
DFlash2 section (module facts, NVFP4 payload 1,082,882,820 B vs W8 2,226,805,248 B,
acceptance 5.50/64.3% vs W8 5.75/67.9% nvfp4full, qat 2.50/21.4%), a new Engine support
section documenting the two required patches (hardcoded `resolve_weights` ID registration;
NVFP4-encoded DFlash2 module vs upstream W8G32_F16S) with the minimal patch-set table
(`feat/dflash2` + `feat/qwen3.8-nvfp4{full,qat}`, additive registration), minimal-set vs
full-fork capability scoping (hq-e8-2b/YaRN/WebUI are NOT in the minimal set), restored
provenance tables (z-lab/Qwen3.8-27B-DFlash2 @ 50307d4c — the old cards wrongly cited
"incoai"), QUASAR @ d8e6fbfa, and corrections: LongBench long envelope 786,432 (cards said
786,144), serve examples dropped `--webui` (feat/webui is outside the minimal set). Branch
analysis: the current stack is already the minimal patch set — no restructuring needed;
registration commits are purely additive (~140/~150 engine lines). Pending owner review of
the card text, then HF publish. HF state at check time: nvfp4full repo public with the
pre-DFlash2 v1 artifact (18,324,059,648 B, Aug 20) + v1 card; qat repo does not exist;
local v2 artifacts ready (`models/qwen3_8_27b_nvfp4full.ninfer` 19,406,942,468 B abb1e120,
`models/qwen3_8_27b_nvfp4qat.ninfer` 18,638,209,796 B 3bd37e03) plus conversion JSONs.
Publishing means ~38 GB of uploads (v1 stays retrievable at its revision) and creating the
qat repo; an OAuth token was in `~/.cache/huggingface/token` (export HF_TOKEN=$(cat …)).
Feature-branch card copies (feat/qwen3.8-nvfp4full/qat tips) still carry the pre-rewrite
cards; refresh them when committing the rewrite.

Owner follow-up (same session): drafter loading is optional, so the cards state the
per-option memory in the simple "Device weights" form the owner prefers. Six fresh bench
cells (`profiles/bench/modelcard-20260910/`, production ninfer_bench, Vision off; the JSONs
carry exact weights used_bytes) measured nvfp4full none/mtp3/dflash2-K7 device weights =
16.02/16.44/17.03 GiB (bytes 17,206,931,200 / 17,658,198,784 / 18,289,801,732) and qat =
15.31/15.73/16.32 GiB; steps +0.42 (MTP) and +1.01 (DFlash2 = the NVFP4 module payload).
The old cards' single "Device weights 16.03/15.31 GiB" rows correspond to the no-spec lane
(measured 16.02/15.31), so the quality tables now carry three "Device weights, <option>"
rows plus one explanatory sentence; the hq notes keep the ≈16/≈15.3 GiB constraint
framing. An earlier free-memory-after-weights table form was replaced at owner request.

Second owner follow-up: the cards' absolute tok/s rows (MTP3/DFlash2 decode, prefill;
99.05/161.49/5,895 and 96.27/119.12/6,147) were dropped as environment-specific and
non-comparable; the cards keep only same-workload relative figures (NVFP4 vs W8 module
5.50/64.3% vs 5.75/67.9%; qat 2.50/21.4% anchored against nvfp4full's 5.50/64.3%). If
absolute throughput is wanted in the cards, the comparable campaign design is: quiet desktop
per the measurement discipline, same binary, paired against the official
models/qwen3_8_27b_nvfp4.ninfer (MTP lane; it is the pre-suffix core form with no DFlash2
module) and against models/qwen3_8_27b_nvfp4full_w8.ninfer (DFlash2 lane, W8 module
encoding).

That campaign ran the same day (owner said "measure them" and the desktop went quiet;
first-cell idle gate 0/0/0%). 16 cells under
`profiles/bench/modelcard-20260910/paired/` (bench JSON + meta with idle samples and
during-run GPU util per cell, median 98–99% during decode; the throwaway campaign/sampler
scripts were removed after the run): tg128, greedy,
INT8, 8192 envelope, -r 3 --warmup 1, fusions/PDL on, two alternating pairs per
comparison. Decode tok/s means: MTP3 official 115.5/115.3, nvfp4full 141.3 (+22.4%,
pairs +22.5/+22.2), qat 125.3 (+8.7%, pairs +8.7/+8.6); DFlash2-K7 w8-splice 125.1,
nvfp4full 126.2 (+0.9%, +1.0/+0.8), qat 121.2 (−4.0% vs full, −4.0/−4.0). Acceptance is
deterministic per artifact and nvfp4full MTP3 41.76% exactly reproduces the earlier
campaign; DFlash2-K7 acceptance: w8 12.90%, full 12.74%, qat 10.96%, official MTP3 35.33%,
qat MTP3 32.12%. prefill_tok_s_mean is null in tg128 cells, so only decode and acceptance were tabulated.
At the owner's direction these paired results live HERE only — the cards carry no
throughput tables; their measurement section keeps artifact size, the per-option device
weights, and the note that absolute throughput is host/revision-specific, and the DFlash2
sections keep the same-workload acceptance comparisons. The frontmatter model-index
carries the three measured baseline rows (GPQA-Diamond, AIME 2026, LongBench v2 short)
for both cards; agentic suites stay out until measured and hq cells remain body-only.

Owner self-organizes the nvfp4full "Model Size" docs; session supplied the data (cells
under `profiles/bench/modelcard-20260910/{kv,vision}/`, scripts removed after the run):
device weights with `--vision` = lane + 295,711,648 B (vision/* section, 333 objects,
0.275 GiB) → none/mtp3/dflash2-K7 +vision = 16.30/16.72/17.30 GiB (CLI prints 3
significant digits; section sums reconcile the lane deltas to the byte: mtp/* 451,267,584
= the +0.42 step, dflash2/* 1,082,862,216 = the +1.01 step, text/* 16.358 GiB incl. the
optimized proposal head, frontend 12,837,177). KV payload at capacity 262,144, exact
bytes: int8 8,858,370,048 (33,792 B/token), fp8 8,657,043,456 (33,024), k8v4
6,744,440,832 (25,728), nvfp4 4,831,838,208 (18,432), hq-e8-2b 2,451,570,756 (9,352) —
int8/k8v4/nvfp4/hq match the populated-campaign figures. bf16 measured via CLI
`--no-cuda-graph` at 131,072 (bench bf16 hits cudaGraphExecUpdateFailure, the known
clean-upstream graph bug on this box): 8.00 GiB = exactly 65,536 B/token, so 16.0 GiB at
262,144 (does not fit beside the weights; ~186k-token practical bf16 ceiling).

The owner then hand-reorganized the nvfp4full card (own structure: condensed
Engine support, `## Size and quality` with subsections) and asked for the data in-card:
`### Model size` renamed to `### Device weights`, now carrying the vision-column weights
table and the KV-per-dtype table (a 24 GB-laptop highlight was added, then removed again at
owner request — no laptop mentions in the cards). The qat card still has the older
`## Quality and size` structure with the small measurement table; mirror the Device weights
section there on request (qat numbers: 15.31/15.73/16.32 + 0.275 vision; full stack ≈18.9
GiB at hq). Throughput tables remain out of both cards per the earlier owner decision.

Finalization (2026-09-11): the owner hand-tightened the nvfp4full card (condensed engine
support, `## Size and quality`, HyperQuant arXiv link + bibtex cite, full explicit eval
commands — `run_card_quality.sh` takes the kv-dtype as its optional third argument and
`run_card_lbv2.sh` takes explicit seeds; the lbv2 script encodes dtype per cell:
short=int8/262k, medium=hq/524k/yarn:2, long=hq/786k/yarn:3, with `lbv2_short_hq` in the
config but not the default cells array). The qat card was rewritten to the identical
structure (91f8a447) and nvfp4full finalized (55152a4f), both committed on their feat tips.
cometkim/dev was rebuilt by tree-replaying the previous validated squash commits (identical
trees, new parents) with the two card blobs overlaid from the tips, then cherry-picking the
dev-only fold/bench/presets/handoff commits; `git diff 03d7d377 cometkim/dev` shows only
the two card files. A from-scratch merge replay was abandoned: the old intermediate squash
trees carry deliberate whole-file resolutions (later-branch content pulled forward), so
clean replay reproduces different conflicts — reuse the validated trees instead. HF publish
(finalized cards + v2 artifacts + conversion JSONs, qat repo created) launched after the
pushes.

ninfer-windows minimal-patch PR (2026-09-11): natpate's master already carries the
nvfp4full registration (their PR #7, v1 artifact row) and upstream W8 DFlash2 (PR #8),
and documents DFlash2 as unsupported on nvfp4full — the gap is exactly our NVFP4 module
execution. PR #13 (https://github.com/natpate/ninfer-windows/pull/13) from origin branch
pr/natpate-dflash2-nvfp4-module: master@aa64ee53 + fb141405/085f3343/96acf990/23d34c77
cherry-picked + a README docs commit (v2 row, flipped note). Verified on their tree:
routes test OK, their ninfer.exe runs the published v2 artifact with --spec dflash2 K7
(exit 0, 7.50 tok/round on a greedy smoke prompt). The 23d34c77 bindings.cpp conflict
(nvfp4full binder vs the module helper) needed care: diff3 base sections tripped the
first resolution; final file = our post-23d34c77 version + their nvfp4full function and
dispatch case (verified: braces balanced, delta nvfp4full-only).

## Current state (2026-09-10)

Ports 1–11 are landed. The 1M-envelope investigation, standalone 1m-context re-port and dev
rebuild are complete. Both 786k and 1M presets boot and serve with the local WebUI. The idle
measurement campaign below supersedes old performance/acceptance records, including the
previously claimed cumulative +77% gain.
The optimized production engine now also completes a quiet, populated 1M request: 647.8
prefill / 14.3 decode tok/s with all four exact retrieval values (details below).

| branch | tip | base / role |
|---|---|---|
| master | a16b6442 | upstream semantic base |
| feat/msvc-test-constexpr | 62acee7e | master |
| feat/windows-port | 985907a4 | master; Windows/TMA/build |
| feat/webui | 81e077f4 | windows-port |
| feat/mtp7 | e3bf30db | master; MTP K1–K7 |
| feat/hyperquant | 5be7eb60 | windows-port; self-contained, PDL-free HQ |
| feat/1m-context | 7061557f | hyperquant; plain rmsnorm → rope; populated-cache bounds and HQ throughput fixed |
| feat/dflash2 | 90fd8a3b | windows-port; unified NVFP4 draft execution |
| feat/qwen3.8-nvfp4full | bc36e9c2 | dflash2 |
| feat/qwen3.8-nvfp4qat | c17ccc30 | dflash2 |
| feat/kernel-perf | 6c3fdbf4 | master; fusions, PDL, prompt/TU splits |

Dev starts at fork base **83ddf726**, with exactly one squash(feat/*) per branch in this order:
msvc-test-constexpr, windows-port, webui, mtp7, hyperquant, 1m-context, dflash2,
qwen3.8-nvfp4full, qwen3.8-nvfp4qat, kernel-perf. The integration fold retains PDL HQ/DFlash
forms and the splice tool. Dev-only supporting commits add same-binary measurement controls,
local preset budgets and session documentation; preserve these or fold their content deliberately.
The populated-history follow-up folds the long-cache fixes and qualified HQ optimization into
the 1M squash and preserves the subsequent integration forms (dev 01afc781 before the follow-up
documentation commit). Feature content f30372e2 is followed by its README tip commit.
The validated standalone build remains in ../ninfer-1m-report on codex/1m-report;
its two untracked Ninja wrappers are local build aids, not feature content.

Each feature has its own README tip commit. All ten intermediate squashes passed the README
skeleton-plus-union check for rows and bullets. Independent feature lineages mean a final
single-feature-to-dev diff includes sibling content: account for it and for qualified fused/PDL
integration forms explicitly. The rebuilt tree preserves the previous integration except the
intended 1M fixes/docs, measurement controls and presets. Removed the accidentally tracked
generated Testing/Temporary/CTestCostData.txt. Do not blindly reuse whole-file conflict
resolutions: cached resolutions can pull kernel-perf back into the standalone 1M branch.

## 1M decode investigation and correction

The historical ~3 tok/s log had **4.07 GiB unbanded workspace**, 13.3 GiB runtime reservation
and only 766 MiB device headroom. Its two full-capacity BF16 scratch planes account for 4 GiB
at 1M keys (2 × 1048576 × 4 heads × 256 × 2 bytes). The banded source uses **1.08 GiB active
workspace** at both 524k and 1M. The old executable was stale relative to that source. The evidence
points to memory pressure from the stale workspace allocation; WDDM spilling itself was not
captured in that historical run.

Whole-inference Nsight Systems node traces, nvfp4full/HQ, a short real prompt and 32 generated
tokens, compared 524288/yarn:2 and 1048576/yarn:4:

| quantity | 524288 envelope | 1048576 envelope |
|---|---:|---:|
| HQ kernel launches | 544 | 544 |
| HQ median kernel time | 21.919 µs | 21.887 µs |
| active workspace | 1.08 GiB | 1.08 GiB |
| non-speculative graph allowance | 12 MiB | 12 MiB |

At that measurement, 27B attention split capacity was capped at 85 (the later narrow HQ tile
below raises the single-token cap to 170). Per-round grids/splits follow live history and use
the same initial graph interval here; there is no current envelope-scaled decode explosion.
Initial 128-token envelope pairs measured 81.2/81.1 and 80.6/79.6 tok/s (524k/1M), establishing
that the old cliff is absent, not measuring long-history speed.

Attribution exposed a separate numerical bug: load-time execution_core() omitted the trailing
RoPE-frequency table from its aggregate initializer. Plain rope rejected zero tables; the fused
route silently captured identity rotations. ExecutionCore::rope_frequencies is now a required
reference, and every construction binds the owned table, including load-time graph capture.
The fix lives in the 1M feature and dev squash. Fresh paired measurements use the corrected
binary. Pre-fix token/acceptance fingerprints are stale.

The re-port carries WI-8 dither/window, YaRN/banding/lanes and retrieval content from
e648f178/8f79eaa9/4008d8de, adapted to hyperquant's plain interfaces, with RopeSide for single
query/key routes. It adds no fused qk_norm_rope operation or PDL dependency. CLI/serve docs
now describe rope-scaling and execution-envelope limits.

Verified on the standalone branch: rope + rope-scaling + HQ codec + all 26 HQ attention
scenarios pass; retrieval gives 5/5 unique top-1 needles at 2.250 bps.
Engine builds and runs a coherent 32-token groupwise-int HQ 1M/yarn:4 real-prompt
response through graphs. On dev, softmax-attention and qk_norm_rope suites pass with controls
enabled and disabled. All four fusion/PDL combinations produce identical greedy real-prompt
output and MTP3 acceptance (42/63; positions 18/14/10).

## Working 786k / 1M presets

qwen3.8-27b-nvfp4full-hq-e8-2b-786k.bat and the -1m.bat choose local models/webui when
present, otherwise auto-download. Passing both --webui and --webui-dir still triggers download;
the offline route must use only --webui-dir. The local bundle has 66 files. Both presets use
two pinned host-state slots and 1024 MiB host KV: the default eight slots plus 8 GiB pinned KV
failed allocation on this 32 GiB host. This retains the cache tier with an explicit local budget.

Live acceptance used exact preset engine flags with a loopback test port: /props, /v1/models
and a non-streaming chat completion succeeded on both. Vision is enabled; 786k also loads
DFlash2 K7, while 1M uses yarn:4 without speculation. Auto capacity reaches the full pool:

| preset | page groups | runtime reservation | free GPU memory |
|---|---:|---:|---:|
| 786432 | 12288/12288 | 8.93 GiB | 4.08 GiB |
| 1048576 | 16384/16384 | 10.6 GiB | 3.06 GiB |

The corrected 1M server decoded 64 tokens at **80.1 tok/s** from a 60-token request. This
proves execution at the envelope; it does not establish quality or speed with a populated
1M-token history. Synthetic codec retrieval is not an end-to-end long-context quality gate.

## Populated history validation

The populated-history follow-up uses the corrected code above. Both runtime variant objects have
valid Ninja header dependencies (116/117). Local scripts and evidence are under
`profiles/bench/populated-20260909/`: prepare.py creates prompts from the frozen benchmark
corpus using the artifact's embedded tokenizer/template; quality.ps1 runs the public CLI with
MTP0; score.py checks exact registry values and Engine token counts. The paired.ps1 campaign
uses combined prefill/decode, identical populated prompts, MTP0 and alternating KV profiles.

The first 524032-token model request exposed a real HQ decode bug: a leftover shared page array
had 64 entries, but long splits need more (about 97 pages at 524k with 85 splits). The HQ decoder
already reads physical pages directly, so the unused staging and reads were removed. A focused
operator reproducer failed at 524k before the change and passes through near-1M afterward.
All 26 original HQ cases plus 11 new sparse, nonzero long-cache oracle cases pass on dev and
the standalone 1M branch (worst new relative L2 0.002316). The new cases cover prompt carry,
cached prompt and small-T decode with reversed page tables and both head geometries.
Compute Sanitizer could not attach (launch timeout), so no sanitizer-clean claim is made.
The same 64-entry allocation remained in all five linear KV kernels despite the documented
524288-key/128-entry contract. Their staging is restored to 128 entries; cached, batched and
workspace entry points now consistently enforce the storage-specific ceiling. A pre-fix BF16
524288-key reproducer fails with cudaErrorIllegalAddress. All ten new sparse nonzero linear
cases (cached T1 and append T6 for each format) pass on both branches, and the full dev softmax
suite passes. The standalone linear test build used the existing MSVC constexpr-sqrt fixture
correction locally; that unrelated fixture change was restored before committing the feature.

The 524k, 786k and near-1M model requests complete with all four exact registry values on the
bounds-corrected binary, before the HQ throughput optimization below. These individual quality
requests are not paired KV comparisons or qualification of a later binary's exact token path.

| actual prompt tokens | YaRN factor | exact registry values | bare JSON | prefill tok/s | decode tok/s |
|---|---:|---:|---|---:|---:|
| 8192 | 1 | 3/3 | yes | 9150 | 71.0 |
| 262144 | 1 | 3/3 | code fence | 1820 | 25.9 |
| 524032 | 2 | 4/4 | yes | 978.1 | 15.8 |
| 786176 | 3 | 4/4 | yes | 669.9 | 10.6 |
| 1048320 | 4 | 4/4 | yes | 511.3 | 8.4 |

All needles are outside the BF16 residual window; higher-context prompts include a fourth
needle just beyond the 262144-key scratch-band boundary. These are synthetic retrieval probes,
not broad long-document reasoning scores. Active workspace remains 1.08 GiB from 262k through
1M (3.61 GiB free after startup at 1M). Engine token counts match the embedded-tokenizer
preparation exactly; no prompt reuse. The user's active follow-up is to assess single-5090
headroom and improve HQ's practical populated-history throughput. A same-binary 262k INT8/HQ
baseline is complete. The subsequent HQ optimization and populated KV comparisons are recorded
below; do not combine their rates with the earlier quality-run timings.

## HyperQuant throughput and single-5090 headroom

Whole-Engine Nsight Systems profiling at P262144/G32, nvfp4full/MTP0, identified HQ attention
as the relevant cost. HQ prompt attention accounts for 68.4% of GPU time and scratch decoding
for another 17.1%; the corresponding INT8 prompt attention takes 66.3 s versus HQ's 97.4 s plus
24.3 s scratch decoding. The model has 16 full-attention layers, 24 query heads, four KV heads,
and head dimension 256. Weight-kernel times are comparable between those routes. These are
attribution runs, not paired throughput baselines.

Nsight Compute counters are now accessible. At a 261120-key cached single-token HQ operator,
the old kernel uses 255 registers/thread, no spills, 5.08% DRAM throughput and 40.03% SM
throughput. About 59% of scheduler cycles have no eligible warp; fixed-latency dependencies
are a major stall source. It is not close to the 5090's memory-bandwidth ceiling. The prompt
scratch decoder separately saturates L2 (86.3%) while using only 4.44% compute: temporary Rice
symbols were being written to and reread from the global BF16 output plane.

The selected implementation keeps the codec, dither and residual semantics intact:

- Prompt decoding keeps temporary symbols in shared memory and uses aligned vector transfers
  for complete lattice words. Only final BF16 values reach the caller's bounded scratch planes.
- Single-token HQ attention uses a 16-key tile with four warps consuming disjoint 64-coordinate
  PV slices. Queries remain in shared memory; each thread keeps 32 output accumulators instead
  of 128. The 27B B1 split cap becomes 170; the alternate geometry is capped at 256. Active
  splits and the reducer use the same policy. Wider speculative tiles retain their prior route.

At the same profiler-controlled 2.40 GHz, the new single-token kernel takes 1.06 ms versus
1.73 ms, with 110 registers/thread, no spills, 8.79% DRAM throughput and 59.06% SM throughput.
This is a kernel-level improvement; substantial decode headroom remains. Heavy instantiations
stay in the per-geometry translation units. The production path has no HQ experiment switch;
the frozen local candidate binaries retain both variants for same-binary alternating A/B runs.

The [NVIDIA Blackwell specification](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf)
gives 209.5 dense BF16 TFLOP/s with FP32 accumulation and 1792 GB/s memory bandwidth for the
5090 at its rated boost clock. The current BF16 attention route needs approximately
`2 * 16 * 24 * 256 * N^2` useful FLOPs for a cold causal prefill. At N=1048320 this is about
216 PFLOPs: attention alone has an ideal lower bound near 17 minutes at rated boost, or
14.5 minutes at the observed 2.85 GHz clock. The earlier 34-minute complete prefill therefore
has useful attention work equivalent to roughly 42–50% of that compute ceiling. Several
thousand tok/s for a cold 1M prompt exceeds this arithmetic path's theoretical peak; short
prefills can reach several thousand tok/s because the attention work grows quadratically.
The quiet optimized run below takes 26m 58.4s for prefill, equivalent to 133.5 TFLOP/s of
useful attention work amortized over the whole prefill: about 54% of the clock-adjusted
ceiling at its sustained 2.827 GHz, or 64% of rated-boost peak. This accounting is a roofline
estimate, not a hardware-counter measurement or a claim that all remaining headroom is attainable.

[Upstream's published NVFP4-weight/INT8-KV result](https://github.com/Neroued/ninfer/blob/master/docs/performance/qwen3.8-27b.md#no-speculation-context-profile)
is 2203.1 prefill / 52.9 decode tok/s at 260096 prompt tokens. It does not publish a 1M result.
Its artifact, Linux serving workload and historical revision differ from this fork's Windows
nvfp4full benchmark. The matched local KV comparisons isolate HQ's cost; the published tables
alone cannot establish or exclude a fork-wide regression.

Measurement caution: an initial candidate Engine pair was contaminated by DWM consuming
35–38% of the 5090 after prelaunch idle checks had passed. That pair is inconclusive. Subsequent
runs monitor DWM throughout each process as well as checking idle GPU utilization before launch.
Do not reuse the contaminated `engine-narrow-*` rates as an optimization baseline.

Two monitored same-binary alternating old/new pairs, nvfp4full, MTP0, P8192 or P262144,
G128, chunk1024, fresh populated corpus, one measured request per process:

| actual prompt tokens | old prefill tok/s | new prefill tok/s | paired prefill gain | old decode tok/s | new decode tok/s | paired decode gain |
|---:|---:|---:|---:|---:|---:|---:|
| 8192 | 9515.2 | 9606.5 | +0.96% | 75.48 | 81.02 | +7.34% |
| 262144 | 1925.9 | 2238.5 | +16.23% | 27.95 | 36.99 | +32.36% |

The two 262k decode gains are +29.67% and +35.05%; its prefill gains are +15.83% and +16.63%.
The 8k prefill difference is small and should not be treated as a robust gain. Prelaunch GPU
checks pass; DWM's 95th-percentile utilization is at most 5% in all eight cells (at most 2% in
the 262k cells). Two brief 19–23% spikes occur during artifact loading, before the measured
prefill, with no such load during decode. This replaces the earlier apparent decode regression.
Local evidence: `engine-idle-narrow-*` reports and desktop samples. These measurements establish
the Engine-level improvement at the populated lengths shown. The complete 1M request is timed
separately below; the earlier 511.3/8.4 tok/s quality-run rates remain pre-optimization.

At 1048320 populated keys, two additional same-binary cached W1 operator pairs measure
6570.1 us before versus 4009.3 us after (1.64x throughput, 39.0% less time). These cold-cache
CUDA-graph timings cover the whole append-free attention Op, including its reducer, with
three warmups and 20 repetitions per cell. They establish that the decode improvement carries
through the 1M range; this operator comparison is distinct from the absolute Engine measurement.

On 2026-09-10, the owner made the desktop quiet and the unchanged optimized production CLI
completed the actual **1048320-token** retrieval prompt with **647.8 prefill / 14.3 decode
tok/s**. Cold prefill took **26m 58.4s**, followed by 5.0s of decode and 73 generated tokens,
ending on the stop token. All **4/4 registry values** are exact, returned as bare JSON; prompt
reuse is zero and the process exits 0. The workload is nvfp4full v2, HQ, MTP0, greedy,
YaRN:4, capacity/envelope 1048576, and the same frozen prompt used by the earlier 1M check.
Workspace remains **1.08 GiB**, with **3.76 GiB** free after startup.

Three prelaunch GPU samples are at most 5%. Throughout the process, 861 DWM samples across
all adapters have p95 **0%**, maximum **3%**. Sustained SM clocks are approximately
2.82–2.84 GHz; NVIDIA's final thermal-slowdown counters remain zero. DWM PIDs and adapter
LUIDs changed after reboot, so the monitor discovers current DWM processes instead of
reusing the previous hardcoded PID/LUID. Evidence is
`quality-c1048576-hq-e8-2b-graph-optimized-quiet20260910.*` and `quality-results.json` in
the populated-history profile directory; the local runner is `quality-quiet.ps1`.
This is one monitored absolute full-model baseline, not an alternating old/new pair. Do not
attribute the whole difference from the earlier 511.3/8.4 rates to the code optimization.

Final production-binary KV comparisons use the explicit nvfp4full v2 artifact, MTP0, G128,
chunk1024, a fresh populated corpus and capacity P+256. Each comparator has its own adjacent
HQ control: two alternating pairs at 8k and one pair at 262k, 18 successful process cells.
Three consecutive prelaunch GPU samples are at most 5%; DWM remains at most 5% throughout
every process. Rates below are cell means; the single long-context pair does not establish
fine rankings between close prefill results. Evidence is in `optimized-paired/` and
`optimized-summary.json` under the populated-history profile directory.

| prompt tokens | comparator | comparator prefill tok/s | HQ prefill tok/s | comparator decode tok/s | HQ decode tok/s | comparator KV GiB | HQ KV GiB |
|---:|---|---:|---:|---:|---:|---:|---:|
| 8192 | int8 | 10066.0 | 9640.4 | 85.31 | 80.90 | 0.266 | 0.106 |
| 8192 | k8v4 | 9984.1 | 9658.3 | 85.57 | 81.11 | 0.202 | 0.106 |
| 8192 | nvfp4 | 9862.8 | 9583.3 | 85.73 | 81.07 | 0.145 | 0.106 |
| 262144 | int8 | 3083.4 | 2246.7 | 59.24 | 37.73 | 8.258 | 2.285 |
| 262144 | k8v4 | 2541.9 | 2234.2 | 63.47 | 37.53 | 6.287 | 2.285 |
| 262144 | nvfp4 | 2318.8 | 2233.6 | 64.00 | 37.61 | 4.504 | 2.285 |

The owner's recalled HQ/INT8 ratio of about 80–85% is recorded at **32k**, not as a constant
across context lengths. The pre-reorganization HANDOFF (`70160d9e^`) records MTP0 P32768/G64
at INT8 69.6 and HQ 58.05 tok/s (83.4%); the pre-sync roadmap also identifies 32k explicitly.
Two current same-binary alternating pairs with P32768/G64 measure INT8 **81.66** and HQ
**73.66 tok/s (90.2%)**. Prelaunch idle checks pass; DWM peaks at 6% in one HQ cell and at
2–4% in the others, with consistent pair ratios. These current cells use nvfp4full, whereas
the archived table describes the official NVFP4 artifact; they are not an exact historical
artifact/corpus/toolchain rerun. The 63.7% ratio at 262k alone therefore does not demonstrate
a post-rebase regression against that 32k record, and these checks do not prove the whole
fork regression-free. Local evidence: `historical-shape-p32768-*`.

Historical source inspection finds the same cooperative Rice decoder and serialized
decode/barrier/MMA structure before the port; an asynchronous HQ pipeline was not lost.
The current FP32 partials are an intentional, oracle-qualified numerical profile. HQ's
remaining long-history gap is material: general-use performance remains the objective,
and the capacity savings do not make that gap an accepted performance target.

Production verification: both branch tips rebuild with the Ninja wrappers. HQ codec, all 44 HQ
attention cases (32 ordinary/nonzero-query and 12 long-cache cases), RoPE, rope-scaling and the
5/5 retrieval gate pass on the standalone branch and dev. The complete dev softmax-attention
suite also passes. Dev CLI, server and benchmark are rebuilt; the standalone CLI is rebuilt.
The optimized production CLI also returns all three exact registry values as bare JSON from
the 8192-token prompt (56 generated tokens, exit 0), and all four values in the quiet populated
1M request above. These retrieval probes do not establish broad long-document reasoning quality.
The feature stays free of fused qk_norm_rope and PDL dependencies. The rebuilt dev retains all
ten feature squashes and each intermediate README union. Final content changes relative to
the preceding integration are the populated-cache bounds fixes and this HQ optimization;
the plain/fused differences remain explicit, validated integration forms.

## Earlier idle seed-token measurements

The following tables record the prior dev 0d56ec5d binary, before the populated-cache fixes.
They remain evidence for those short-history workloads, not measurements of the new binary.

RTX 5090, sm_120a, Windows, MSVC 19.44.35228, nvcc 13.3.73, CMake 4.4.2, Ninja wrappers.
JSON reports CUDA runtime/driver API 13.4. Exact artifacts: models/qwen3_8_27b_nvfp4full.ninfer
and models/qwen3_8_27b_nvfp4qat.ninfer (v2).

One frozen ninfer_bench.exe, sequential processes, three consecutive GPU-utilization samples
≤5% before each process, one warmup + three measured repetitions, two alternating A/B pairs
per comparison. Command shape:

```text
--weights <artifact> -n 128 --max-ctx <envelope> --kv-dtype <profile> -r 3 --warmup 1 -o json
```

Add `--spec mtp --draft-tokens K` for K>0.
These are **tg128 seed-token workloads**: 8k/262k are allocated envelopes, not populated history
lengths. Acceptance is corpus-specific. Pair gains are means of B/A ratios, not historical records.

NINFER_BENCH_FUSIONS=0 selects plain q/k rmsnorm→rope and separate sigmoid output gating;
it does not disable upstream GDN fusion. NINFER_BENCH_PDL=0 removes dependent-launch
serialization attributes while retaining the same compiled kernels. Process-fixed controls
default to enabled; see bench/README.md.

Fusion/PDL contribution, nvfp4full MTP3 INT8. Each row is its own adjacent A/B comparison;
the range gives the two observed pair gains, not a confidence interval.

| envelope | change | A tok/s | B tok/s | mean paired gain | pair range |
|---|---|---:|---:|---:|---:|
| 8192 | fusion on, PDL off | 127.62 | 131.22 | +2.84% | +1.78 to +3.89% |
| 8192 | PDL on, fusion off | 130.33 | 130.72 | +0.30% | +0.10 to +0.49% |
| 8192 | PDL on, fusion on | 129.48 | 131.57 | +1.62% | +1.32 to +1.91% |
| 262144 | fusion on, PDL off | 128.49 | 129.77 | +1.00% | +0.50 to +1.50% |
| 262144 | PDL on, fusion off | 127.01 | 128.82 | +1.46% | -0.34 to +3.25% |
| 262144 | PDL on, fusion on | 129.68 | 129.93 | +0.20% | +0.02 to +0.38% |

Fusion yields a modest gain. PDL gains are small and vary with the comparison; the 262k
unfused pair changes sign between repeats. These results do not support the old cumulative
+77% attribution. All fusion/PDL cells retain 41.76% MTP acceptance.

MTP K1–K7 at the 8192 envelope, INT8, fusions and PDL enabled. Every K2–K7 cell is paired
with a fresh K1 control; K1 is shown beside each width instead of treating a distant control
as interchangeable. Round time includes the measured decode interval divided by MTP rounds.
K1 acceptance is 67.11% for nvfp4full and 60.76% for QAT.

| profile | K | paired K1 tok/s | K tok/s | paired gain | K ms/round | acceptance |
|---|---:|---:|---:|---:|---:|---:|
| nvfp4full | 2 | 122.66 | 140.07 | +14.19% | 15.49 | 58.47% |
| nvfp4full | 3 | 122.31 | 130.66 | +6.83% | 17.19 | 41.76% |
| nvfp4full | 4 | 122.65 | 130.73 | +6.59% | 18.47 | 35.41% |
| nvfp4full | 5 | 122.44 | 107.39 | -12.29% | 20.20 | 23.45% |
| nvfp4full | 6 | 123.60 | 106.76 | -13.62% | 21.80 | 22.29% |
| nvfp4full | 7 | 121.88 | 131.06 | +7.54% | 22.20 | 27.81% |
| nvfp4qat | 2 | 122.32 | 122.06 | -0.21% | 15.42 | 43.38% |
| nvfp4qat | 3 | 120.69 | 115.08 | -4.65% | 17.12 | 32.12% |
| nvfp4qat | 4 | 119.92 | 116.01 | -3.26% | 18.09 | 27.80% |
| nvfp4qat | 5 | 119.95 | 104.10 | -13.21% | 20.16 | 21.85% |
| nvfp4qat | 6 | 119.95 | 96.87 | -19.24% | 21.31 | 18.23% |
| nvfp4qat | 7 | 120.58 | 105.08 | -12.84% | 22.15 | 19.52% |

For this corpus, nvfp4full favors K2; QAT favors K1, with K2 effectively tied. Round time
increases gradually with width. The old large missing-route cliff claim is not supported
by these clean pairs; acceptance materially affects the resulting width ranking.

MTP0/MTP7 records, INT8, both controls enabled:

| profile | envelope | MTP0 tok/s | MTP7 tok/s | paired gain | K7 tokens/round | K7 ms/round |
|---|---:|---:|---:|---:|---:|---:|
| nvfp4full | 8192 | 81.18 | 128.64 | +58.46% | 2.909 | 22.61 |
| nvfp4full | 262144 | 81.28 | 128.14 | +57.65% | 2.909 | 22.70 |
| nvfp4qat | 8192 | 84.16 | 104.56 | +24.24% | 2.327 | 22.26 |
| nvfp4qat | 262144 | 84.73 | 104.97 | +23.88% | 2.327 | 22.17 |

HQ versus other KV profiles, nvfp4full MTP3, both controls enabled. Different KV codecs can
change greedy acceptance, so throughput combines kernel cost and the resulting token path.
The entire tg128 history fits in HQ's BF16 residual window; these HQ cells measure that
window route and allocation envelope, not steady-state compressed long-history attention.

| envelope | comparator | comparator tok/s | HQ tok/s | HQ paired gain | comparator acceptance | HQ acceptance |
|---|---|---:|---:|---:|---:|---:|
| 8192 | int8 | 129.46 | 134.45 | +3.86% | 41.76% | 46.25% |
| 8192 | k8v4 | 131.19 | 134.27 | +2.35% | 43.37% | 46.25% |
| 8192 | nvfp4 | 136.28 | 134.65 | -1.19% | 46.54% | 46.25% |
| 262144 | int8 | 128.50 | 133.72 | +4.06% | 41.76% | 46.25% |
| 262144 | k8v4 | 129.53 | 134.45 | +3.80% | 43.37% | 46.25% |
| 262144 | nvfp4 | 135.88 | 135.35 | -0.39% | 46.54% | 46.25% |

All 112 process cells completed successfully, with 336 measured repetitions. Prelaunch idle
checks passed for every cell. Small differences remain subject to run-to-run variation; these
short seed-token cells do not establish long-history throughput or model quality.

Local ignored provenance: profiles/bench/post-rebase-20260909/paired.ps1, summarize.py,
paired JSON/metadata, envelope traces, branch tests and preset responses. Historical loaded or
unpaired cells, width-cliff claims and old record comparisons must not be reused.

## Preserved ports and cautions

- Upstream remains the semantic authority for DFlash2 selector walk, K1..15 × B1..8, both heads,
  serving, host/prefix scheduling and the five original KV profiles. All fork ports are retained.
- Kernel-perf contains qk norm/rope and sigmoid-gate fusions, INT8 fp16-PV and INT8/BF16 prompt
  key splits, PDL publish-at-end, per-dtype/per-geometry launcher TUs and per-request error
  boundaries. Do not restore misordered BF16 partials from intermediate port history.
- V2 NVFP4 DFlash2 execution is complete through one unified binder. Module directories were
  renamed to upstream names in place; *.head.bak backups remain beside artifacts, payloads and
  offsets unchanged. The pre-rebase selector misread K16M128x4 scales as row-major; its old
  3.70 tok/round result is invalid evidence for preferring the former fork walk.
- models/qwen3_8_27b_nvfp4full_w8.ninfer is the v3 66-object W8 splice, made by
  tools.convert.qwen3_8_27b.splice_w8_module preserving base payloads; it is the measured
  W8-module DFlash2 ceiling artifact (K1 58.75%) — retained. The groupwise-int
  models/qwen3_8_27b.ninfer has no draft companion; official nvfp4 is the pre-suffix core form.
  Post-closure cleanup removed the rejected/intermediate artifacts (v3b, publichead,
  mtpdense_fullq6, mtpdenseqat, mtp_original_fullq6, ~99 GB) and their orphaned eval
  receipts; verdicts remain in the campaign docs. models/ now holds the six
  live artifacts: groupwise-int, official nvfp4, nvfp4full v2 + v3, W8 splice, qat.
  Full PR-form W8-module conversions are not prerequisites for the executable v2 files.
- HQ retains half-cell subtractive dither, BF16 sink32/recent512 residual window, absolute
  ownership epochs, rejected-draft invalidation and coherent prefix inheritance. Prompt scratch
  bands at 262144 keys with online-softmax carry. Linear KV caps at 524288; only 27B HQ permits 1M.
- BF16 CUDA-graph Engine execution fails cudaGraphExecUpdateFailure on clean upstream on this
  box; this is separate from the corrected RoPE binding. The old context-materialize replay
  failure was a test-only stream-ordering race, already fixed, not an engine defect.
- PDL producers publish after stores; consumers wait before dependent reads on all paths. No
  host synchronization/copies inside captures. Required aggregate members must not default away.

## Next work and build discipline

1. Further HQ throughput work should target its instruction-heavy entropy reconstruction.
   Complete-model paired optimization evidence is at 8k and 262k; 1M now has both an operator
   A/B comparison and an absolute quiet full-model baseline. Raising the cold-prefill arithmetic
   ceiling requires a separately qualified precision/path change, not a larger graph envelope
   or scratch allocation. Do not repeat the completed 1M baseline without a change or a live
   comparison that could alter the implementation decision.
2. On the next kernel-perf rebase, drop its duplicated Windows TMA fix (already in windows-port).
3. For long-context quality, use populated prompts on the corrected binary. Old coding/model-card
   campaigns need a fresh lane choice and rerun before being advertised as post-rebase results.

Use powershell -ExecutionPolicy Bypass -File configure-ninja.ps1 once per build directory and
after CMakeLists edits; then build-ninja.ps1 [-Target <name>]. No developer prompt or numeric
job limit. Converter Python: ~/Workspace/ninfer-src/cvenv/Scripts/python.exe.

A local Ninja cache had zero header dependencies on the runtime variant object, so a header fix
did not rebuild it. The MSVC include prefix must match compiler locale; inspect ninja -t deps
when incremental builds appear stale. Both runtime variants were explicitly rebuilt and now
have tracked header dependencies before this campaign's binary froze. A no-op link is not
proof that a changed template header reached the executable.

GPU jobs never overlap. Use hidden PowerShell Start-Process with stdout/stderr files, retain
Handle before WaitForExit (PS5), and check ExitCode. LNK1104 usually means a stray holds the
exe. Run ctest from build-ninja and compute-sanitizer.exe directly. Use apply_patch or file-based
UTF-8 Python scripts for C/C++ edits; Git Bash heredocs and escaped inline Python mangle
newline literals. Never implicitly download or regenerate local artifacts.
