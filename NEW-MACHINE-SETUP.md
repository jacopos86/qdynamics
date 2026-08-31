# New machine bootstrap

Contract for a fresh agent session on a new machine. Execute top to bottom;
each step states its verification. Context: this repo was recovered on
2026-08-31 after the previous machine was lost — `RECOVERY-AUDIT.md` has the
full account; `AGENTS.md` is the agent router. Update the "Active checkout"
path in `AGENTS.md` once cloned.

## 1. GitHub auth (repo is private)

If `git ls-remote git@github.com:20jastrobel/Holstein_test.git` fails:

```bash
ssh-keygen -t ed25519 -C "20jastrobel@gmail.com" -f ~/.ssh/id_ed25519 -N "" -q && cat ~/.ssh/id_ed25519.pub
```

Give the printed public key to the user to add at
https://github.com/settings/ssh/new (the private half never leaves the
machine). Machines migrated with Apple Migration Assistant already carry a
working key in `~/.ssh`.

## 2. Clone

```bash
git clone git@github.com:20jastrobel/Holstein_test.git ~/Holstein_test
```

`main` is canonical (promoted 2026-08-31). Do not delete any branch; several
are sole-copy archives (see `RECOVERY-AUDIT.md` §5b).

## 3. Python environment

System Python is too old. Use uv (user-local, no admin):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```bash
cd ~/Holstein_test && ~/.local/bin/uv python install 3.12 && ~/.local/bin/uv venv --python 3.12 .venv && ~/.local/bin/uv pip install --python .venv/bin/python -r requirements.txt
```

## 4. Verify

From the repo root (imports require cwd = root):

```bash
cd ~/Holstein_test && .venv/bin/python -m pytest test --collect-only -q 2>&1 | tail -1
```

Expected as of 2026-08-31: ~6,519 tests collected, 10 known errors
(`RECOVERY-AUDIT.md` §9). Then a targeted smoke:

```bash
cd ~/Holstein_test && .venv/bin/python -m pytest test/test_hh_staged_noise_workflow.py test/test_ap_mclachlan_exchange_selector.py -q
```

Known-red baseline: 7 era-drift failures in the resurrected staged-noise
suites, 7 pre-existing failures in `test_ra_adapt_semantic_closure_routes.py`.

The full suite has not been run end-to-end since recovery — heavy integration
tests; respect the 10 GB RAM invariant (`AGENTS.md` Hard invariants,
`pipelines/shell/ram_guard.py`). Running it once on this machine and
recording the result in `RECOVERY-AUDIT.md` is the first useful task.

## 5. Optional: GitNexus agent tooling

Only needed for the GitNexus skills/MCP referenced in `CLAUDE.md`:
install Node.js, then from the repo root `npx gitnexus analyze`, and register
the GitNexus MCP server in Claude Code. Until then treat CLAUDE.md's GitNexus
MUST-rules as inoperative (the index and registry were lost with the old
machine).

## 6. Smoke-run entrypoints (optional)

```bash
cd ~/Holstein_test && .venv/bin/python pipelines/exact_bench/cross_check_suite.py --problem hh --L 2 --omega0 1.0 --g-ep 0.5 --n-ph-max 1
```

Paper II dynamics runners need an ADAPT artifact JSON produced first
(`pipelines/static_adapt/adapt_pipeline.py --skip-pdf --output-json ...`);
all gitignored `artifacts/` evidence from before 2026-03 lives only on
`archive/main-pre-recovery` (curated subset re-tracked on `main`).

## 7. Open items carried forward

- Fetch CHTC submit-node archives before purge:
  `/home/jsstrobel/Holstein_paper_ii_staged_accuracy_euler_v1_*/...tar.gz`
  (paths recorded in the receipts under `chtc/paper_ii_staged_accuracy_euler_v1_*`).
- Deferred decisions: Paper I code lineage (`paper-i-integration-20260831`),
  Q45 code re-application, 10 residual collection errors — `RECOVERY-AUDIT.md` §6.
- User-level skills from the old machine ($journal-math-manuscript-refiner,
  $gpt-pro-handoff, $pedagogical-math-primer) were never in the repo and
  remain lost unless recovered from another source; per `AGENTS.md`, absent
  adapters fail soft.
