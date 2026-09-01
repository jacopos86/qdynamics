# New machine bootstrap

Target hardware (verified on the replacement machine, 2026-08-31): Apple
M4 Pro, 14-core CPU, 20-core GPU, 24 GB RAM, 1 TB SSD, macOS 26.6.2. (An
earlier draft said M3 8-core; the machine that actually arrived is the
M4 Pro.) Implications:

- **RAM invariant (user-set 2026-08-31): 18 GB aggregate ceiling** for all
  agent-launched work, per `agent_guidance/shared/memory-budget.md` (revised
  from the lost 16 GB machine's 10 GB). Single process tree: 12 GB default
  guard, at most 17 GB for a 16 GB-profile lane (e.g. the Paper II t=10
  comparators, now locally feasible one-at-a-time). Many-job campaigns still
  go to CHTC — one heavy local job at a time remains the rule.
- **Disk:** 1 TB comfortably holds the full CHTC rescue fetch (~65–78 GB,
  `CHTC-FETCH.md`) alongside the repo.
- **CPU:** 14 cores; RAM, not CPU, is the binding constraint for the heavy
  integration tests — run the full suite serially; parallel workers multiply
  resident memory.
- **GPU:** irrelevant to this stack (qiskit-aer CPU path).
- **Migration note (superseded):** an earlier draft assumed Migration
  Assistant would carry over `~/.ssh`, `~/Holstein_test`, and uv. The
  machine actually arrived fresh — nothing carried over. All steps below
  were executed from scratch on 2026-08-31 (agent session): GitHub auth
  via `gh auth login --web` over HTTPS (no SSH key on this machine; the
  SSH path in step 1 remains valid as an alternative), clone, uv env,
  verification (collection 7,535 tests / 9 known errors — exact match to
  the recovery baseline), GitNexus rebuild. Toolchain lives user-local
  under `~/.local` (git via CLT, gh 2.98, node 24 LTS, uv, no Homebrew,
  no sudo required).

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

Expected as of 2026-08-31 (verified on the M4 Pro): 7,535 tests collected,
9 known errors (`RECOVERY-AUDIT.md` §9; an earlier draft said ~6,519/10).
Then a targeted smoke:

```bash
cd ~/Holstein_test && .venv/bin/python -m pytest test/test_hh_staged_noise_workflow.py test/test_ap_mclachlan_exchange_selector.py -q
```

Known-red baseline (re-measured and classified on the M4 Pro, 2026-08-31 —
full accounting in `RECOVERY-AUDIT.md` §12): 7 pre-existing failures in
`test_ra_adapt_semantic_closure_routes.py` (confirmed: 7 failed / 85
passed); 9 data-missing failures across the staged suites on the vanished
`output/paper_ii_staged` exact references; 1 `paper-ii-run/SKILL.md`
doc/test mismatch (user decision pending); 1 LaTeX-engine env failure
until TeX is installed. The 7 era-drift failures previously recorded did
**not** reproduce (smoke files 70/70), and the 16 `search_profile`
TypeErrors found during re-measurement were a recovery-seam regression,
fixed on `main` (see §12).

The full suite has not been run end-to-end since recovery — heavy integration
tests; respect the 18 GB RAM invariant (`AGENTS.md` Hard invariants,
`pipelines/shell/ram_guard.py`). Running it once on this machine and
recording the result in `RECOVERY-AUDIT.md` is the first useful task.

## 5. GitNexus agent tooling (do this — the repo's contracts assume it)

`CLAUDE.md` carries MUST-rules (impact analysis before edits, detect_changes
before commits) that assume a live GitNexus index, and the worklogs show it
was load-bearing in real campaigns (e.g. the Route C gated-dead verdict). The
gitignored `.gitnexus/` index and `~/.gitnexus/` registry were lost with the
old machine — rebuild:

1. Install Node.js 20+ (nodejs.org installer, or Homebrew if present).
2. From the repo root:

```bash
cd ~/Holstein_test && npx gitnexus analyze
```

   On npm 11.x, if `npx` crashes during install (`node.target is null`):
   `npm i -g gitnexus` then `gitnexus analyze`
   (workaround per `.claude/skills/gitnexus/gitnexus-cli/SKILL.md`, GitNexus #1939).
3. Register the GitNexus MCP server in Claude Code (the gitnexus CLI's own
   setup output / `npx gitnexus --help` names the command), then restart
   Claude Code so the server loads. `.claude/skills/gitnexus/` documents the
   tools once live.
4. Sanity: `node .gitnexus/run.cjs status` — expect the repo registered with
   ~97k symbols (CLAUDE.md's counts are from the old index; new counts will
   differ slightly after the recovery grafts).

Until this is done, treat CLAUDE.md's GitNexus MUST-rules as inoperative
(AGENTS.md's fail-soft rule for absent adapters applies).

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
