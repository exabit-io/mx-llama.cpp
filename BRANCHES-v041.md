# The v0.4.1 branch topology (2026-09-20 .. 22)

Everything below sits on a **common base of v0.4.1**, because patches expressed against one base are
composable terms: leave-one-out attribution and build composition are only valid that way.

| branch | what it is |
|---|---|
| `gfx906-substrate-v041` | v0.4.1 + the mx-llama.cpp substrate (157 commits, **squash-merged**) + five build-failure resolutions + the 29 Exabit terms as `terms/*.patch`. **Builds clean and gated**: `test-backend-ops` 16198/16198 on all four dies, 16K/6 perplexity 5.6122 ± 0.062 (inside the reference cluster). This is the survey's base `R`. |
| `c4-series` | the substrate plus our 20 applied terms — the arm measured across 276 cells. Group verdict: **+2.79% decode** at 4×64K over the substrate, nothing measurable at 1×254K. |
| `gfx906-both` | **destination bin** for patches confirmed to improve *both* axes. Holds one confirmed term: `31c734dd9`, the `GGML_TP_AR_MAX_NE` size gate. |
| `gfx906-single`, `gfx906-multi`, `gfx906-required` | destination bins, currently **at the substrate** — empty by design until verdicts fill them. |
| `backup/gfx906-{multi,single}-presurvey-20260920` | the pre-survey tips of those branches, preserved before they were reset to be bins. They forked *before* the five build fixes and did not compile. |
| `backup/gfx906-pre-v041-20260920`, tag `gfx906-pre-v041-20260920` | the pre-rebase state. |

## Why the substrate is a squash, and what it costs

`git merge --squash` was chosen over replaying the fork's 148 commits because sequential replay fights
every intermediate state against upstream churn: **11 conflict hunks total for the squash, versus 30 in
the fork's first commit alone.** Measured cost of that choice: only **53 of the fork's 138 code commits**
reverse-apply against the squash as discrete delta-minus units; the other 85 are separable only as
**12 feature groups**. A forward replay onto v0.4.1 was measured at **131 hard conflicts**, so per-commit
separation is not realistic.

## Build requirements — not optional

    -DGGML_HIP_RCCL=ON                                  # REQUIREMENTS R3.11; upstream default is OFF
    -DGGML_CUDA_FA_QUANTS=f16-f16;q8_0-q8_0;q8_0-q4_0   # the q8_0-q4_0 kernel is required
    -DGGML_HIP_GRAPHS=ON -DGGML_NATIVE=ON

With RCCL off, `GGML_USE_NCCL` is undefined, nothing links `librccl`, and the collective silently falls
back nccl → internal (which needs `n_devices == 2`) → none → **meta-backend butterfly**, costing
**18.6% prefill**. An FA combination that is not compiled does **not** fail — it silently runs on a
slower generic path. Both failure modes are invisible at runtime, which is why
`tools/assert-build-config.sh` and `tools/assert-arms-comparable.sh` in the benchmarking repo exist and
run before any measurement.

Verdicts, raw data and method: https://github.com/exabit-io/llama.cpp-gfx906-tuning
(`survey/`, `data/raw/2026-09-19..22/`, `RESURVEY-ACTION-PLAN.md`).
