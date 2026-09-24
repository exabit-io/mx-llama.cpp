# Branches of exabit-io/mx-llama.cpp (2026-09-24)

This repository is Exabit's only llama.cpp code repository for gfx906 (Radeon Pro Vega II / MI50). It is a fork of
[mxxm-t/mx-llama.cpp](https://github.com/mxxm-t/mx-llama.cpp), Marko Tombak's gfx906 fork of llama.cpp.

| branch | what it is |
|---|---|
| `master` | **the substrate**: mxxm-t's master merged with the latest llama.cpp release (today v0.5.0, `7fe450e`) + RCCL on by default. Offered to mxxm-t as [PR #17](https://github.com/mxxm-t/mx-llama.cpp/pull/17); once merged, `master` tracks mxxm-t's master. |
| `merge-v0.5.0` | the branch PR #17 was opened from (same commit as `master`); deleted when the PR is merged |
| `gfx906-both` | `master` + the patches binned as improving **both** the single-user and the multi-user profile |
| `gfx906-single` | `gfx906-both` + patches that improve the single-user profile only |
| `gfx906-multi` | `gfx906-both` + patches that improve the multi-user profile only |
| `gfx906-candidates` | `master` + Exabit patches not yet binned (this branch) |

Each branch's own commits are its bin: `git log gfx906-both ^master` is the `both` set. On every upstream release:
merge it into `master`, offer that to mxxm-t, then move the four `gfx906-*` branches onto the new `master`.
Every state is pinned by an annotated tag (`gfx906/v0.5.0/r0/*` for today's).

Build: `cmake -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx906`. `gfx906-both` and the branches on it default
`GGML_CUDA_FA_QUANTS=all` and `GGML_TP_AR_MAX_NE=20481`; `master` defaults `GGML_HIP_RCCL=ON`.

Measurements, plans and records: [exabit-io/llama.cpp-gfx906-tuning](https://github.com/exabit-io/llama.cpp-gfx906-tuning).
History before 2026-09-24 (the v0.4.1 and earlier branches): [exabit-io/llama.cpp](https://github.com/exabit-io/llama.cpp).
