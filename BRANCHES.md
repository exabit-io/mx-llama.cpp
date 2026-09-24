# Branches of exabit-io/mx-llama.cpp (2026-09-24)

This repository is Exabit's only llama.cpp code repository for gfx906 (Radeon Pro Vega II / MI50). It is a fork of
[mxxm-t/mx-llama.cpp](https://github.com/mxxm-t/mx-llama.cpp), Marko Tombak's gfx906 fork of llama.cpp.

| branch | what it is |
|---|---|
| `master` | **the build**: the substrate + the Exabit patches binned as improving **both** the single-user and the multi-user profile |
| `gfx906-single` | `master` + patches that improve the single-user profile only |
| `gfx906-multi` | `master` + patches that improve the multi-user profile only |
| `gfx906-candidates` | the substrate + Exabit patches not yet binned (this branch) |
| `merge-v0.5.0` | **the pure substrate**: mxxm-t's master merged with llama.cpp v0.5.0 (`7fe450e`) + RCCL on by default, offered to mxxm-t as [PR #17](https://github.com/mxxm-t/mx-llama.cpp/pull/17); deleted once merged, after which mxxm-t's own `master` is the substrate |

The bins are git queries: `git log master ^merge-v0.5.0` is the `both` set, `git log gfx906-single ^master` the single-user
set. On every upstream release: merge it into mxxm-t's master on a `merge-vX` branch and offer that to mxxm-t; merge the
branch into `master`; move `gfx906-single`, `gfx906-multi` and `gfx906-candidates` onto the new `master`.
Every state is pinned by an annotated tag (`gfx906/v0.5.0/r0/*` for today's).

Build: `cmake -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx906`. `master` and the branches on it default `GGML_HIP_RCCL=ON`,
`GGML_CUDA_FA_QUANTS=all` and `GGML_TP_AR_MAX_NE=20481`.

Measurements, plans and records: [exabit-io/llama.cpp-gfx906-tuning](https://github.com/exabit-io/llama.cpp-gfx906-tuning).
History before 2026-09-24 (the v0.4.1 and earlier branches): [exabit-io/llama.cpp](https://github.com/exabit-io/llama.cpp).
