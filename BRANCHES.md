# Branches of exabit-io/mx-llama.cpp (2026-11-27)

This repository is Exabit's llama.cpp for gfx906 (Radeon VII, Instinct MI50/MI60, Radeon Pro Vega II) and Exabit's only
llama.cpp code repository. It began as a fork of [mxxm-t/mx-llama.cpp](https://github.com/mxxm-t/mx-llama.cpp), Marko
Tombak's gfx906 fork of llama.cpp, and carries his gfx906 work. Since v1.1.0 it is maintained as a project of its own:
it follows upstream llama.cpp releases, keeps the gfx906 patches (his and Exabit's), and takes later changes from his
fork on their merit.

| branch | what it is |
|---|---|
| `master` | **the build**: the substrate + the Exabit patches binned as improving **both** the single-user and the multi-user profile |
| `gfx906-single` | `master` + patches that improve the single-user profile only (none at present: equal to `master`) |
| `gfx906-multi` | `master` + patches that improve the multi-user profile only (none at present: equal to `master`) |
| `gfx906-candidates` | the substrate + RCCL on by default + Exabit patches not yet binned (this branch) |
| `merge-v1.1.0` | **the substrate**: mxxm-t's master `557a6068f` (his own merge of llama.cpp b12500, 2026-11-18) merged with llama.cpp v1.1.0 (`1715275`, b12774), plus three fixes (the fused top-k MoE softmax on 64-lane rows, the tile-kernel selection for 576/512 attention, the carried MTP row of a sequence that starts at position 0). Offered to mxxm-t as [PR #25](https://github.com/mxxm-t/mx-llama.cpp/pull/25) |
| `merge-v1.0.0`, `merge-v0.9.0`, `merge-v0.7.0`, `merge-v0.5.0` | history: the earlier substrates, offered as PRs #24, #23, #22 and #17 |

The bins are git queries: `git log master ^merge-v1.1.0` is the `both` set (its first commit turns RCCL on by default,
which mxxm-t keeps off), `git log gfx906-single ^master` the single-user set. On every upstream release: merge it into
the previous substrate on a `merge-vX` branch, together with what mxxm-t's fork has added since where it serves gfx906,
and move the Exabit commits of `master`, `gfx906-single`, `gfx906-multi` and `gfx906-candidates` onto it. Whether
following upstream costs anything against his fork is measured for each release before it is packaged (v1.1.0: the same
patches on v1.1.0 and on his b12500 tree are within half a percent of each other in every cell timed). Every state is
pinned by an annotated tag (`gfx906/v1.1.0/r0/*` for today's, `gfx906/v1.0.0/r0/*` to `r2`, `gfx906/v0.9.0/r0/*`,
`gfx906/v0.7.0/r0/*`, `gfx906/b11553/r0/master` and `gfx906/v0.5.0/*` for the previous cycles).

History of the merges offered to mxxm-t: [PR #17](https://github.com/mxxm-t/mx-llama.cpp/pull/17) (v0.5.0) was closed
when he merged b11553 himself on 2026-10-13. PRs #22 (v0.7.0), #23 (v0.9.0) and #24 (v1.0.0) were not merged. He merged
b12500 himself on 2026-11-18; #25 (v1.1.0) starts from that tree and is open.

Build: `cmake -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx906`, with `HF_UI_VERSION=b12774` in the environment (the web
UI assets are downloaded at build time by upstream build number, and a fork's commit count names none). `master` and
the branches on it default `GGML_HIP_RCCL=ON`, `GGML_CUDA_FA_QUANTS=all` and the max-ilp scheduler, and set the size
gate between the fork's custom all-reduce and RCCL to 16 rows at four GPUs (`GGML_TP_AR_MAX_NE` unset = 81921; 20481
at two and three GPUs). This branch keeps each unbinned patch as it was written and has none of those defaults, nor
`master`'s MMQ from 9 columns (its MMVQ window is 16 columns): build it with ROCm 10.0's compiler.

Measurements, plans and records: [exabit-io/llama.cpp-gfx906-tuning](https://github.com/exabit-io/llama.cpp-gfx906-tuning).
History before 2026-09-24 (the v0.4.1 and earlier branches): [exabit-io/llama.cpp](https://github.com/exabit-io/llama.cpp).
