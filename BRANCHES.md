# Branch topology (current, 2026-09-24)

**Substrate: [`exabit-io/mx-llama.cpp` branch `merge-v0.5.0`](https://github.com/exabit-io/mx-llama.cpp/tree/merge-v0.5.0)
(`528384980`)** — mxxm-t's gfx906 fork ([`mxxm-t/mx-llama.cpp`](https://github.com/mxxm-t/mx-llama.cpp) @ `eefc4e732`,
its 182 commits kept individually) merged with llama.cpp **v0.5.0** (`7fe450e`), plus RCCL on by default. It is the
substrate for every gfx906 build here until mxxm-t merges it
([mxxm-t/mx-llama.cpp#17](https://github.com/mxxm-t/mx-llama.cpp/pull/17)); after that, mxxm-t's `master` is.

| branch | commit | contents |
|---|---|---|
| `gfx906-required` | `528384980` | the substrate, exactly |
| `gfx906-both` | `a23e12438` | required + the `GGML_TP_AR_MAX_NE` size gate (default 20481) + `GGML_CUDA_FA_QUANTS=all` default — the patches binned as improving **both** profiles |
| `gfx906-single`, `gfx906-multi` | `a23e12438` | gfx906-both + the patches binned for that profile only (none yet) |
| `c4-series` | this branch | the Exabit code patches on the substrate, awaiting their bins |

Each branch's own commits are its bin: `git log gfx906-both ^gfx906-required` is the `both` set. The branches move; every
state they pass through is an annotated tag (`gfx906/mx-merge-v0.5.0/*` now; `gfx906/v0.5.0+mxxm-t-eefc4e732/*`,
`gfx906/v0.5.0/*`, `gfx906/v0.4.1/*`, `import/*` before).

## Build

    cmake -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx906

`gfx906-both` and the branches built on it default `GGML_HIP_RCCL=ON` and `GGML_CUDA_FA_QUANTS=all`; both are required
(an uncompiled FlashAttention K/V pair silently falls back to f16, and without RCCL tensor split falls back to the
meta-backend butterfly). `GGML_TP_AR_MAX_NE` defaults to 20481 because the substrate turns the custom AllReduce on by
default and its own 262144 threshold costs ~19% prefill at 4 x 64K.

`BRANCHES-v041.md` describes the earlier v0.4.1 topology and are kept as history.
