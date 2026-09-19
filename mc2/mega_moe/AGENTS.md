# AGENTS.md — mega_moe

## What this is

`mega_moe` is a single Ascend C fused MoE operator inside the `ops-transformer` monorepo (`mc2/` module). It fuses Dispatch + Linear1 + SwiGLU + Linear2 + Combine into one operator with communication-computation overlap.

This directory is **not** a standalone project. All build/test commands run from the **repo root** (`ops-transformer/`).

## Build commands (run from repo root)

```bash
# Build mega_moe host libs + kernel for a specific SoC
bash build.sh --ophost --opapi --opkernel --ops=mega_moe --soc=ascend910b -j16

# Build for Ascend 950 (arch35)
bash build.sh --ophost --opapi --opkernel --ops=mega_moe --soc=ascend950 -j16

# Build only the mc2 module (includes mega_moe)
bash build.sh --ophost --module=mc2 --soc=ascend910b -j16

# Build run package
bash build.sh --pkg --soc=ascend910b --ops=mega_moe

# Build torch_extension wheel only
bash build.sh --torch_extension_only --ops=mega_moe
```

## Test commands (run from repo root)

```bash
# Run ophost UTs for mega_moe
bash build.sh --ophost_test --ops=mega_moe --soc=ascend910b

# Run opapi UTs
bash build.sh --opapi_test --ops=mega_moe --soc=ascend910b

# Compile UTs without executing
bash build.sh --ophost_test --ops=mega_moe --noexec

# Run a specific example (eager mode)
bash build.sh --run_example mega_moe eager
```

## Lint / format

```bash
# C++ formatting (clang-format, config at repo root .clang-format)
clang-format -i -style=file <file>

# Python linting (ruff)
ruff check --fix <file>
ruff format <file>

# Pre-commit (runs all hooks)
pre-commit run --files <files>
```

## Directory structure

| Path | Purpose |
|---|---|
| `op_api/` | aclnn API entry point (`aclnn_mega_moe.cpp`) |
| `op_host/` | Host-side op definition, infershape, tiling host code |
| `op_host/op_tiling/` | Tiling host logic; split into `arch22/` and `arch35/` |
| `op_kernel/arch22/` | Kernel code for Ascend910B / Ascend910_93 (A2/A3) |
| `op_kernel/arch35/` | Kernel code for Ascend 950PR/950DT |
| `op_graph/` | Graph plugin (currently minimal) |
| `tests/ut/` | Google Test based UTs (`op_api/`, `op_host/`) |
| `examples/` | aclnn example code (`test_aclnn_mega_moe.cpp`) |
| `torch_extension/` | PyTorch binding (`mega_moe.py`, `csrc/mega_moe.cpp`) |

## Architecture mapping

| SoC version | Arch dir | Products |
|---|---|---|
| `ascend910b` | `arch22` | Atlas A2 training/inference |
| `ascend910_93` | `arch22` | Atlas A3 training/inference |
| `ascend950` | `arch35` | Ascend 950PR / 950DT |

## Key conventions

- The op is registered as `aclnn_inner` type (not standard `aclnn`). The op type name is `mega_moe`, ACL type `aclnnInner`.
- Kernel code uses `.cpp`, `.h`, and `.hpp` extensions. Tiling headers use `_tiling` suffix.
- arch22 and arch35 kernels are completely separate implementations — do not mix them.
- The Ascend 950 path uses `--cce-auto-sync=off`; other archs use `--cce-auto-sync=on` (set in `op_host/CMakeLists.txt`).
- MC2 operators depend on `mc2/common`, `mc2/3rd`, and MoE dispatch/combine helpers. These are declared via `mega_moe_depends` / `mega_moe_apt_depends` in `op_host/CMakeLists.txt`.
- The PyTorch extension registers as `torch.ops.cann_ops_transformer.npu_mega_moe` via `OpBuilder`.

## Quantization scenarios

The operator supports multiple dtype scenarios (A16W16, A8W8-INT, A8W4-INT, A8W8-FP, A8W4-FP, A4W4-FP). Each has different input requirements and kernel paths. The `dispatchQuantMode` attribute selects the mode:
- `0` = non-quantized (A16W16)
- `2` = INT8 quantized (A8W8-INT, A8W4-INT)
- `4` = MXFP quantized (A8W8-FP, A8W4-FP, A4W4-FP) — Ascend 950 only

## Prerequisites

- CANN Toolkit must be installed; `ASCEND_CANN_PACKAGE_PATH` or `ASCEND_HOME_PATH` must be set.
- `bisheng` compiler must be available (ships with CANN).
- Python deps: see `requirements.txt` at repo root.
