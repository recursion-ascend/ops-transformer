# MegaMoE readiness-aware GMM1 A/B benchmark

This benchmark is intended for Ascend 950PR/950DT and the arch35 MTE Wave path.

It creates deterministic "hot expert" routing so that one local expert on every
rank receives multiple 256-row GMM1 M-blocks. This is important: the usual B64
case often leaves every expert below 256 rows and therefore cannot exercise the
readiness-aware scheduling mechanism.

## Recommended cases

With two ranks, every token routes to one hot expert on each rank. Therefore the
hot expert receives:

- `--tokens 256` -> M=512 -> 2 M-blocks
- `--tokens 384` -> M=768 -> 3 M-blocks
- `--tokens 512` -> M=1024 -> 4 M-blocks

For the project target geometry use `H=6144`, `intermediate=2048`, A8W8 E4M3.

## Build

From the repository root, build the exact checkout being tested:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
bash build.sh --ophost --opapi --opkernel --ops=mega_moe --soc=ascend950 -j16
bash build.sh --torch_extension_only --ops=mega_moe
```

Install/deploy the generated operator package and Python extension using the
same procedure for both checkouts. Do not reuse a kernel package produced by
the other checkout.

Using two separate git worktrees is strongly recommended so build products do
not contaminate each other:

```bash
git worktree add ../ops-main main
git worktree add ../ops-ready experiment/mega-moe-readiness-aware-gmm1
```

## Run baseline and candidate

Run exactly the same command in the environment built from each checkout:

```bash
python mc2/mega_moe/examples/benchmark_readiness_aware.py \
  --world-size 2 \
  --tokens 384 \
  --hidden 6144 \
  --intermediate 2048 \
  --topk 2 \
  --num-experts 8 \
  --fp8 e4m3 \
  --warmup 10 \
  --iters 50 \
  --output-dir /tmp/mega_moe_main
```

Then run the candidate and write to another directory:

```bash
python mc2/mega_moe/examples/benchmark_readiness_aware.py \
  --world-size 2 \
  --tokens 384 \
  --hidden 6144 \
  --intermediate 2048 \
  --topk 2 \
  --num-experts 8 \
  --fp8 e4m3 \
  --warmup 10 \
  --iters 50 \
  --output-dir /tmp/mega_moe_ready
```

Compare correctness and latency:

```bash
python mc2/mega_moe/examples/compare_readiness_benchmark.py \
  /tmp/mega_moe_main /tmp/mega_moe_ready
```

The benchmark records the slowest-rank wall time for every iteration. Report
P50/P90, not the single best run.

## Four-rank target-style stress case

The routing generator always places one hot expert on every rank. For four
ranks, `--tokens 192` creates M=768 for each hot expert:

```bash
python mc2/mega_moe/examples/benchmark_readiness_aware.py \
  --world-size 4 --tokens 192 --topk 8 --num-experts 256 \
  --hidden 6144 --intermediate 2048 --fp8 e4m3 \
  --warmup 10 --iters 50 --output-dir /tmp/mega_moe_run
```

This case is memory-heavy because every rank owns 64 experts. Use the two-rank
case first for scheduler validation.

## Profiling

After correctness is established, profile one representative run with msProf.
For this MC2 fused operator the most useful views are the communication-compute
pipeline plus AI Core pipeline/cache metrics. Collect the same workload and
profiling options on baseline and candidate.

Do not infer a win only from end-to-end latency. Check whether the candidate
actually reduces GMM1 waiting/Cube idle without causing a larger weight/L2 or
occupancy penalty.
