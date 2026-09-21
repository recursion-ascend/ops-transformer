#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
#
# A/B benchmark for MegaMoE readiness-aware GMM1 scheduling on Ascend 950.
# Run the same command against a main build and a candidate build.

import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu
from cann_ops_transformer.ops import get_symm_buffer_for_mega_moe, mega_moe


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def percentile(values, q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    idx = int(round((len(ordered) - 1) * q))
    return float(ordered[idx])


def git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def build_hot_routing(
    num_tokens: int, topk: int, num_experts: int, world_size: int
) -> torch.Tensor:
    if num_experts % world_size != 0:
        raise ValueError("num_experts must be divisible by world_size")
    local_experts = num_experts // world_size
    hot = [rank * local_experts for rank in range(world_size)]
    if topk < len(hot):
        raise ValueError(
            f"topk={topk} must be >= world_size={world_size} so every rank gets one hot expert"
        )
    if topk > num_experts:
        raise ValueError("topk cannot exceed num_experts")

    non_hot = [expert for expert in range(num_experts) if expert not in set(hot)]
    rows = []
    for token_idx in range(num_tokens):
        row = list(hot)
        need = topk - len(row)
        if need:
            start = (token_idx * need) % len(non_hot)
            for j in range(need):
                row.append(non_hot[(start + j) % len(non_hot)])
        rows.append(row)
    return torch.tensor(rows, dtype=torch.int32)


def make_fp8_weight(shape, fp8_dtype, value: float):
    # Keep host peak memory bounded: one expert tensor is converted and moved at a time.
    host = torch.full(shape, value, dtype=torch.float32)
    return host.to(fp8_dtype).npu()


def make_e8m0_scale(shape):
    # 127 corresponds to exponent 0 / scale 1 in the simple test payload.
    host = torch.full(shape, 127, dtype=torch.uint8).view(torch.float8_e8m0fnu)
    return host.npu()


def make_weights(args, rank, fp8_dtype):
    local_experts = args.num_experts // args.world_size
    l1_weights = []
    l2_weights = []
    l1_scales = []
    l2_scales = []

    for local_expert in range(local_experts):
        # Use deterministic constants; scheduler performance does not depend on random weights.
        value = 0.125 + 0.001 * ((rank * local_experts + local_expert) % 7)
        l1_weights.append(
            make_fp8_weight((2 * args.intermediate, args.hidden), fp8_dtype, value)
        )
        l2_weights.append(
            make_fp8_weight((args.hidden, args.intermediate), fp8_dtype, value)
        )
        l1_scales.append(
            make_e8m0_scale(
                (2 * args.intermediate, ceil_div(args.hidden, 64), 2)
            )
        )
        l2_scales.append(
            make_e8m0_scale(
                (args.hidden, ceil_div(args.intermediate, 64), 2)
            )
        )

    return l1_weights, l2_weights, l1_scales, l2_scales


def make_inputs(args, rank):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed + rank)
    x = torch.randn(
        (args.tokens, args.hidden), generator=generator, dtype=torch.float32
    ).to(torch.bfloat16).npu()

    topk_ids = build_hot_routing(
        args.tokens, args.topk, args.num_experts, args.world_size
    ).npu()
    topk_weights = torch.full(
        (args.tokens, args.topk),
        1.0 / args.topk,
        dtype=torch.bfloat16,
    ).npu()
    return x, topk_ids, topk_weights


def summarize(latencies_us):
    return {
        "samples": len(latencies_us),
        "mean_us": statistics.fmean(latencies_us),
        "stdev_us": statistics.pstdev(latencies_us)
        if len(latencies_us) > 1
        else 0.0,
        "min_us": min(latencies_us),
        "p50_us": percentile(latencies_us, 0.50),
        "p90_us": percentile(latencies_us, 0.90),
        "p99_us": percentile(latencies_us, 0.99),
        "max_us": max(latencies_us),
    }


def worker(rank: int, args):
    torch_npu.npu.set_device(rank)
    torch_npu.npu.config.allow_internal_format = True

    dist.init_process_group(
        backend="hccl",
        rank=rank,
        world_size=args.world_size,
        init_method=f"tcp://{args.master_addr}:{args.master_port}",
    )
    ep_group = dist.new_group(backend="hccl", ranks=list(range(args.world_size)))
    # Force HCCL EP communicator initialization before allocating the symmetric buffer.
    _ = ep_group._get_backend(torch.device("npu")).get_hccl_comm_name(
        rank, init_comm=True
    )

    fp8_dtype = (
        torch.float8_e4m3fn if args.fp8 == "e4m3" else torch.float8_e5m2
    )
    x, topk_ids, topk_weights = make_inputs(args, rank)
    l1_weights, l2_weights, l1_scales, l2_scales = make_weights(
        args, rank, fp8_dtype
    )

    sym_buffer = get_symm_buffer_for_mega_moe(
        ep_group,
        num_experts=args.num_experts,
        num_max_tokens_per_rank=args.tokens,
        num_topk=args.topk,
        hidden=args.hidden,
        intermediate_hidden=args.intermediate,
        dispatch_quant_mode=4,
        dispatch_quant_out_dtype=fp8_dtype,
        combine_quant_mode=0,
        topk_weights_type=0,
    )

    def run_once():
        return mega_moe(
            x,
            topk_ids,
            topk_weights,
            l1_weights,
            l2_weights,
            sym_buffer,
            l1_weights_sf=l1_scales,
            l2_weights_sf=l2_scales,
        )

    last_y = None
    last_counts = None
    for _ in range(args.warmup):
        last_y, last_counts = run_once()
    torch.npu.synchronize()
    dist.barrier(group=ep_group)

    latencies_us = []
    for _ in range(args.iters):
        dist.barrier(group=ep_group)
        torch.npu.synchronize()
        start_ns = time.perf_counter_ns()
        last_y, last_counts = run_once()
        torch.npu.synchronize()
        elapsed_us = (time.perf_counter_ns() - start_ns) / 1000.0

        # Distributed step latency is the slowest rank.
        elapsed = torch.tensor([elapsed_us], dtype=torch.float32, device="npu")
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX, group=ep_group)
        if rank == 0:
            latencies_us.append(float(elapsed.cpu().item()))

    counts_cpu = last_counts.cpu()
    expected_hot_m = args.world_size * args.tokens
    actual_hot_m = int(counts_cpu[0].item())
    if actual_hot_m != expected_hot_m:
        raise RuntimeError(
            f"rank {rank}: hot local expert received {actual_hot_m} rows, "
            f"expected {expected_hot_m}"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "rank": rank,
            "y": last_y.cpu(),
            "expert_token_nums": counts_cpu,
        },
        output_dir / f"rank{rank}.pt",
    )

    if rank == 0:
        result = {
            "git_head": git_head(),
            "device_name": torch_npu.npu.get_device_name(),
            "world_size": args.world_size,
            "tokens_per_rank": args.tokens,
            "hidden": args.hidden,
            "intermediate": args.intermediate,
            "topk": args.topk,
            "num_experts": args.num_experts,
            "fp8": args.fp8,
            "hot_expert_m": expected_hot_m,
            "hot_expert_m_groups_256": ceil_div(expected_hot_m, 256),
            "warmup": args.warmup,
            "iterations": args.iters,
            "latency": summarize(latencies_us),
        }
        with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(json.dumps(result, indent=2), flush=True)

    dist.barrier(group=ep_group)
    sym_buffer.destroy()
    dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ascend 950 MegaMoE readiness-aware GMM1 A/B benchmark"
    )
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--tokens", type=int, default=384)
    parser.add_argument("--hidden", type=int, default=6144)
    parser.add_argument("--intermediate", type=int, default=2048)
    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument("--fp8", choices=["e4m3", "e5m2"], default="e4m3")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--master-addr", default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=50121)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.num_experts % args.world_size != 0:
        raise ValueError("num_experts must be divisible by world_size")
    if args.topk < args.world_size:
        raise ValueError("topk must be >= world_size for the hot-expert routing")
    mp.set_start_method("spawn", force=True)
    mp.spawn(worker, args=(args,), nprocs=args.world_size, join=True)
