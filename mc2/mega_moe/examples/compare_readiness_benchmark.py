#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch


def pct(delta, base):
    return 100.0 * delta / base


def main():
    p = argparse.ArgumentParser()
    p.add_argument("baseline")
    p.add_argument("candidate")
    p.add_argument("--atol", type=float, default=0.0)
    p.add_argument("--rtol", type=float, default=0.0)
    args = p.parse_args()

    base_dir = Path(args.baseline)
    cand_dir = Path(args.candidate)
    base = json.loads((base_dir / "summary.json").read_text())
    cand = json.loads((cand_dir / "summary.json").read_text())

    for key in (
        "world_size",
        "tokens_per_rank",
        "hidden",
        "intermediate",
        "topk",
        "num_experts",
        "fp8",
        "hot_expert_m",
    ):
        if base[key] != cand[key]:
            raise RuntimeError(f"benchmark mismatch for {key}: {base[key]} != {cand[key]}")

    max_abs = 0.0
    all_close = True
    for rank in range(base["world_size"]):
        b = torch.load(base_dir / f"rank{rank}.pt", map_location="cpu")
        c = torch.load(cand_dir / f"rank{rank}.pt", map_location="cpu")
        diff = (b["y"].float() - c["y"].float()).abs()
        if diff.numel():
            max_abs = max(max_abs, float(diff.max().item()))
        if not torch.allclose(
            b["y"].float(), c["y"].float(), atol=args.atol, rtol=args.rtol
        ):
            all_close = False
        if not torch.equal(b["expert_token_nums"], c["expert_token_nums"]):
            raise RuntimeError(f"rank {rank}: expert_token_nums differ")

    b50 = base["latency"]["p50_us"]
    c50 = cand["latency"]["p50_us"]
    b90 = base["latency"]["p90_us"]
    c90 = cand["latency"]["p90_us"]

    result = {
        "outputs_close": all_close,
        "max_abs_error": max_abs,
        "baseline_p50_us": b50,
        "candidate_p50_us": c50,
        "p50_delta_us": c50 - b50,
        "p50_delta_pct": pct(c50 - b50, b50),
        "baseline_p90_us": b90,
        "candidate_p90_us": c90,
        "p90_delta_us": c90 - b90,
        "p90_delta_pct": pct(c90 - b90, b90),
    }
    print(json.dumps(result, indent=2))
    if not all_close:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
