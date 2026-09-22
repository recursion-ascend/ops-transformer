#!/usr/bin/env python3
"""Set explicit build-time ablation switches; rebuild host AND kernels afterwards."""
import argparse
import pathlib
import re

SWITCHES = (
    "MEGAMOE_LOAD_AWARE", "MEGAMOE_READY_AWARE", "MEGAMOE_GMM1_SWIZZLE",
    "MEGAMOE_CREDIT_AWARE", "MEGAMOE_AIV1_ARBITRATION",
)
PRESETS = {
    "baseline": (0, 0, 0, 0, 0),
    "load": (1, 0, 0, 0, 0),
    "ready": (0, 1, 0, 0, 0),
    "ready-swizzle": (0, 1, 1, 0, 0),
    "load-ready": (1, 1, 1, 0, 0),
    "credit": (0, 0, 0, 1, 0),
    "arbitration": (0, 0, 0, 0, 1),
    "full": (1, 1, 1, 1, 1),
}

def configure(text, preset):
    values = {"MEGAMOE_ONLINE_SCHEDULER": 0, **dict(zip(SWITCHES, PRESETS[preset]))}
    for name, value in values.items():
        text, count = re.subn(rf"^#define {name} [^\n]+$", f"#define {name} {value}", text,
                             flags=re.MULTILINE)
        if count != 1:
            raise ValueError(f"Expected exactly one definition of {name}, found {count}")
    return text

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=PRESETS, required=True)
    parser.add_argument("--check", action="store_true", help="check only; do not change the header")
    args = parser.parse_args()
    path = pathlib.Path(__file__).resolve().parents[1] / "op_kernel/arch35/common/mega_moe_online_policy.h"
    original = path.read_text()
    updated = configure(original, args.preset)
    if args.check:
        if updated != original:
            raise SystemExit(f"Header does not match {args.preset}")
        print(f"Configuration matches {args.preset}")
    else:
        path.write_text(updated)
        print(f"Configured {args.preset}: {path}\nRebuild host and device; existing binaries are unchanged.")

if __name__ == "__main__":
    main()
