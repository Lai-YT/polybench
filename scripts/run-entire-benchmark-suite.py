#!/usr/bin/env python3

import argparse
import os
import subprocess
import sys
from pathlib import Path

BENCHMARKS = [
    "linear-algebra/kernels",
    "linear-algebra/solvers",
    "datamining",
    "image-processing",
    "stencils",
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        sys.argv[0],
        description="A driver script to run the entire PolyBench/C benchmark suite.",
    )
    parser.add_argument(
        "root", type=Path, metavar="ROOT", help="root directory of the benchmark suite"
    )
    parser.add_argument("machine", metavar="MACHINE")
    parser.add_argument("-t", "--tool", choices=["ppcg"])
    parser.add_argument(
        "--ppcg-root",
        type=Path,
        metavar="PPCG_ROOT",
        help='root directory of PPCG; only needed if "--tool ppcg" is specified',
    )
    parser.add_argument(
        "-b",
        "--benchmark",
        metavar="BENCHMARK",
        choices=BENCHMARKS,
        help="run a specific benchmark",
    )
    args = parser.parse_args()

    benchmarks = BENCHMARKS
    if args.benchmark:
        benchmarks = [args.benchmark]

    # * POLYBENCH_TIME output the execution time of the benchmark.
    # TODO: The malloc version is broken.
    # * POLYBENCH_TEST_MALLOC use malloc instead of stack allocation, which is to align with PolyBench/GPU.
    DEFINES = "-DPOLYBENCH_TIME"
    for benchmark in benchmarks:
        if args.tool == "ppcg":
            CUDA_GPU_ARCH = "sm_86"
            ret = subprocess.run(
                [
                    "scripts/ppcg-runall.sh",
                    str(args.ppcg_root.resolve()),
                    benchmark,
                    args.machine,
                ],
                # The script favors to be run from the root directory of the benchmark suite.
                cwd=args.root,
                env={
                    "COMPILER_COMMAND": f"clang++-18 -O3 --cuda-gpu-arch={CUDA_GPU_ARCH} -lcudart -lrt -ldl -pthread -L /usr/local/cuda/lib64 {DEFINES}"
                }
                | os.environ,
            )
        else:
            ret = subprocess.run(
                ["scripts/runall.sh", benchmark, args.machine],
                # The script favors to be run from the root directory of the benchmark suite.
                cwd=args.root,
                env={"COMPILER_COMMAND": f"clang-18 -O3 {DEFINES}"} | os.environ,
            )
