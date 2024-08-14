#!/usr/bin/env python3

import argparse
import logging
import subprocess
import sys
import tempfile
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
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "root", type=Path, metavar="ROOT", help="root directory of the benchmark suite"
    )
    parser.add_argument("-n", help="number of times to run each benchmark", default=5)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        metavar="OUTPUT",
        default=Path.cwd() / "results.csv",
        help="output file to write the results to",
    )
    benchmark_group = parser.add_mutually_exclusive_group()
    benchmark_group.add_argument(
        "--ppcg", type=Path, metavar="PPCG_ROOT", help="run benchmarks with PPCG"
    )
    benchmark_group.add_argument(
        "--pluto", type=Path, metavar="PLUTO_ROOT", help="run benchmarks with PLUTO"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="enable verbose output"
    )
    args = parser.parse_args()

    if args.verbose:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO
    logging.basicConfig(level=log_level, format="[%(levelname)s] %(message)s")

    args.output.touch(exist_ok=True)
    # Since we will be appending to the output file, we need to clear it first.
    args.output.write_text("")
    for benchmark in BENCHMARKS:
        tmp = tempfile.mktemp()
        if args.ppcg is not None:
            # First, run PPCG to generate the CUDA code.
            for bench in (args.root / benchmark).iterdir():
                source = bench / f"{bench.name}.c"
                if not source.exists():
                    logging.warning(f"skipping {source} as it does not exist")
                    continue
                compiler_opts = []
                if (bench / "compiler.opts").is_file():
                    compiler_opts = (
                        (bench / "compiler.opts").read_text().strip().split()
                    )
                cmd = [
                    f"{args.root}/scripts/ppcg-compile.py",
                    args.ppcg,
                    args.root,
                    *compiler_opts,
                    source,
                ]
                if args.verbose:
                    cmd.append("-v")
                # PPCG may be stuck in same cases, so we set a timeout.
                try:
                    ret = subprocess.run(cmd, timeout=60)
                except subprocess.TimeoutExpired:
                    logging.error(f"ppcg timed out for {source}")
                    continue
                if ret.returncode != 0:
                    logging.error(f"ppcg failed to compile {source}")
            # Then, run the generated CUDA code.
            ret = subprocess.run(
                [
                    f"{args.root}/scripts/run-benchmark.py",
                    benchmark,
                    "-n",
                    str(args.n),
                    "-o",
                    tmp,
                    "--sort",
                    "--dir",
                    args.root,
                    "--cuda",
                    "--cuda-compiler",
                    "clang++-18",
                    "-v" if args.verbose else "--quiet",
                ]
            )
        elif args.pluto is not None:
            # First, run Pluto to generate the tiled code.
            for bench in (args.root / benchmark).iterdir():
                source = bench / f"{bench.name}.c"
                if not source.exists():
                    logging.warning(f"skipping {source} as it does not exist")
                    continue
                cmd = [
                    f"{args.root}/scripts/pluto-compile.py",
                    args.pluto,
                    args.root,
                    source,
                    "-Xpluto=--noparallel",
                    # PLUTO outputs the tiling information by default, which we don't need when benchmarking.
                    "-Xpluto=-q",
                ]
                try:
                    ret = subprocess.run(cmd, timeout=60)
                except subprocess.TimeoutExpired:
                    logging.error(f"pluto timed out for {source}")
                    continue
                if ret.returncode != 0:
                    logging.error(f"pluto failed to compile {source}")
            # Then, run the generated tiled code.
            ret = subprocess.run(
                [
                    f"{args.root}/scripts/run-benchmark.py",
                    benchmark,
                    "-n",
                    str(args.n),
                    "-o",
                    tmp,
                    "--sort",
                    "--dir",
                    args.root,
                    "--compiler",
                    "clang-18",
                    "-v" if args.verbose else "--quiet",
                    "--suffixes",
                    ".pluto.c",
                ]
            )
        else:
            ret = subprocess.run(
                [
                    f"{args.root}/scripts/run-benchmark.py",
                    benchmark,
                    "-n",
                    str(args.n),
                    "-o",
                    tmp,
                    "--sort",
                    "--dir",
                    args.root,
                    "--compiler",
                    "clang-18",
                    "-v" if args.verbose else "--quiet",
                ]
            )
        if ret.returncode == 0:
            logging.info(
                f"successfully ran {benchmark}; appending results to {args.output}"
            )
            with args.output.open("a") as f:
                f.write(Path(tmp).read_text())
        Path(tmp).unlink(missing_ok=True)
