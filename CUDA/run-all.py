#!/usr/bin/env python3

import argparse
import logging
import subprocess
import sys
import tempfile
from pathlib import Path

POLYBENCH_DEFINES = ["-DPOLYBENCH_TIME", "-DDATA_TYPE=double"]

# Mapping from benchmark names to the name of their CUDA files, as they may differ.
BENCHMARKS = {
    "2mm": "2mm",
    "3mm": "3mm",
    "atax": "atax",
    "bicg": "bicg",
    "corr": "correlation",
    "covar": "covariance",
    "fdtd-2d": "fdtd2d",
    "gemm": "gemm",
    "gesummv": "gesummv",
    "gramschm": "gramschmidt",
    "mvt": "mvt",
    "syr2k": "syr2k",
    "syrk": "syrk",
    ## The followings doesn't align with PolyBench/C 2.0.
    # "2dconV": "2DConvolution",
    # "3dconV": "3DConvolution",
}


class Runner:
    def __init__(self, bench_root: Path, times: int) -> None:
        self._bench_root = bench_root
        self._times = times

    def run(self) -> dict[str, list[float]]:
        res: dict[str, list[float]] = {}
        for benchmark in BENCHMARKS:
            logging.info(f"Benchmark: {benchmark}")

            # Directory name is upper case.
            bench_dir = self._bench_root / benchmark.upper()
            source = (bench_dir / BENCHMARKS[benchmark]).with_suffix(".cu")
            output = tempfile.mktemp()
            if self._compile(source, output) != 0:
                continue
            res[benchmark] = self._execute(output)
            if not res[benchmark]:
                # remove the benchmark from the results if it failed to execute
                del res[benchmark]
            Path(output).unlink(missing_ok=True)
        return res

    def _compile(self, source: Path, output: str) -> int:
        """
        Returns:
            The return code of the compilation process.
        """
        compile_command = [
            *self._compiler_command(),
            "-O3",
            *POLYBENCH_DEFINES,
            "-lm",
            str(source),
            "-o",
            output,
        ]
        logging.debug(f"{' '.join(compile_command)}")
        ret = subprocess.run(compile_command)
        if ret.returncode != 0:
            logging.error(f"failed to compile {source}")
        return ret.returncode

    # NOTE: Rely on args.
    def _compiler_command(self) -> list[str]:
        if args.cuda_compiler.startswith("nvcc"):
            return [args.cuda_compiler, f"-arch={args.cuda_gpu_arch}"]
        return [
            args.cuda_compiler,
            "-lcudart",
            "-lrt",
            "-ldl",
            "-L/usr/local/cuda/lib64",
            f"--cuda-gpu-arch={args.cuda_gpu_arch}",
        ]

    def _execute(self, executable: str) -> list[float]:
        res = []
        for _ in range(self._times):
            ret = subprocess.run([executable], capture_output=True, text=True)
            if ret.returncode != 0:
                logging.error(f"failed to execute {executable}")
                return []
            res.append(float(ret.stdout))
            logging.info(f"{res[-1]}")
        return res


def write_results(output: Path, res: dict[str, list[float]]) -> None:
    logging.info(f"writing results to {output}")
    output.touch()
    with output.open("w") as f:
        for bench, times in res.items():
            f.write(f"{bench},{','.join(map(str, times))}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        sys.argv[0],
        description="A driver script to run the entire PolyBench/GPU benchmark suite.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "root",
        type=Path,
        metavar="ROOT",
        help="root directory of the CUDA benchmark suite (note: not the entire PolyBench/GPU)",
    )
    parser.add_argument("-n", help="number of times to run each benchmark", default=5)
    parser.add_argument("-s", "--sort", action="store_true", help="sort the N results")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        metavar="OUTPUT",
        default=Path.cwd() / "results.csv",
        help="output file to write the results to",
    )

    def check_cuda_compiler(cuda_compiler: str) -> str:
        if cuda_compiler.startswith("clang++") or cuda_compiler.startswith("nvcc"):
            return cuda_compiler
        raise argparse.ArgumentTypeError("must be clang++* or nvcc*")

    parser.add_argument(
        "--cuda-compiler", type=str, default="clang++-18", help="clang++* or nvcc*"
    )
    parser.add_argument(
        "--cuda-gpu-arch", type=str, default="sm_86", help="the architecture of the GPU"
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

    runner = Runner(args.root, int(args.n))
    res = runner.run()

    if args.sort:
        logging.info("sorting results")
        for times in res.values():
            times.sort()

    write_results(args.output, res)
