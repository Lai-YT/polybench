#!/usr/bin/env python3

import argparse
import logging
import subprocess
import sys
import tempfile
from pathlib import Path

POLYBENCH_DEFINES = ["-DPOLYBENCH_TIME", "-DDATA_TYPE=double"]


class Runner:
    def __init__(self, bench_cat: str, bench_root: Path, times: int):
        self._bench_cat = bench_cat
        self._bench_root = bench_root
        self._times = times

    def run(self) -> dict[str, list[float]]:
        logging.info(f"Benchmark category: {self._bench_cat}")

        bench_dir = self._bench_root / self._bench_cat
        if not bench_dir.exists():
            logging.error(f"directory {bench_dir} does not exist")
            return {}

        res: dict[str, list[float]] = {}
        for bench in bench_dir.iterdir():
            if sources := self._look_for_sources(bench):
                logging.info(f"running benchmark {bench.name}")
                output = tempfile.mktemp()
                if self._compile(bench, sources, output) != 0:
                    continue
                res[bench.name] = self._execute(output)
                if not res[bench.name]:
                    # remove the benchmark from the results if it failed to execute
                    del res[bench.name]
                Path(output).unlink()
            else:
                logging.warning(f"no sources; skipping {bench.name}")
        return res

    # NOTE: Rely on args.
    def _look_for_sources(self, bench: Path) -> list[Path]:
        logging.info(f"looking for files with suffixes {args.suffixes}")
        sources = set(filter(lambda f: f.is_file(), bench.glob(f"*{args.suffixes}")))
        excessive_suffix = set(bench.glob(f"*.*{args.suffixes}"))
        if excessive_suffix:
            logging.debug(
                f"files excluded due to excessive suffix: {list(map(str, excessive_suffix))}"
            )
        return list(sources - excessive_suffix)

    def _compile(self, bench: Path, sources: list[Path], output: str) -> int:
        """
        Returns:
            The return code of the compilation process.
        """
        compile_command = [
            *self._compiler_command(),
            "-O3",
            *POLYBENCH_DEFINES,
            "-lm",
            "-I",
            f"{self._bench_root}/utilities",
            f"{self._bench_root}/utilities/instrument.c",
            *list(map(str, sources)),
            "-o",
            output,
        ]
        if (bench / "compiler.opts").is_file():
            compile_command.extend(
                (bench / "compiler.opts").read_text().strip().split()
            )
        logging.debug(f"{' '.join(compile_command)}")
        ret = subprocess.run(compile_command)
        if ret.returncode != 0:
            logging.error(f"failed to compile {list(map(str, sources))}")
        return ret.returncode

    # NOTE: Rely on args.
    def _compiler_command(self) -> list[str]:
        if args.cuda:
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
        return [args.compiler]

    def _execute(self, executable: str) -> list[float]:
        res = []
        for _ in range(self._times):
            ret = subprocess.run([executable], capture_output=True, text=True)
            if ret.returncode != 0:
                logging.error(f"failed to execute {executable}")
                return []
            # NOTE: Outputting to stderr doesn't effect the result.
            res.append(float(ret.stdout))
            logging.info(f"{res[-1]}")
        return res


def write_results(output: Path, res: dict[str, list[float]]) -> None:
    logging.info(f"writing results to {output}")
    output.touch()
    with output.open("w") as f:
        for bench, times in res.items():
            f.write(f"{bench},{','.join(map(str, times))}\n")


BENCHMARKS = [
    "linear-algebra/kernels",
    "linear-algebra/solvers",
    "datamining",
    "stencils",
    "image-processing",
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        sys.argv[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Run a Polybench benchmark and record the results in a CSV file.",
    )
    parser.add_argument(
        "-n", type=int, default=5, help="number of times to run each benchmark"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path.cwd() / "results.csv",
        help="output file to write the results to",
    )
    parser.add_argument("-s", "--sort", action="store_true", help="sort the N results")
    parser.add_argument(
        "-d",
        "--dir",
        type=Path,
        default=Path.cwd(),
        help="the directory to the polybench folder",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="suppress output; overrides -v"
    )

    def check_compiler(compiler: str) -> str:
        if compiler.startswith("clang") or compiler.startswith("gcc"):
            return compiler
        raise argparse.ArgumentTypeError("compiler must be clang* or gcc*")

    parser.add_argument(
        "--compiler", default="clang-18", help="clang* or gcc*", type=check_compiler
    )
    parser.add_argument(
        "benchmark",
        metavar="BENCHMARK_CATEGORY",
        type=str,
        choices=BENCHMARKS,
        help=str(BENCHMARKS),
    )

    DEFAULT_C_SUFFIX = ".c"
    DEFAULT_CUDA_SUFFIX = ".cu"
    parser.add_argument(
        "--suffixes",
        type=str,
        help=f'suffixes of the source files, e.g., ".c" and ".pluto.c"; if not provided, {DEFAULT_C_SUFFIX} for C files and {DEFAULT_CUDA_SUFFIX} for CUDA files',
    )

    cuda_group = parser.add_argument_group("CUDA benchmarks")
    cuda_group.add_argument(
        "--cuda", action="store_true", help="look for CUDA files instead of C files"
    )

    def check_cuda_compiler(cuda_compiler: str) -> str:
        if cuda_compiler.startswith("clang++") or cuda_compiler.startswith("nvcc"):
            return cuda_compiler
        raise argparse.ArgumentTypeError("cuda-compiler must be clang++* or nvcc*")

    cuda_group.add_argument(
        "--cuda-compiler",
        type=str,
        default="clang++-18",
        help="clang++* or nvcc*; used if --cuda",
    )
    cuda_group.add_argument(
        "--cuda-gpu-arch",
        type=str,
        default="sm_86",
        help="the architecture of the GPU; used if --cuda",
    )

    args = parser.parse_args()

    if args.quiet:
        log_level = logging.ERROR
    elif args.verbose:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO
    logging.basicConfig(level=log_level, format="[%(levelname)s] %(message)s")

    if args.suffixes is None:
        if args.cuda:
            args.suffixes = DEFAULT_CUDA_SUFFIX
        else:
            args.suffixes = DEFAULT_C_SUFFIX

    runner = Runner(args.benchmark, args.dir, args.n)
    res: dict[str, list[float]] = runner.run()
    if args.sort:
        logging.info("sorting results")
        for times in res.values():
            times.sort()

    write_results(args.output, res)
