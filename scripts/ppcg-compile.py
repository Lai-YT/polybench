#!/usr/bin/env python3

import argparse
import logging
import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        sys.argv[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="NOTE: CUDA files are generated in the same directory as FILE.",
    )
    parser.add_argument("ppcg_root", metavar="PPCG_ROOT", type=Path)
    parser.add_argument("polybench_root", metavar="POLYBENCH_ROOT", type=Path)
    parser.add_argument("-D", metavar="NAME[=VALUE]", action="append", default=[])
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("file", metavar="FILE", type=Path, help="the C file to compile")
    parser.add_argument(
        "-Xppcg",
        metavar="OPT",
        action="append",
        default=[],
        help="options to pass to PPCG (note: use -Xppcg=OPT to avoid argument parsing issues)",
    )
    args = parser.parse_args()

    if args.verbose:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO
    logging.basicConfig(level=log_level, format="[%(levelname)s] %(message)s")

    # Absolute paths to allow changing the working directory.
    args.ppcg_root = args.ppcg_root.resolve()
    args.polybench_root = args.polybench_root.resolve()
    args.file = args.file.resolve()

    command = [
        f"{args.ppcg_root}/ppcg",
        *list(map(lambda x: f"-D{x}", args.D)),
        "-I",
        f"{args.polybench_root}/utilities",
        str(args.file),
    ]
    command.extend(args.Xppcg)
    logging.debug(" ".join(command))
    ret = subprocess.run(
        command,
        # CUDA files are generated in the same directory as the input file.
        cwd=args.file.parent,
    )
    sys.exit(ret.returncode)
