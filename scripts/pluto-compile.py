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
    )
    parser.add_argument("pluto_root", metavar="PLUTO_ROOT", type=Path)
    parser.add_argument("polybench_root", metavar="POLYBENCH_ROOT", type=Path)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "--no-silent", action="store_true", help="do not pass --silent to PLUTO"
    )
    parser.add_argument("file", metavar="FILE", type=Path, help="the C file to compile")
    parser.add_argument(
        "-Xpluto",
        metavar="OPT",
        action="append",
        default=[],
        help="options to pass to PLUTO (note: use -Xpluto=OPT to avoid argument parsing issues)",
    )
    args = parser.parse_args()

    if args.verbose:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO

    logging.basicConfig(level=log_level, format="[%(levelname)s] %(message)s")

    # Absolute paths to allow changing the working directory.
    args.pluto_root = args.pluto_root.resolve()
    args.polybench_root = args.polybench_root.resolve()
    args.file = args.file.resolve()

    command = [
        f"{args.pluto_root}/polycc",
        str(args.file),
    ]
    if not args.no_silent:
        # The default setup of PLUTO already outputs some information that is generally too much.
        command.append("--silent")
    command.extend(args.Xpluto)
    logging.debug(" ".join(command))
    ret = subprocess.run(
        command,
        # Files are generated in the same directory as the input file.
        cwd=args.file.parent,
    )
    sys.exit(ret.returncode)
