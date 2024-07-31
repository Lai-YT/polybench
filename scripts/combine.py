#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path


def create_output_files(dir: Path, type: str, machine_name: str) -> None:
    output_file = dir / f"{machine_name}.csv"
    output_file.touch(exist_ok=True)
    # Clear the file if it already exists.
    output_file.write_text("")

    if any(dir.glob(f"{machine_name}-ppcg-*.csv")):
        ppcg_output_file = dir / f"{machine_name}-ppcg.csv"
        ppcg_output_file.touch(exist_ok=True)
        ppcg_output_file.write_text("")


def combine_csv(dir: Path, type: str, machine_name: str) -> None:
    create_output_files(dir, type, machine_name)

    # Append the lines from each input file to the output file.
    for file in dir.glob(f"{machine_name}-*.{type}"):

        if file.name.startswith(f"{machine_name}-ppcg"):
            output_file = dir / f"{machine_name}-ppcg.{type}"
        else:
            output_file = dir / f"{machine_name}.{type}"

        if file == output_file:
            continue

        lines = file.read_text()
        if lines[-1] != "\n":
            lines += "\n"
        with output_file.open("a") as f:
            f.write(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        sys.argv[0],
        # description='Given a machine name, e.g. "x86-64", combine all lines from all csv files named "x86-64-*.csv" into a single file named "x86-64.csv". If there exists PPCG output files named "x86-64-ppcg-*.csv", they will be combined into "x86-64-ppcg.csv" separately.',
        description="Give a type of file and a machine name, combine all lines from all files named 'MACHINE-*.TYPE' into a single file named 'MACHINE.TYPE'. If there exists PPCG output files named 'MACHINE-ppcg-*.TYPE', they will be combined into 'MACHINE-ppcg.TYPE' separately.",
    )
    parser.add_argument("dir", type=Path, metavar="DIR", help="directory of the files")
    parser.add_argument("-t", "--type", choices=["csv", "tsv"], default="csv")
    parser.add_argument("machine_name", metavar="MACHINE")
    args = parser.parse_args()

    combine_csv(args.dir, args.type, args.machine_name)
