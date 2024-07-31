#!/usr/bin/env python3
import argparse
import logging
import sys
from pathlib import Path


def remove_consecutive_commas(data: str) -> str:
    while ",," in data:
        data = data.replace(",,", ",")
    return data


def main(dir: Path) -> None:
    logging.debug(f"Looking for .dat files in {dir}...")
    for file in dir.iterdir():
        if file.is_file() and file.suffix == ".dat":
            data = file.read_text()
            logging.debug(f"Converting {file} to .csv...")
            data = data.replace(" ", ",")
            if ",," in data:
                logging.debug("Removing consecutive commas...")
                data = remove_consecutive_commas(data)
            logging.debug(f"Writing to {file.with_suffix('.csv')}...")
            file.with_suffix(".csv").write_text(data)
        else:
            logging.debug(f"{file} skipped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        sys.argv[0],
        description='Converts the ".dat" space separated file to ".csv" comma separated file.',
    )
    parser.add_argument(
        "dir", type=Path, metavar="DIR", help='directory of the ".dat" files'
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="show verbose messages"
    )
    args = parser.parse_args()

    if args.verbose:
        logging_level = logging.DEBUG
    else:
        logging_level = logging.INFO
    logging.basicConfig(level=logging_level, format="[%(levelname)s] %(message)s")

    main(args.dir)
