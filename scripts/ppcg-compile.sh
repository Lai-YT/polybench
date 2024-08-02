#!/usr/bin/env sh

if [ $# -ne 4 ]; then
  echo "Usage: $0 <ppcg abs root> <root dir> <defines> <input file>"
  echo ""
  echo "Compile a C file with PPCG. The output CUDA files are under the same directory with the input file."
  echo ""
  echo "Arguments:"
  echo "  <ppcg abs root>          The absolute path to the PPCG root directory."
  echo "  <root dir>               The root directory of the benchmark suite."
  echo "  <defines>                The defines to be passed to PPCG."
  echo "                           A quoted string in the form of -D<define1> -D<define2> ...."
  echo "  <input file>             The input C file to be compiled."
  exit 1
fi

PPCG_ROOT=$1
ROOT_DIR=$2
DEFINES=$3
INPUT_FILE=$4

TIME_LIMIT=100

# Find the directory of the input file.
INPUT_DIR=$(dirname "$INPUT_FILE")
# Get the file name without the directory, but with suffix.
INPUT_NAME="${INPUT_FILE##*/}"
# Also record the current directory, so that we can return to it later.
CUR_DIR=$(pwd)

echo "Changing to $INPUT_DIR..."
cd "$INPUT_DIR" || exit 1

PPCG_COMPILE="$PPCG_ROOT/ppcg $DEFINES -I $ROOT_DIR/utilities $INPUT_NAME"
echo "Compiling $INPUT_NAME with PPCG...; time limit: ${TIME_LIMIT}s"
echo "$PPCG_COMPILE"
# Set time limit for PPCG, as it may be stuck in some cases.
# NOTE: word splitting is necessary to treat it as a command.
if ! timeout $TIME_LIMIT $PPCG_COMPILE; then
  echo "Error: PPCG failed to compile the input file." >&2
  exit 1
fi

echo "Restoring the current directory..."
cd "$CUR_DIR" || exit 1

