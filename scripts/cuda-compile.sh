#!/usr/bin/env sh

if [ $# -ne 4 ]; then
  echo "Usage: $0 <root dir> <compiler command> <input files> <output file>"
  echo ""
  echo "Arguments:"
  echo "  <root dir>         The root directory of this benchmark;"
  echo "                     since this script may be called from any directory,"
  echo "                     it needs to know where the benchmark is located."
  echo "  <compiler command> The command to use to compile the benchmark;"
  echo "                     the instrumented code will be compiled automatically and"
  echo "                     linked with the benchmark, as well as the libm library."
  echo "  <input files>      The benchmark CUDA files to compile; quoted and separated by spaces."
  echo "  <output file>      The name of the output executable to generate."
  exit 1
fi

ROOT_DIR="$1"
COMPILER_COMMAND="$2"
INPUT_FILES="$3"
OUTPUT_FILE="$4"

CUDA_COMPILE="$COMPILER_COMMAND -I $ROOT_DIR/utilities $ROOT_DIR/utilities/instrument.c $INPUT_FILES -o $OUTPUT_FILE -lm"

echo "$CUDA_COMPILE"
$CUDA_COMPILE
