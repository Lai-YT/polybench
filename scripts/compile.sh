#!/bin/sh

if [ $# -ne 4 ]; then
    echo "Usage: compile.sh <root dir> <compiler command> <input file> <output file>";
    echo ""
    echo "Arguments:"
    echo "  <root dir>         The root directory of this benchmark;"
    echo "                     since this script may be called from any directory,"
    echo "                     it needs to know where the benchmark is located."
    echo "  <compiler command> The command to use to compile the benchmark;"
    echo "                     the instrumented code will be compiled automatically and"
    echo "                     linked with the benchmark, as well as the libm library."
    echo "  <input file>       The benchmark C file to compile."
    echo "  <output file>      The name of the output executable to generate."
    exit 1;
fi;

ROOT_DIR="$1";
COMPILER_COMMAND="$2";
INPUT_FILE="$3";
OUTPUT_FILE="$4";

$COMPILER_COMMAND  -lm -I $ROOT_DIR/utilities $ROOT_DIR/utilities/instrument.c $INPUT_FILE -o $OUTPUT_FILE

echo "$COMPILER_COMMAND  -lm -I $ROOT_DIR/utilities $ROOT_DIR/utilities/instrument.c $INPUT_FILE -o $OUTPUT_FILE"


exit 0;
