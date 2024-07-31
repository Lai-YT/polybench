#!/usr/bin/env sh

if [ $# -ne 5 ]; then
  echo "Usage: $0 <ppcg abs root> <root dir> <extra compiler command> <input file> <output file>"
  echo ""
  echo "Compile a C file with PPCG, and then compile the generated CUDA file with clang++-18."
  echo ""
  echo "Arguments:"
  echo "  <ppcg abs root>          The absolute path to the PPCG root directory."
  echo "  <root dir>               The root directory of the benchmark suite."
  echo "  <extra compiler command> The extra compiler command to define the macro."
  echo "  <input file>             The input C file to be compiled."
  echo "  <output file>            The output exectuable name."
  exit 1
fi

PPCG_ROOT=$1
ROOT_DIR=$2
EXTRA_COMPILER_COMMAND=$3
INPUT_FILE=$4
OUTPUT_FILE=$5

rm -f ./*.cu ./*.hu

PPCG_COMPILE="$PPCG_ROOT/ppcg -I $ROOT_DIR/utilities $INPUT_FILE"

cuda_compile() {
  echo "clang++-18 -O3 --cuda-gpu-arch=\"sm_86\" -lcudart -ldl -lrt -pthread \"$EXTRA_COMPILER_COMMAND\" -I \"$ROOT_DIR\"/utilities -L/usr/local/cuda/lib64 -I ./isl/include -I ./pet/include \"$ROOT_DIR\"/utilities/instrument.c ./*.cu -o \"$OUTPUT_FILE\" -lm"

  # Host code and kernel code are generated in the same directory;
  # Compile them together with clang++-18.
  clang++-18 -O3 --cuda-gpu-arch="sm_86" -lcudart -ldl -lrt -pthread "$EXTRA_COMPILER_COMMAND" -I "$ROOT_DIR/utilities" -L/usr/local/cuda/lib64 -I ./isl/include -I ./pet/include "$ROOT_DIR/utilities/instrument.c" ./*.cu -o "$OUTPUT_FILE" -lm
  return $?
}

echo "$PPCG_COMPILE"
# Set time limit for PPCG, as it may be stuck in some cases.
# NOTE: word splitting is necessary to treat it as a command.
if ! timeout 100 $PPCG_COMPILE; then
  echo "Error: PPCG failed to compile the input file." >&2
  exit 1
fi

if ! cuda_compile; then
  echo "Error: clang++-18 failed to compile the generated CUDA file." >&2
  # Remove the generated cuda files.
  rm -f ./*.cu ./*.hu
  exit 1
fi

# Remove the generated cuda files.
rm -f ./*.cu ./*.hu
