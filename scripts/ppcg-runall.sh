#!/bin/sh

if [ $# -ne 3 ]; then
  echo "Usage: $0 <ppcg-abs-root> <benchmark-category> <machine-acronym>"
  echo "ex:    $0 /home/lai-yt/Desktopo/ppcg linear-algebra/kernels nehalem-gcc"
  exit 1
fi

## Default value for the compilation line.
if [ -z "$COMPILER_COMMAND" ]; then
  COMPILER_COMMAND="clang++ -O3 -lcudart -ldl -rt -pthread -lm -L/usr/local/cuda/lib64"
fi

PPCG_ROOT="$1"
BENCHCAT="$2"
MACHINE="$3"
echo "Machine: $MACHINE"
echo "Benchmark category: $BENCHCAT"
echo "PPCG root: $PPCG_ROOT"
rootdir=$(pwd)
mkdir -p "$rootdir/data"
cd "$BENCHCAT" \
  && for i in $(ls); do
    if [ -d "$i" ] && [ -f "$i/$i.c" ]; then
      echo "Testing benchmark $i"
      DATA_FILE="$rootdir/data/$MACHINE-ppcg-$i.dat"
      rm -f "$DATA_FILE"
      if [ -f "$i/compiler.opts" ]; then
        read comp_opts < $i/compiler.opts
        COMPILER_F_COMMAND="$COMPILER_COMMAND $comp_opts"
      else
        COMPILER_F_COMMAND="$COMPILER_COMMAND"
      fi
      for j in $(find "$i" -name "*.c"); do
        echo "Testing $j"
        # FIXME: comp_opts may contains things other than defines.
        "$rootdir/scripts/ppcg-compile.sh" "$PPCG_ROOT" "$rootdir" "$comp_opts" "$j"
        if [ $? -ne 0 ]; then
          echo "Problem when compiling $j with PPCG"
        else
          CUDA_FILES=$(find "$i" -name "*.cu" | tr '\n' ' ')
          echo "CUDA files: $CUDA_FILES"
          "$rootdir/scripts/cuda-compile.sh" "$rootdir" "$COMPILER_F_COMMAND" "$CUDA_FILES" "transfo"
          if [ $? -ne 0 ]; then
            echo "Problem when compiling $CUDA_FILES"
          else
            val=$(./transfo)
            if [ $? -ne 0 ]; then
              echo "Problem when executing $j"
            else
              cnt=0
              res=""
              while [ $cnt -lt 5 ]; do
                val=$(./transfo)
                if [ $? -ne 0 ]; then
                  echo "Problem when executing $j"
                  res="-1"
                else
                  echo "execution time: $val"
                  res="$res $val"
                fi
                cnt=$((cnt + 1))
              done
              output=$(echo "$res" | sed -e "s/s//g")
              echo "$j $output" >> "$DATA_FILE"
            fi
            rm ./transfo
          fi
          fi
      done
    fi
  done
cd ..
