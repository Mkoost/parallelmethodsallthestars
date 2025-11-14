#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
OUTPUT_DIR="$SCRIPT_DIR/build"
OUTPUT_FILE="$OUTPUT_DIR/program.out"

mkdir -p "$OUTPUT_DIR"

clang++ -fopenmp -O2 "$SCRIPT_DIR/newmain.cpp" -o "$OUTPUT_FILE"

"$OUTPUT_FILE"
