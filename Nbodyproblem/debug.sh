SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
OUTPUT_DIR="$SCRIPT_DIR/build"
OUTPUT_FILE="$OUTPUT_DIR/debug.out"

mkdir -p "$OUTPUT_DIR"

clang++ -fopenmp -g -O0 -DEBUG "$SCRIPT_DIR/main.cpp" -o "$OUTPUT_FILE"

"$OUTPUT_FILE"
