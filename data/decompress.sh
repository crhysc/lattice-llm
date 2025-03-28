#!/bin/bash

INPUT_DIR="./raw/compressed"
OUTPUT_DIR="./raw/decompressed"

mkdir -p "$OUTPUT_DIR"

for file in "$INPUT_DIR"/*; do
    case "$file" in
        *.zip)
            unzip -d "$OUTPUT_DIR" "$file"
            ;;
        *.tar.gz | *.tgz)
            tar -xzf "$file" -C "$OUTPUT_DIR"
            ;;
        *.tar.bz2)
            tar -xjf "$file" -C "$OUTPUT_DIR"
            ;;
        *.gz)
            outfile="$OUTPUT_DIR/$(basename "${file%.gz}")"
            gunzip -c "$file" > "$outfile"
            ;;
        *)
            echo "Skipping unsupported file: $file"
            ;;
    esac
done

echo "Decompression complete."

