#!/bin/bash

INPUT_DIR="./raw/compressed"
OUTPUT_DIR="./raw/decompressed"

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
            gunzip -c "$file" > "$OUTPUT_DIR/$(basename "${file%.gz}")"
            ;;
        *)
            echo "Skipping unsupported file: $file"
            ;;
    esac
done

