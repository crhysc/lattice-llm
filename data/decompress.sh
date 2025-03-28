#!/bin/bash

INPUT_DIR="./raw/compressed"
OUTPUT_DIR="./raw/decompressed"

mkdir -p "$OUTPUT_DIR"

# Get total number of files to decompress
total_files=$(find "$INPUT_DIR" -type f | wc -l)
count=0

for file in "$INPUT_DIR"/*; do
    echo "[$count/$total_files] Decompressing $(basename "$file")..."

    case "$file" in
        *.zip)
            unzip -q -d "$OUTPUT_DIR" "$file"
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

    count=$((count + 1))
done

echo "Decompression complete. Output saved to: $OUTPUT_DIR"


