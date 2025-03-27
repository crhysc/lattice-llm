#!/bin/bash

OUTPUT_DIR="/lattice-llm/data/raw/compressed"

while IFS= read -r url; do
    wget -P "$OUTPUT_DIR" "$url"
done < wget.txt

