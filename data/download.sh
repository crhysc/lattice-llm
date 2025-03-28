#!/bin/bash

# Set default output directory or use the one passed as argument
OUTDIR="${1:-downloads}"

# Create the output directory if it doesn't exist
mkdir -p "$OUTDIR"

# Make sure urls.txt exists
if [ ! -f urls.txt ]; then
  echo "urls.txt not found!"
  exit 1
fi

# Loop through each URL in the file
while read -r url; do
  # Extract the filename from the query string
  filename=$(echo "$url" | sed -n 's/.*filename=\(.*\)/\1/p')
  
  # Full path for output file
  filepath="$OUTDIR/$filename"

  # Download the file
  echo "Downloading $filename to $OUTDIR..."
  wget -q --show-progress -O "$filepath" "$url"
done < urls.txt

echo "All downloads complete. Files saved to: $OUTDIR"


