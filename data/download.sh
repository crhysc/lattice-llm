#!/bin/bash

# Set default output directory or use the one passed as argument
OUTDIR="${1:-./raw}"

# Create the output directory if it doesn't exist
mkdir -p "$OUTDIR"

# Make sure urls.txt exists
if [ ! -f urls.txt ]; then
  echo "urls.txt not found!"
  exit 1
fi

# Total number of URLs
total_urls=$(wc -l < urls.txt)
count=0

# Loop through each URL in the file
while read -r url; do
  # Extract the filename from the query string
  filename=$(echo "$url" | sed -n 's/.*filename=\(.*\)/\1/p')
  
  # Full path for output file
  filepath="$OUTDIR/$filename"

  # Download the file
  echo "[$count/$total_urls] Downloading $filename to $OUTDIR..."
  wget -O "$filepath" "$url"

  # Increment count
  count=$((count + 1))

done < urls.txt

echo "All downloads complete. Files saved to: $OUTDIR"


