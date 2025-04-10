#!/bin/bash

# Set default output directory or use the one passed as argument
INFILE="data/urls.txt"
OUTDIR="data/raw/"

# Create the output directory if it doesn't exist
mkdir -p "$OUTDIR"

# Make sure urls.txt exists
if [ ! -f $INFILE ]; then
  echo "urls.txt not found!"
  exit 1
fi

# Total number of URLs
total_urls=$(wc -l < $INFILE)
count=0

# Loop through each URL in the file
while read -r url; do
  # Extract the filename from the query string
  filename=$(echo "$url" | sed -n 's/.*filename=\(.*\)/\1/p')
  
  # Full path for output file
  filepath="$OUTDIR/$filename"

  # Download the file
  echo "[$count/$total_urls] Downloading $filename to $OUTDIR..."
  wget -q -O "$filepath" "$url"

  # Increment count
  count=$((count + 1))

done < $INFILE

echo "All downloads complete. Files saved to: $OUTDIR"


