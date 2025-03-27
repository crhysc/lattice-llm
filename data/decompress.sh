#!/bin/bash

for archive in file*; do
	tar -xvjf "$archive"
done
