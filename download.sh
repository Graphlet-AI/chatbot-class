#!/usr/bin/env bash

# Download the network motif papers to RAG from the source
curl -o data/Network_Motif_Papers.tar.gz http://rjurneyopen.s3.amazonaws.com/Network_Motif_Papers.tar.gz

# Extract the papers
tar -xvzf data/Network_Motif_Papers.tar.gz -C data/
