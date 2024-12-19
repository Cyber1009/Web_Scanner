#!/bin/bash
# Install necessary Python packages
pip install -r requirements.txt

# Download NLTK resources
python -m nltk.downloader wordnet
