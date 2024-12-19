#!/bin/bash
# Install dependencies
pip install -r requirements.txt

# Download NLTK resources
python -m nltk.downloader wordnet
python -m nltk.downloader omw-1.4
