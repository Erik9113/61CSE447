#!/bin/bash

mkdir -p data
cd data

echo "Downloading Wikipedia dump..."
wget https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2

echo "Done.";