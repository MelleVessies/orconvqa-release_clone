mkdir data
wget -e robots=off --no-parent -rnH -R "index.html*" --cut-dirs=2 -P ./data https://ciir.cs.umass.edu/downloads/ORConvQA/
gunzip data/*.gz