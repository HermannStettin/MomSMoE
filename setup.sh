#!/bin/bash

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

git clone https://github.com/laekov/fastmoe.git
cd fastmoe

python setup.py install

cd ..
mkdir data/
cd data/
wget --quiet --continue https://dax-cdn.cdn.appdomain.cloud/dax-wikitext-103/1.0.1/wikitext-103.tar.gz
tar -xzf wikitext-103.tar.gz
cd wikitext-103
mv wiki.train.tokens train.txt
mv wiki.valid.tokens valid.txt
mv wiki.test.tokens test.txt
cd ../..
ls data/wikitext-103


git clone --branch mars https://github.com/AverageMetaheuristicsEnjoyer/Multi-Head-MomentumSMoE.git
cd Multi-Head-MomentumSMoE
mkdir -p scripts/

pip install -q wandb ninja dm-tree tqdm torchmetrics
wandb login 5996ed2fae48728cc24696a43fa357670a110267