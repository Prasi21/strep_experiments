#!/bin/bash
echo "downloading datasets"
mkdir -p ./datasets/raw_data/no_pharyngitis
mkdir -p ./datasets/raw_data/pharyngitis

wget -O ./datasets/full_pharyngitis_dataset.zip https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/8ynyhnj2kz-2.zip
unzip ./datasets/full_pharyngitis_dataset.zip -d ./datasets/
unzip datasets/no\ pharyngitis\ v2.zip -d datasets/raw_data/no_pharyngitis
unzip datasets/pharyngitis\ v2.zip -d datasets/raw_data/pharyngitis


