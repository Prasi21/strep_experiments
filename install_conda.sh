#!/bin/bash

# Download the Anaconda installer script
wget https://repo.anaconda.com/archive/Anaconda3-latest-Linux-x86_64.sh -O anaconda.sh

# Run the installer script
bash anaconda.sh -b -p $HOME/anaconda

# Activate the installation
source $HOME/anaconda/etc/profile.d/conda.sh

# Clean up the installer script
rm anaconda.sh

# Verify the installation
conda --version



wget https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh

https://www.hostinger.com/tutorials/how-to-install-anaconda-on-ubuntu/