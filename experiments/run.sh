#!/bin/bash
cd /mshvol2/users/mohammad/optimization/burst-denoising-forked
cp /root/ssh_mount/id_rsa* /root/.ssh/
chmod 400 ~/.ssh/id_rsa
# apt update
# apt install -y software-properties-common
# add-apt-repository -y ppa:deadsnakes/ppa
# apt install -y python3.7 python3.7-distutils python3.7-venv exiftool nvidia-cuda-toolkit
# # source ./venv/bin/activate
# # pip install --upgrade pip
# python3.7 -m pip install --upgrade pip
# python3.7 -m pip install setuptools 
# python3.7 -m pip install imageio tensorflow-gpu==1.15.0 scikit-image==0.16.2 tqdm PyExifTool piq lpips plotly==5.6.0 pandas kaleido
# pip install wandb natsort matplotlib scipy numpy tensorflow-gpu==1.13.1 plotly pandas kaleido scikit-image opencv-python==4.2.0.32 tensorboardX jaxlib jax torch
conda env create -f req.yml
python -c """import imageio
imageio.plugins.freeimage.download()
"""
export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64:/usr/local/cuda-10.0/:$LD_LIBRARY_PATH
export PYTHONPATH=`pwd`:/mshvol2/users/mohammad/cvgutils/
echo command:
echo python $@
python $@