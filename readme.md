# ATTRIBUTE-GUIDED VISUAL REPROGRAMMING FOR VISION-LANGUAGE MODELS
This is the implementation of our paper submitted to ICLR2025.

## Installation
    conda create -n reprogram
    conda activate reprogram
    pip install -r requirement.txt

## Dataset Preparation
To implement the results, please follow Appendix A.2.1 to download the datasets and modify `DOWNSTREAM_PATH = ""` in `cfg.py`.

## Step 1: Generating DesAttr and DistAttr
Please first enter your API Key in `generate_attributes.py`, then run the code:
        
    python generated_attributes.py

## Step 2.1: Running Code for Baselines
    
    python experiments/fs_vp.py --dataset [dataset]
    python experiments/fs_ar.py --dataset [dataset]

## Step 2.2: Running Code for AttrVR

    python experiments/fs_attrvr.py --dataset [dataset]
