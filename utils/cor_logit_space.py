# Inspired by "Deep Learning-Based Damage Mapping With InSAR Coherence Time Series" by Stephenson et al. (2021).
# Converts coherences in [0,1] to logit space in (-inf, +inf). 
# By default, logit transformation is applied to the square of the coherence values as they are mathematically closely equivalent to the "logarithm of the variance of the interferometric phase".
# Example Usage:
# python cor_logit_space.py -i /aria/zulun_sharing/TEST_KZ_TGNN_sharing_20250214/merged/cors_ALL -s 1 -eps 1e-4 &> cor_logit_space_$(date +%Y%m%d_%H%M%S).log
# python cor_logit_space.py -i /aria/tgnn_dpm5/usa_california_wildfires/P064/merged/cors -s 1 -eps 1e-4 &> cor_logit_space_$(date +%Y%m%d_%H%M%S).log


import os
import sys
import argparse
import glob
import re
import numpy as np 
# from utils.isce_tools import sizeFromXml, readCor


# Set up argparse
def cmdLineParse():
    parser = argparse.ArgumentParser(description='Converts the coherences in the domain of [0,1] to logit space (-inf, +inf)')
    parser.add_argument('-i', dest='in_dir', type=str, required=True, 
            help = 'Path to merged/cor directory containing the coherence files')
    parser.add_argument('-s', dest='save', type=int, default=0,
            help = 'Flag to save logit-transformed squared coherences (1: save, 0: do not save)'),
    parser.add_argument('-eps', dest='eps', type=float, default=1e-4,
            help = 'Small epsilon to clip squared coherence values and avoid division by zero or log(0) (default: 1e-4)')
    
    args = parser.parse_args()

    if len(sys.argv) <= 1:
        print('')
        parser.print_help()
        sys.exit(1)
    else:
        return args


def sizeFromXml(file):
    fileXml = file + '.xml'
    tree = ET.parse(fileXml)
    root = tree.getroot()
    width = int(root.find(".//property[@name='width']/value").text)
    length = int(root.find(".//property[@name='length']/value").text)
    return width, length   


def readCor(file,bbox=None):
    # TO-DO for CT: optimise reading binary file using bytes to avoid reading whole image into memory
    # bbox is in the form of [start_x, end_x, start_y, end_y]
    # Convention is x = length, y = width
    width, length = sizeFromXml(file)
    with open(file, 'rb') as f:
        data = np.fromfile(f, dtype=np.float32)
        data2 = data.reshape([length * 2, width])
        # Use the second band (odd rows) as the coherence (edge) feature
        cor = data2[1::2, :]
    
    # If bbox is provided, crop the coherence image to the specified bounding box
    if bbox is not None:
        cor = cor[bbox[0]:bbox[1]+1, bbox[2]:bbox[3]+1]

    # Optionally disable plotting for batch processing
    # plt.imshow(cor, cmap='gray', vmin=0, vmax=5)
    # plt.colorbar(); plt.title("Cor 2"); plt.show()

    # Check for the presence of pixels with NaN values in the coherence image 
    # If any pixels with NaN values are present, assign them a value of "0"
    nan_infill_value = 0
    has_nan_cor = np.isnan(cor).any()
    if has_nan_cor:
        print(f"WARNING: Pixels with NaN values have been detected in the coherence image and were converted to {nan_infill_value}.")
        cor = np.nan_to_num(cor, nan = nan_infill_value)

    return cor


# Coherence to logit space transformation
def cor_to_logit(cor, eps=1e-10):

    # Clip squared coherence values into (0, 1) to avoid 0 and 1
    cor_squared = np.clip(np.square(cor), eps, 1 - eps)

    # Report how many values were clipped
    num_clipped = np.sum((cor_squared <= eps) | (cor_squared >= 1 - eps))
    if num_clipped > 0:
        print(f"[WARNING] {num_clipped} coherence values were clipped to avoid numerical issues.")

    # Apply logit transformation on squared coherences
    logit_cor = np.log(cor_squared / (1 - cor_squared))

    return logit_cor


# Logit space to coherence transformation
def logit_to_cor(logit_cor):
    cor = np.sqrt(1 / (1 + np.exp(-logit_cor)))
    return cor


def saveCor(width, length, cor, cor_ori):
    # Write to new coherence file
    cor_new = cor_ori[:-4] + f'_logit.cor'
    cor_data = np.zeros((length * 2, width), dtype=np.float32)
    cor_data[1:length * 2:2, :] = cor
    cor_data.astype(np.float32).tofile(cor_new)

    # Create VRT and XML files
    vrt_ori = cor_ori + '.vrt'
    vrt_new = cor_new + '.vrt'
    with open(vrt_ori, 'r') as f:
        content = f.read()
    content = content.replace('lks.cor', 'lks_logit.cor')
    with open(vrt_new, 'w') as f:
        f.write(content)
    xml_ori = cor_ori + '.xml'
    xml_new = cor_new + '.xml'
    with open(xml_ori, 'r') as f:
        content = f.read()
    content = content.replace('lks.cor', 'lks_logit.cor')
    with open(xml_new, 'w') as f:
        f.write(content)


# Convert logit-transformed coherence images (GT, Pred) back to regular coherence space. 
# Compute the residuals in reglar coherence space.
def convert_images_to_reg_cor(gt_img_logit, pred_img_logit):
    gt_img_reg_cor = logit_to_cor(gt_img_logit)
    pred_img_reg_cor = logit_to_cor(pred_img_logit)
    residuals_reg_cor = gt_img_reg_cor - pred_img_reg_cor
    abs_residuals_reg_cor = np.abs(residuals_reg_cor)
    return gt_img_reg_cor, pred_img_reg_cor, residuals_reg_cor, abs_residuals_reg_cor


# Main script to convert coherences to logit space
if __name__ == '__main__':

    # Parse command line arguments
    inps = cmdLineParse()

    # Get list of coherence files
    # Note for KZ: Check the nomenclature of the raw coherence files after processing the multi-temporal stack. The Glob / RegEx filter may require changes.
    glob_str = inps.in_dir + '/cor_????????_????????/b01_16r4alks.cor'
    cor_files = sorted(glob.glob(glob_str))

    # Read the first coherence file to get dimensions (width and length)
    width, length = sizeFromXml(cor_files[0])
    
    # Convert coherences to logit space
    for cor_file in cor_files:
        print(f'Reading coherence file: {cor_file}')
        
        # Read the coherence data
        cor = readCor(cor_file, [0, length, 0, width])
    
        # Convert coherence to logit space
        logit_cor = cor_to_logit(cor)

        # Print statistics of the coherence
        print(f"\n"
              f"Statistics for original coherence [ MIN | 1st | 25th | 50th | 75th | 99th | MAX ]\n"
              f"{np.min(cor):.4f} | {np.percentile(cor, 1):.4f} | {np.percentile(cor, 25):.4f} | {np.median(cor):.4f} | {np.percentile(cor, 75):.4f} | {np.percentile(cor, 99):.4f} | {np.max(cor):.4f}\n"
              f"\n"
              f"Statistics for logit space transformed coherence [ MIN | 1st | 25th | 50th | 75th | 99th | MAX ]\n"
              f"{np.min(logit_cor):.4f} | {np.percentile(logit_cor, 1):.4f} | {np.percentile(logit_cor, 25):.4f} | {np.median(logit_cor):.4f} | {np.percentile(logit_cor, 75):.4f} | {np.percentile(logit_cor, 99):.4f} | {np.max(logit_cor):.4f}\n"
              f"\n")
        
        # Save the logit-transformed coherences if specified
        if inps.save == 1:
            logit_cor_file = cor_file[:-4] + '_logit.cor'
            saveCor(width, length, logit_cor, cor_file)
            print(f"Saved logit-transformed coherence to: {logit_cor_file}\n\n")