# Normalises a stack of amplitude files globally
# Example commands:
# python amp_normalise.py -i /aria/zulun_sharing/TEST_KZ_TGNN_sharing_20250214/merged/cors_ALL -s 1 &> amp_normalise_$(date +%Y%m%d_%H%M%S).log
# python amp_normalise.py -i /aria/zulun_sharing/TEST_KZ_TGNN_sharing_20250214/merged/cors_ALL -s 1 --start 20240101 --end 20250110 &> amp_normalise_$(date +%Y%m%d_%H%M%S).log

import sys
import argparse
import glob
import re
import numpy as np
from utils.isce_tools import sizeFromXml, readAmp


def DN_to_dB(DN):
    # Apply log10 only where DN > 0
    # DN = sqrt(I^2 + Q^2)
    # dB = 10 * log10(DN^2) = 20 * log10(DN)
    dB = np.zeros_like(DN)
    nonzero_mask = DN != 0
    dB[nonzero_mask] = 20 * np.log10(DN[nonzero_mask])
    return dB


def saveAmp(width, length, amp1, amp2, amp_ori):
    # Write to new amplitude file 
    amp_new = amp_ori[:-4] + '_norm.amp'
    amp_data = np.zeros((length, width * 2), dtype=np.float32)
    amp_data[:, 0:width * 2:2] = amp1
    amp_data[:, 1:width * 2:2] = amp2
    amp_data.astype(np.float32).tofile(amp_new)

    # Create VRT and XML files
    vrt_ori = amp_ori + '.vrt'
    vrt_new = amp_new + '.vrt'
    with open(vrt_ori, 'r') as f:
        content = f.read()
    content = content.replace('lks.amp', 'lks_norm.amp')
    with open(vrt_new, 'w') as f:
        f.write(content)
    xml_ori = amp_ori + '.xml'
    xml_new = amp_new + '.xml'
    with open(xml_ori, 'r') as f:
        content = f.read()
    content = content.replace('lks.amp', 'lks_norm.amp')
    with open(xml_new, 'w') as f:
        f.write(content)


def cmdLineParse():
    parser = argparse.ArgumentParser(description='Normalise a stack of amplitudes globally')
    parser.add_argument('-i', dest='in_dir', type=str, required=True, 
            help = 'Path to merged/cor directory containing the stack of amplitudes')
    parser.add_argument('-s', dest='save', type=int, default=0,
            help = 'Flag to save normalised amplitudes (1: save, 0: do not save)')
    parser.add_argument('--start', dest='start_date', type=str,
            help = 'Start date to compute global normalisation')
    parser.add_argument('--end', dest='end_date', type=str,
            help = 'End date to compute global normalisation')
    if len(sys.argv) <= 1:
        print('')
        parser.print_help()
        sys.exit(1)
    else:
        return parser.parse_args()


if __name__ == '__main__':
    # Parse command line arguments
    inps = cmdLineParse()

    # Get list of amplitude files
    glob_str = inps.in_dir + '/cor_????????_????????/b01_16r4alks.amp'
    amp_files = sorted(glob.glob(glob_str))

    # Get dates of each amplitude file
    amp_list = []
    date_format = re.compile(r'cor_(\d{8})_(\d{8})')
    for amp_file in amp_files:
        match = date_format.search(amp_file)
        if match:
             date1, date2 = match.groups()
             amp_list.append({
                 'file': amp_file,
                 'date1': date1,
                 'date2': date2})
    
    # Get start and end dates if not specified in inputs
    if inps.start_date is None or inps.end_date is None:
        inps.start_date = amp_list[0]['date1']
        inps.end_date = amp_list[-1]['date2']
        print('No start and end dates provided, using first and last dates from the list')
    
    # Filter amplitude files based on start and end dates
    amp_list_subset = [r for r in amp_list if r['date1'] >= inps.start_date and r['date2'] <= inps.end_date]
    print(f'Start date: {inps.start_date}')
    print(f'End date: {inps.end_date}')
    print(f'Number of amplitude files found: {len(amp_list)}')
    print(f'Number of amplitude files within start-end period: {len(amp_list_subset)}\n')
    
    # Get size of the first amplitude file
    width, length = sizeFromXml(amp_list_subset[0]['file'])
    
    # Keep track of which dates have been read/stored
    stored_dates = set()
    
    # Dictionary to store read amplitudes by date
    amp_data = {}
    
    # Loop through each amplitude file
    for amp in amp_list_subset:
        
        # Initialise
        print(f'Reading amplitude file: {amp["file"]}')
        date1 = amp['date1']
        date2 = amp['date2']

        # Skip if both dates have already been read/stored 
        if date1 in stored_dates and date2 in stored_dates:
            continue

        # Read the amplitude file
        amp1, amp2 = readAmp(amp['file'],[0, length, 0, width]) 
        
        # Convert amplitude from DN to dB scale
        # Note that original amplitudes are in DN = sqrt(I^2 + Q^2) (e.g. from slcstk2cor.py)
        amp1_dB = DN_to_dB(amp1)
        amp2_dB = DN_to_dB(amp2)
        
        # Store amplitudes as needed
        if date1 not in stored_dates:
            amp_data[date1] = amp1_dB
            stored_dates.add(date1)
            print(f'Adding {date1} to data')
        if date2 not in stored_dates:
            amp_data[date2] = amp2_dB
            stored_dates.add(date2)
            print(f'Adding {date2} to data')
    
    print(f'\nNumber of unique dates read: {len(stored_dates)}\n')
    
    # Stack all amplitudes into a 3D array (date, length, width)
    amp_all = np.stack(list(amp_data.values()))
    
    # Compute global min and max of amplitudes in dB scale
    # Ignore 0s (0 = no data)
    # Use 1st and 99th percentiles instead of actual min and max to avoid outliers
    nonzero_mask = amp_all != 0
    nonzero_values = amp_all[nonzero_mask]
    global_min = np.percentile(nonzero_values,1)
    global_max = np.percentile(nonzero_values,99)
    print(f'Global min in dB scale: {np.min(nonzero_values)}')
    print(f'Global max in dB scale: {np.max(nonzero_values)}')
    print(f'Global 1st percentile in dB scale: {global_min}')
    print(f'Global 99th percentile in dB scale: {global_max}\n')
    del amp_all, nonzero_mask, nonzero_values
    
    if inps.save == 1:
        
        # Loop through each amplitude file  
        print('Saving normalised amplitudes...')  
        for amp in amp_list_subset:
            
            # Initialise
            date1 = amp['date1']
            date2 = amp['date2']

            # Read the amplitude file
            amp1, amp2 = readAmp(amp['file'],[0, length, 0, width]) 
            
            # Convert amplitude from DN to dB scale
            # Note that original amplitudes are in DN = sqrt(I^2 + Q^2) (e.g. from slcstk2cor.py)
            amp1_dB = DN_to_dB(amp1)
            amp2_dB = DN_to_dB(amp2)
            
            # Initialise output arrays
            amp1_dB_norm = np.zeros_like(amp1_dB)
            amp2_dB_norm = np.zeros_like(amp2_dB)
            
            # Normalise/scale the amplitudes from 0 to 1 using the global min and max
            # Ignore 0s (0 = no data)
            mask1 = amp1_dB != 0
            mask2 = amp2_dB != 0
            amp1_dB_norm[mask1] = (amp1_dB[mask1] - global_min) / (global_max - global_min)
            amp2_dB_norm[mask2] = (amp2_dB[mask2] - global_min) / (global_max - global_min)
            
            # Clip values to ensure outliers stay in the [0, 1] range
            amp1_dB_norm = np.clip(amp1_dB_norm, 0, 1)
            amp2_dB_norm = np.clip(amp2_dB_norm, 0, 1)
            
            # Save normalised amplitudes
            saveAmp(width, length, amp1_dB_norm, amp2_dB_norm, amp['file'])
            print(f'Created normalised amplitude file: {amp["file"][:-4]+"_norm.amp"}')
