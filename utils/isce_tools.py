import os
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt


def readAmp(file,bbox):
    # TO-DO for CT: optimise reading binary file using bytes to avoid reading whole image into memory
    # bbox is in the form of [start_x, end_x, start_y, end_y]
    # Convention is x = length, y = width
    width, length = sizeFromXml(file)
    with open(file, 'rb') as f:
        data = np.fromfile(f, dtype='<f4')
        data2 = data.reshape([length, width * 2])
        amp1 = data2[:, ::2]
        amp2 = data2[:, 1::2]
    amp1 = amp1[bbox[0]:bbox[1]+1, bbox[2]:bbox[3]+1]
    amp2 = amp2[bbox[0]:bbox[1]+1, bbox[2]:bbox[3]+1]

    # Optionally disable plotting for batch processing
    # plt.imshow(amp1, cmap='gray', vmin=0, vmax=5)
    # plt.colorbar(); plt.title("Amplitude 1"); plt.show()
    # plt.imshow(amp2, cmap='gray', vmin=0, vmax=5)
    # plt.colorbar(); plt.title("Amplitude 2"); plt.show()

    # Check for the presence of pixels with NaN values in the amplitude images
    # If any pixels with NaN values are present, assign them a value of "0"
    nan_infill_value = 0
    has_nan_amp1 = np.isnan(amp1).any()
    has_nan_amp2 = np.isnan(amp2).any()
    if has_nan_amp1:
        print(f"WARNING: Pixels with NaN values have been detected in the amplitude image and were converted to {nan_infill_value}.")
        amp1 = np.nan_to_num(amp1, nan = nan_infill_value)
    if has_nan_amp2:
        print(f"WARNING: Pixels with NaN values have been detected in the amplitude image and were converted to {nan_infill_value}.")
        amp2 = np.nan_to_num(amp2, nan = nan_infill_value)

    return amp1, amp2


def readCor(file,bbox):
    # TO-DO for CT: optimise reading binary file using bytes to avoid reading whole image into memory
    # bbox is in the form of [start_x, end_x, start_y, end_y]
    # Convention is x = length, y = width
    width, length = sizeFromXml(file)
    with open(file, 'rb') as f:
        data = np.fromfile(f, dtype=np.float32)
        data2 = data.reshape([length * 2, width])
        # Use the second band (odd rows) as the coherence (edge) feature
        cor = data2[1::2, :]
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


def sizeFromXml(file):
    fileXml = file + '.xml'
    tree = ET.parse(fileXml)
    root = tree.getroot()
    width = int(root.find(".//property[@name='width']/value").text)
    length = int(root.find(".//property[@name='length']/value").text)
    return width, length   


def computePatches(image_x, image_y, patch_length, patch_overlap):
    # Initialize the output patchList which will contain a list of patch bounding boxes in the form of [start_x, end_x, start_y, end_y]
    patchList = []
    # Compute step sizes
    step_x = patch_length - patch_overlap
    step_y = patch_length - patch_overlap
    # Define patches for normal area (row-major order)
    for start_y in range(0, image_y - patch_length + 1, step_y):  # Iterate over y first (rows)
        for start_x in range(0, image_x - patch_length + 1, step_x):  # Iterate over x inside y-loop (columns)
            patchList.append([start_x, start_x + patch_length - 1, start_y, start_y + patch_length - 1])
    # Handle the right edge (x-axis)
    if (image_x - patch_length) % step_x != 0:
        for start_y in range(0, image_y - patch_length + 1, step_y):
            patchList.append([image_x - patch_length, image_x - 1, start_y, start_y + patch_length - 1])
    # Handle the bottom edge (y-axis)
    if (image_y - patch_length) % step_y != 0:
        for start_x in range(0, image_x - patch_length + 1, step_x):
            patchList.append([start_x, start_x + patch_length - 1, image_y - patch_length, image_y - 1])
    # Handle the bottom-right corner
    if (image_x - patch_length) % step_x != 0 and (image_y - patch_length) % step_y != 0:
        patchList.append([image_x - patch_length, image_x - 1, image_y - patch_length, image_y - 1])
    
    print("Total image width (x-axis): ", image_x)
    print("Total image length (y-axis): ", image_y) 
    print("Number of patches: ", len(patchList)) 
    print("Patch bbox: ", patchList)
    return patchList


def prepareData(data_in_dir, data_out_dir, bbox, base_filename='b01_16r4alks'):

    # Create a directory "image_data" in the current folder to store outputs
    save_dir = os.path.join(data_out_dir, "image_data")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Dictionaries and lists to hold the node features and edge features
    date_to_feature = {}   # date (str) -> flattened amplitude vector
    edge_features_list = []  # list to store flattened edge features for each folder
    edge_node_pairs = []     # list to store corresponding (date1, date2) for each edge

    # Get a list of subdirectories with names like "cor_YYYYMMDD_YYYYMMDD"
    subdirs = [d for d in os.listdir(data_in_dir) if d.startswith("cor_") and os.path.isdir(os.path.join(data_in_dir, d))]
    subdirs = sorted(subdirs, key=lambda x: (int(x.split("_")[2]), int(x.split("_")[1])))  # Prioritise sorting by the second date, then sorting by first date

    # Read all image pairs within the patch    
    for folder in subdirs:
        parts = folder.split('_')
        if len(parts) < 3:
            print(f"Skipping folder {folder} due to unexpected name format.")
            continue
        date1, date2 = parts[1], parts[2]
        folder_path = os.path.join(data_in_dir, folder)
        amp_file = os.path.join(folder_path, base_filename + ".amp")
        cor_file = os.path.join(folder_path, base_filename + ".cor")

        # Check that both the .amp and .cor files (and their XML metadata) exist
        if not (os.path.exists(amp_file) and os.path.exists(amp_file + '.xml')):
            print(f"Skipping folder {folder}: amplitude file or its XML is missing.")
            continue
        if not (os.path.exists(cor_file) and os.path.exists(cor_file + '.xml')):
            print(f"Skipping folder {folder}: coherence file or its XML is missing.")
            continue

        # Load node features only if they haven't been loaded before
        if date1 not in date_to_feature or date2 not in date_to_feature:
            try:
                amp1, amp2 = readAmp(amp_file,bbox)
            except Exception as e:
                print(f"Error reading amplitude file in {folder}: {e}")
                continue
            if date1 not in date_to_feature:
                date_to_feature[date1] = amp1.flatten()
            if date2 not in date_to_feature:
                date_to_feature[date2] = amp2.flatten()

        # Always load the edge feature from the coherence file
        try:
            cor_feature = readCor(cor_file,bbox)
        except Exception as e:
            print(f"Error reading coherence file in {folder}: {e}")
            continue
        edge_features_list.append(cor_feature.flatten())
        edge_node_pairs.append((date1, date2))
        print(f"Processed folder {folder} with dates {date1} and {date2}.")

    # Create a sorted list of unique dates and a mapping: date -> node ID (starting at 1)
    sorted_dates = sorted(date_to_feature.keys())
    date_to_id = {date: idx+1 for idx, date in enumerate(sorted_dates)}

    # Build the node features array: each row is the flattened feature for a node
    node_features = np.array([date_to_feature[date] for date in sorted_dates])
    # Build the edge features array: each row is the flattened edge feature for an edge
    edge_features = np.array(edge_features_list)

    # Save the node mapping to a text file (each line: "nodeID date")
    mapping_file = os.path.join(save_dir, "node_mapping.txt")
    with open(mapping_file, "w") as f:
        for date in sorted_dates:
            f.write(f"{date_to_id[date]} {date}\n")
    print(f"Saved node mapping to '{mapping_file}'.")

    # Save the node mapping to a dataframe
    node_mapping = pd.DataFrame({
        'nodeID': [date_to_id[date] for date in sorted_dates],
        'date': sorted_dates
    })

    # Save the node and edge features as NumPy arrays
    node_features_file = os.path.join(save_dir, "node_features.npy")
    edge_features_file = os.path.join(save_dir, "edge_features.npy")
    np.save(node_features_file, node_features)
    np.save(edge_features_file, edge_features)
    print(f"Saved node features to '{node_features_file}' and edge features to '{edge_features_file}'.")

    # Save the edge node pairs (as node IDs) into a text file.
    edge_pairs_file = os.path.join(save_dir, "edge_node_pairs.txt")
    with open(edge_pairs_file, "w") as f:
        for date1, date2 in edge_node_pairs:
            src = date_to_id[date1]
            tgt = date_to_id[date2]
            # Save with a tab separation between source and target node
            f.write(f"{src}\t{tgt}\n")
    print(f"Saved edge node pairs to '{edge_pairs_file}'.")

    # Report counts
    num_nodes = node_features.shape[0]
    num_edges = edge_features.shape[0]
    print(f"Number of nodes: {num_nodes}")
    print(f"Number of edges: {num_edges}")

    # Optionally return the features and mapping information
    return node_features, edge_features, date_to_id, edge_node_pairs, node_mapping


def stitchPatchMean(merged_dir, patch_file, patch_bbox, patch_length):
    
    # Patch file to stitch to the merged file
    patch = np.load(os.path.join("image_result",patch_file))   

    # Merged file and initialise for first patch if it doesn't exist
    merged_img_file = os.path.join(merged_dir, patch_file.replace("pred_", "pred_merged_img_"))
    merged_wgt_file = os.path.join(merged_dir, patch_file.replace("pred_", "pred_merged_wgt_"))
    if not os.path.exists(merged_img_file):
        merged_img = np.full((patch_length,patch_length), np.nan, dtype=np.float64)
        merged_wgt = np.zeros((patch_length,patch_length), dtype=np.float64)
        np.save(merged_img_file, merged_img)
        np.save(merged_wgt_file, merged_wgt)
    else:
        merged_img = np.load(merged_img_file)
        merged_wgt = np.load(merged_wgt_file)   
    
    # Get size of existing merged image
    merged_x, merged_y = np.shape(merged_img)

    # If existing merged image is too small, compute the new required size
    if patch_bbox[1]+1 > merged_x or patch_bbox[3]+1 > merged_y:
        new_x = max(patch_bbox[1]+1, merged_x)
        new_y = max(patch_bbox[3]+1, merged_y)
        
        # Create a bigger merged image with the new sizes
        new_img = np.full((new_x, new_y), np.nan, dtype=np.float64)
        new_wgt = np.zeros((new_x, new_y), dtype=np.float64)
        
        # Copy existing information into the updated merged image
        new_img[:merged_x, :merged_y] = merged_img
        new_wgt[:merged_x, :merged_y] = merged_wgt
        merged_img = new_img
        merged_wgt = new_wgt
        merged_x, merged_y = np.shape(merged_img)

    # Begin stitching
    # Original data and weights over the region of the patch
    orig_img = merged_img[patch_bbox[0]:patch_bbox[1]+1, patch_bbox[2]:patch_bbox[3]+1]
    orig_wgt = merged_wgt[patch_bbox[0]:patch_bbox[1]+1, patch_bbox[2]:patch_bbox[3]+1]

    # Update the weights over the region of the patch
    merged_wgt[patch_bbox[0]:patch_bbox[1]+1, patch_bbox[2]:patch_bbox[3]+1] += 1

    # Add the patch to the original data and divide by the updated weights 
    # If the original data is NaN, replace with 0 so that it does not affect the summing operation
    merged_img[patch_bbox[0]:patch_bbox[1]+1, patch_bbox[2]:patch_bbox[3]+1] = (np.nan_to_num(orig_img, nan=0.0) * orig_wgt + patch) / merged_wgt[patch_bbox[0]:patch_bbox[1]+1, patch_bbox[2]:patch_bbox[3]+1]
    
    # Save the updated merged image and weight
    np.save(merged_img_file, merged_img)
    np.save(merged_wgt_file, merged_wgt)
    
    # Save figure of the updated merged image 
    plt.imshow(merged_img, cmap='gray', vmin=0, vmax=1)
    plt.colorbar()
    save_path = merged_img_file[:-4] + ".png"
    plt.savefig(save_path)
    plt.close()
    print(f"Saved merged image to {save_path}")
    
    # Optionally return the features and mapping information
    return merged_img, merged_wgt


def stitchPatchMedian(merged_dir, patch_list, pred_file, x, y, chunk_size):

    # Initialise merged image if it doesn't exist
    print(f"Starting merging (median) for {pred_file}...")
    merged_img_file = os.path.join(merged_dir, pred_file.replace("pred_", "pred_merged_img_"))
    if not os.path.exists(merged_img_file):
        merged_img = np.full((x,y), np.nan, dtype=np.float64)
        np.save(merged_img_file, merged_img)
    else:
        merged_img = np.load(merged_img_file)

    # Get list of bounding boxes for each chunk (splitting whole image into chunks to reduce memory usage)
    chunk_list = []
    for i in range(0, y, chunk_size):  
        for j in range(0, x, chunk_size): 
            chunk_list.append((j, min(j+chunk_size,x)-1, i, min(i+chunk_size,y)-1))
    print("Total image length (x-axis): ", x)
    print("Total image width (y-axis): ", y) 
    print("Number of chunks: ", len(chunk_list)) 
    print("Chunk bbox: ", chunk_list)

    # Find all the patches which have overlap with a chunk for each chunk
    patch_to_chunks_id = [[] for _ in range(len(chunk_list))]
    patch_to_chunks_bbox = [[] for _ in range(len(chunk_list))]
    for patch_id, patch in enumerate(patch_list):
        px1, px2, py1, py2 = patch
        for chunk_id, chunk_bbox in enumerate(chunk_list):
            cx1, cx2, cy1, cy2 = chunk_bbox
            if not (px2 < cx1 or px1 > cx2 or py2 < cy1 or py1 > cy2):
                # Overlap condition: 
                # check if the right edge of the patch is to the left of the left edge of the chunk,
                # check if the left edge of the patch is to the right of the right edge of the chunk,
                # check if the top edge of the patch is to the bottom of the bottom edge of the chunk,
                # check if the bottom edge of the patch is to the top of the top edge of the chunk,
                # if any of these conditions are false, then there is an overlap
                patch_to_chunks_id[chunk_id].append(patch_id+1) 
                patch_to_chunks_bbox[chunk_id].append(patch) 
    
    # Loop through each chunk 
    for chunk_id, chunk_bbox in enumerate(chunk_list):
        print(f"Chunk {chunk_id+1} with bbox {chunk_bbox} overlaps with the following {len(patch_to_chunks_id[chunk_id])} patch folders:")
        print(patch_to_chunks_id[chunk_id])
        cx1, cx2, cy1, cy2 = chunk_bbox

        # Initialise empty 3D chunk array
        chunk_stack = np.full((cx2-cx1+1, cy2-cy1+1, len(patch_to_chunks_id[chunk_id])), np.nan, dtype=np.float64)

        # Loop through each patch that overlaps with the chunk
        for idx, patch_id in enumerate(patch_to_chunks_id[chunk_id]):
            patch_file = os.path.join("patch_"+f"{patch_id:04d}", "image_result", pred_file)

            if os.path.exists(patch_file):
                # Load data from patch file
                patch_img = np.load(patch_file)

                # Get the bounding box of the patch
                px1, px2, py1, py2 = patch_to_chunks_bbox[chunk_id][idx]

                # Get indices of the chunk for which the patch fits within the chunk
                ccx1 = max(px1, cx1) - cx1
                ccx2 = min(px2, cx2) - cx1
                ccy1 = max(py1, cy1) - cy1
                ccy2 = min(py2, cy2) - cy1

                # Get indices of the patch for which the patch fits within the chunk
                ppx1 = max(px1, cx1) - px1
                ppx2 = min(px2, cx2) - px1
                ppy1 = max(py1, cy1) - py1
                ppy2 = min(py2, cy2) - py1

                # Stack the patch into the 3D chunk array
                chunk_stack[ccx1:ccx2+1, ccy1:ccy2+1, idx] = patch_img[ppx1:ppx2+1, ppy1:ppy2+1]
            
            else:
                print(f"Patch file {patch_file} not found.")

        # Compute median of the 3D chunk array and output to the larger merged image
        merged_img[cx1:cx2+1, cy1:cy2+1] = np.nanmedian(chunk_stack, axis=2)

        # Save the updated merged image at the end of each chunk
        np.save(merged_img_file, merged_img)
        print(f"Updated merged image with chunk {chunk_id+1} in {merged_img_file}")

     # Plot and save merged image once all chunks are processed
    plt.imshow(merged_img, cmap='gray', vmin=0, vmax=1)
    plt.colorbar()
    save_path = os.path.join(merged_dir, pred_file[:-3].replace("pred_", "pred_merged_img_")+'png')
    plt.savefig(save_path)
    plt.close()
    print(f"Saved merged image for {pred_file} to {save_path}")
