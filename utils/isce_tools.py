import os
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt


def readAmp(file,bbox):
    # TO-DO for CT: optimise reading binary file using bytes to avoid reading whole image into memory
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
    return amp1, amp2


def readCor(file,bbox):
    # TO-DO for CT: optimise reading binary file using bytes to avoid reading whole image into memory
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
    return cor


def sizeFromXml(file):
    fileXml = file + '.xml'
    tree = ET.parse(fileXml)
    root = tree.getroot()
    width = int(root.find(".//property[@name='width']/value").text)
    length = int(root.find(".//property[@name='length']/value").text)
    return width, length   


def computePatches(image_y, image_x, patch_length, patch_overlap):
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
    print("Total image height (y-axis): ", image_y) 
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
    subdirs.sort()  # sorting ensures that if the same date appears in multiple folders, the earliest is processed first

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
    return node_features, edge_features, date_to_id, edge_node_pairs


def stitchPatch(merged_dir, patch_file, patch_bbox, patch_length, merge_method='mean'):
    
    # Patch file to stitch to the merged file
    patch = np.load(os.path.join("image_result",patch_file))   

    # Merged file and initialise for first patch if it doesn't exist
    merged_img_file = os.path.join(merged_dir, patch_file.replace("pred_", "pred_merged_img_"))
    merged_wgt_file = os.path.join(merged_dir, patch_file.replace("pred_", "pred_merged_wgt_"))
    if not os.path.exists(merged_img_file):
        merged_img = np.zeros((patch_length,patch_length), dtype=np.float64)
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
        new_x = max(merged_x, patch_bbox[1]+1)
        new_y = max(merged_y, patch_bbox[3]+1)
        
        # Create a bigger merged image with the new sizes
        new_img = np.zeros((new_x, new_y), dtype=np.float64)
        new_wgt = np.zeros((new_x, new_y), dtype=np.float64)
        
        # Copy existing information into the updated merged image
        new_img[:merged_x, :merged_y] = merged_img
        new_wgt[:merged_x, :merged_y] = merged_wgt
        merged_img = new_img
        merged_wgt = new_wgt
        merged_x, merged_y = np.shape(merged_img)

    # Stitch patch to the merged image using either mean, min, or max method
    # TO-DO: median requires storing values of individual patches and can't be implemented sequentially 
    if merge_method == 'mean':
        # Original data and weights over the region of the patch
        orig_img = merged_img[patch_bbox[0]:patch_bbox[1]+1, patch_bbox[2]:patch_bbox[3]+1]
        orig_wgt = merged_wgt[patch_bbox[0]:patch_bbox[1]+1, patch_bbox[2]:patch_bbox[3]+1]
        # Update the weights over the region of the patch
        merged_wgt[patch_bbox[0]:patch_bbox[1]+1, patch_bbox[2]:patch_bbox[3]+1] += 1
        # Add the patch to the original data and divide by the updated weights 
        merged_img[patch_bbox[0]:patch_bbox[1]+1, patch_bbox[2]:patch_bbox[3]+1] = (orig_img * orig_wgt + patch) / merged_wgt[patch_bbox[0]:patch_bbox[1]+1, patch_bbox[2]:patch_bbox[3]+1]
    elif merge_method == 'min' or merge_method == 'max':
        # Create a dummy merged image populated with the patch only
        patch_img = np.zeros((merged_x, merged_y), dtype=np.float64)
        patch_img[patch_bbox[0]:patch_bbox[1]+1, patch_bbox[2]:patch_bbox[3]+1] = patch
        # Find non-overlapping region between patch and merged image (mask=1 if weight is still 0, mask=0 if weight is > 0)
        mask = (merged_wgt == 0)
        # For non-overlapping region, copy the patch values to the merged image
        merged_img[mask] = patch_img[mask]
        # For overlapping region, apply min or max method
        if merge_method == 'min':
            merged_img[~mask] = np.minimum(merged_img[~mask], patch_img[~mask])
        elif merge_method == 'max':
            merged_img[~mask] = np.maximum(merged_img[~mask], patch_img[~mask])
    else:
        raise ValueError("Invalid merge_method. Choose from 'mean', 'min', or 'max'.")
    
    # Save the updated merged image and weight
    np.save(merged_img_file, merged_img)
    np.save(merged_wgt_file, merged_wgt)
    
    # Save figure of the updated merged image 
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(merged_img, cmap='gray', vmin=0, vmax=1)
    axs[0].set_title("Merged")
    axs[0].axis('off')
    plt.tight_layout()
    save_path = merged_img_file[:-4] + ".png"
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved merged image to {save_path}")
    
    # Optionally return the features and mapping information
    return merged_img, merged_wgt
