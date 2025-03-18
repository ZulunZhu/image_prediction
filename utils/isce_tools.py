import os
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt


def readAmp(file,bbox):
    # To-do for Cheryl: optimise reading binary file using bytes to avoid reading whole image into memory
    width, length = sizeFromXml(file)
    with open(file, 'rb') as f:
        data = np.fromfile(f, dtype='<f4')
        data2 = data.reshape([length, width * 2])
        amp1 = data2[:, ::2]
        amp2 = data2[:, 1::2]
    amp1 = amp1[bbox[0]:bbox[1], bbox[2]:bbox[3]]
    amp2 = amp2[bbox[0]:bbox[1], bbox[2]:bbox[3]]
    # Optionally disable plotting for batch processing
    plt.imshow(amp1, cmap='gray', vmin=0, vmax=5)
    plt.colorbar(); plt.title("Amplitude 1"); plt.show()
    plt.imshow(amp2, cmap='gray', vmin=0, vmax=5)
    plt.colorbar(); plt.title("Amplitude 2"); plt.show()
    return amp1, amp2


def readCor(file,bbox):
    # To-do for Cheryl: optimise reading binary file using bytes to avoid reading whole image into memory
    width, length = sizeFromXml(file)
    with open(file, 'rb') as f:
        data = np.fromfile(f, dtype=np.float32)
        data2 = data.reshape([length * 2, width])
        # Use the second band (odd rows) as the coherence (edge) feature
        cor = data2[1::2, :]
    cor = cor[bbox[0]:bbox[1], bbox[2]:bbox[3]]
    # Optionally disable plotting for batch processing
    plt.imshow(cor, cmap='gray', vmin=0, vmax=5)
    plt.colorbar(); plt.title("Cor 2"); plt.show()
    return cor


def sizeFromXml(file):
    fileXml = file + '.xml'
    tree = ET.parse(fileXml)
    root = tree.getroot()
    width = int(root.find(".//property[@name='width']/value").text)
    length = int(root.find(".//property[@name='length']/value").text)
    return width, length   


def computePatches(image_width, image_length, patch_size, patch_overlap):
    # Compute step sizes
    patchList = []
    step_width = patch_size - patch_overlap
    step_length = patch_size - patch_overlap
    # Define patches
    for start_width in range(0, image_width - patch_size + 1, step_width):
        for start_length in range(0, image_length - patch_size + 1, step_length):
            patchList.append([start_width, start_width + patch_size, start_length, start_length + patch_size])
    # Handle the right and bottom edges
    if (image_width - patch_size) % step_width != 0:
        for start_length in range(0, image_length - patch_size + 1, step_length):
            patchList.append([image_width - patch_size, image_width, start_length, start_length + patch_size])
    if (image_length - patch_size) % step_length != 0:
        for start_width in range(0, image_width - patch_size + 1, step_width):
            patchList.append([start_width, start_width + patch_size, image_length - patch_size, image_length])
    if (image_width - patch_size) % step_width != 0 and (image_length - patch_size) % step_length != 0:
        patchList.append([image_width - patch_size, image_width, image_length - patch_size, image_length])
    print("Total image width: ", image_width)
    print("Total image width: ", image_length)
    print("Number of patches: ", np.shape(patchList)[0])
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

