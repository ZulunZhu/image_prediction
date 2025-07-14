import logging
import time
import sys
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import warnings
import shutil
import json
import math
import re
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from tgb.linkproppred.evaluate import Evaluator

from models.TGAT import TGAT
from models.MemoryModel import MemoryModel, compute_src_dst_node_time_shifts
from models.CAWN import CAWN
from models.TCL import TCL
from models.GraphMixer import GraphMixer
from models.DyGFormer import DyGFormer
from models.modules import MergeLayer
from utils.isce_tools import sizeFromXml, computePatches, prepareData, stitchPatchMean, stitchPatchMedian
from utils.utils import set_random_seed, convert_to_gpu, get_parameter_sizes, create_optimizer
from utils.utils import get_neighbor_sampler, NegativeEdgeSampler
from evaluate_models_utils import evaluate_model_link_prediction, evaluate_image_link_prediction_without_dataloader, evaluate_image_link_prediction_visualiser, evaluate_image_link_prediction_plot_distributions
from utils.DataLoader import get_idx_data_loader, get_link_prediction_tgb_data, get_link_prediction_image_data, \
get_link_prediction_image_data_split_by_nodes, load_edge_node_pairs, get_sliding_window_data_loader
from utils.temporal_shifting import find_furthest_node_within_temporal_baseline
from utils.EarlyStopping import EarlyStopping
from utils.load_configs import get_link_prediction_args
                                

if __name__ == "__main__":

    # Ignore warnings
    warnings.filterwarnings('ignore')

    # Set Numpy print options
    np.set_printoptions(suppress=True, precision=5)

    # Set PyTorch print options
    torch.set_printoptions(sci_mode=False, precision=5)
    
    # Increase recursion limit to avoid crashes
    sys.setrecursionlimit(100000)  

    # Input data
    # data_dir = "/aria/zulun_sharing/TEST_KZ_TGNN_sharing_20250214/merged/cors_ALL"  # Directory containing original image data (uncropped from ISCE2)
    data_dir = "/aria/tgnn_dpm5/usa_california_wildfires/P064/merged/cors"  # Directory containing original image data (cropped to tight bounding box)

    # Set up main logger
    work_dir = os.getcwd()
    main_log_dir = os.path.join(work_dir, 'main_logs')
    os.makedirs(main_log_dir, exist_ok=True)
    log_main = logging.getLogger("LogMain")
    log_main.setLevel(logging.DEBUG)
    log_main.handlers = []  # Clear any existing handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh = logging.FileHandler(os.path.join(main_log_dir, f"main_logger_{time.strftime('%Y%m%d_%H%M%S')}.log"))  # Create file handler 
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    log_main.addHandler(fh)
    ch = logging.StreamHandler(sys.__stdout__)  # Create console handler 
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    log_main.addHandler(ch)
    
    # Get arguments from load_configs.py
    args = get_link_prediction_args(is_evaluation=False)
    log_main.info(f"\n\n\n~~~~~ The script '{sys.argv[0]}' has STARTED! ~~~~~\n\n\n"
                  f"CONFIGURATION: {args}\n")
    
    # Start time for the main script
    start_time_main = time.time()

    # Get list bounding boxes for each patch
    sub_dirs = [d for d in os.listdir(data_dir) if d.startswith("cor_") and os.path.isdir(os.path.join(data_dir, d))]  # Get list of all cor_YYYYMMDD_YYYYMMDD subfolders
    cor_files = [f for f in os.listdir(os.path.join(data_dir, sub_dirs[0])) if f.endswith(".cor")]  # Get list of all .cor files from the first cor_YYYYMMDD_YYYYMMDD subfolder
    if not sub_dirs or not cor_files:
        log_main.error("ERROR: There are no 'cor_YYYYMMDD_YYYYMMDD' subfolders or '.cor' files found in the data directory. The script will be terminated.")
        sys.exit(1)
    ref_img = os.path.join(data_dir, sub_dirs[0], cor_files[0])  # Use first file in first cor_YYYYMMDD_YYYYMMDD subfolder as reference image for patch splitting
    y, x = sizeFromXml(ref_img)  # Get size of full image    
    patchList = computePatches(ref_img, x, y, args.patch_length, args.patch_overlap)   # Get list of bounding boxes for each patch
    log_main.info(f"\n"
                  f"Total image width (x-axis): {x}\n"
                  f"Total image height (y-axis): {y}\n"
                  f"Number of patches: {len(patchList)}\n"
                  f"Patch bbox: {patchList}\n")

    # Initialize merged image directory
    merged_dir = os.path.join(work_dir,'merged')
    os.makedirs(merged_dir, exist_ok=True)

    ######### PATCH LOOP #########
    # Iterate over each image patch
    for patch_id, patch_bbox in enumerate(patchList):
        
        # Debugging: run the patch only if it overlaps with the target area
        # x1, x2, y1, y2 = patch_bbox
        # target_x1, target_x2 = 650, 850
        # target_y1, target_y2 = 20, 220
        # if (x2 < target_x1 or x1 > target_x2 or y2 < target_y1 or y1 > target_y2):
        #     continue
        
        # Create sub-directory for the patch
        patch_dir = os.path.join(work_dir,'patch_' + f"{patch_id+1:05d}")
        os.makedirs(patch_dir, exist_ok=True)
        
        # Set up logger for this patch 
        patch_log_dir = patch_dir + '/patch_logs'
        os.makedirs(patch_log_dir, exist_ok=True)
        log_patch = logging.getLogger("LogPatch")
        log_patch.setLevel(logging.DEBUG)
        log_patch.handlers = []  # Clear any existing handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh = logging.FileHandler(os.path.join(patch_log_dir, f"patch_logger_{time.strftime('%Y%m%d_%H%M%S')}.log"))  # Create file handler 
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        log_patch.addHandler(fh)
        ch = logging.StreamHandler(sys.__stdout__)  # Create console handler 
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        log_patch.addHandler(ch)

        # Process dataset for the patch
        log_patch.info(f"\n********** PATCH {patch_id + 1:05d} **********\n")
        log_patch.info(f"Patch bbox: {patch_bbox}")
        _, _, date_to_id, edge_node_pairs, node_mapping = prepareData(log_patch, data_dir, patch_dir, patch_bbox, amp_norm=args.amp_norm, cor_logit=args.cor_logit)

        # Go into the patch directory
        os.chdir(patch_dir)

        # Get data for training, validation and testing
        node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data, eval_neg_edge_sampler, eval_metric_name = \
            get_link_prediction_image_data_split_by_nodes(log_patch, dataset_name=args.dataset_name)

        # The neighbor sampler lists for each node: all the other nodes that it interacts with regardless of backward or forward direction, and the edge and time at which the interaction occurs
        # Initialize training neighbor sampler to retrieve temporal graph
        train_neighbor_sampler = get_neighbor_sampler(data=train_data, sample_neighbor_strategy=args.sample_neighbor_strategy,
                                                    time_scaling_factor=args.time_scaling_factor, seed=0)

        # Initialize validation and test neighbor sampler to retrieve temporal graph
        full_neighbor_sampler = get_neighbor_sampler(data=full_data, sample_neighbor_strategy=args.sample_neighbor_strategy,
                                                    time_scaling_factor=args.time_scaling_factor, seed=1)

        # Initialize train negative sampler
        # train_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=train_data.src_node_ids, dst_node_ids=train_data.dst_node_ids)

        # Print a summary of the training / validation / test nodes & edges
        log_patch.info(f"Date to ID mapping:\n{date_to_id}\n")
        log_patch.info(f"train_data.src_node_ids ({len(train_data.src_node_ids)} edges):\n{train_data.src_node_ids}\n")
        log_patch.info(f"train_data.dst_node_ids ({len(train_data.dst_node_ids)} edges):\n{train_data.dst_node_ids}\n")
        log_patch.info(f"train_data.edge_ids ({len(train_data.edge_ids)} edges):\n{train_data.edge_ids}\n")
        log_patch.info(f"val_data.src_node_ids ({len(val_data.src_node_ids)} edges):\n{val_data.src_node_ids}\n")
        log_patch.info(f"val_data.dst_node_ids ({len(val_data.dst_node_ids)} edges):\n{val_data.dst_node_ids}\n")
        log_patch.info(f"val_data.edge_ids ({len(val_data.edge_ids)} edges):\n{val_data.edge_ids}\n")
        log_patch.info(f"test_data.src_node_ids ({len(test_data.src_node_ids)} edges):\n{test_data.src_node_ids}\n")
        log_patch.info(f"test_data.dst_node_ids ({len(test_data.dst_node_ids)} edges):\n{test_data.dst_node_ids}\n")
        log_patch.info(f"test_data.edge_ids ({len(test_data.edge_ids)} edges):\n{test_data.edge_ids}\n")

        # Compute the desired batch size (number of edges)
        # By default, 1 date is paired with the next 10 dates, hence 11 dates in total for one temporal window
        # batch_size = math.comb(args.temporal_window_num_neighbours + 1, 2)   # Max number of edges in one temporal window

        # Training nodes
        train_data_src_node_start = min(train_data.src_node_ids)
        train_data_src_node_end = max(train_data.src_node_ids)
        train_data_dst_node_start = min(train_data.dst_node_ids)
        train_data_dst_node_end = max(train_data.dst_node_ids)
            
        # Validation nodes
        val_data_src_node_start = min(val_data.src_node_ids)
        val_data_src_node_end = max(val_data.src_node_ids)
        val_data_dst_node_start = min(val_data.dst_node_ids)
        val_data_dst_node_end = max(val_data.dst_node_ids)

        # Testing nodes
        test_data_src_node_start = min(test_data.src_node_ids)
        test_data_src_node_end = max(test_data.src_node_ids)
        test_data_dst_node_start = min(test_data.dst_node_ids)
        test_data_dst_node_end = max(test_data.dst_node_ids)

        # Load the edge node pairs from the file
        edge_node_pairs = load_edge_node_pairs('./image_data/edge_node_pairs.txt')

        # Storage arrays for metrics
        val_metric_all_runs, test_metric_all_runs = [], []

        ######### RUN LOOP #########
        for run in range(args.num_runs):

            # Set random seed. The seed is set to be the same as the run number.
            set_random_seed(seed=run)
            args.seed = run
            args.save_model_name = f"{args.model_name}_seed{args.seed}_{time.strftime('%Y%m%d_%H%M%S')}"

            # Start the timer for the run
            run_start_time = time.time()

            log_patch.info(f"********** PATCH {patch_id + 1:05d} | RUN {run + 1} **********\n")
            log_patch.info(f"CONFIGURATION: {args}\n")
            
            # Create model
            if args.model_name == 'TGAT':
                dynamic_backbone = TGAT(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler,
                                        time_feat_dim=args.time_feat_dim, output_dim=args.output_dim, num_layers=args.num_layers, num_heads=args.num_heads,
                                        dropout=args.dropout, device=args.device)
            elif args.model_name in ['JODIE', 'DyRep', 'TGN']:
                # four floats that represent the mean and standard deviation of source and destination node time shifts in the training data, which is used for JODIE
                src_node_mean_time_shift, src_node_std_time_shift, dst_node_mean_time_shift_dst, dst_node_std_time_shift = \
                    compute_src_dst_node_time_shifts(train_data.src_node_ids, train_data.dst_node_ids, train_data.node_interact_times)
                dynamic_backbone = MemoryModel(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler,
                                            time_feat_dim=args.time_feat_dim, output_dim=args.output_dim, model_name=args.model_name, num_layers=args.num_layers, num_heads=args.num_heads,
                                            dropout=args.dropout, src_node_mean_time_shift=src_node_mean_time_shift, src_node_std_time_shift=src_node_std_time_shift,
                                            dst_node_mean_time_shift_dst=dst_node_mean_time_shift_dst, dst_node_std_time_shift=dst_node_std_time_shift, device=args.device)
            elif args.model_name == 'CAWN':
                dynamic_backbone = CAWN(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler,
                                        time_feat_dim=args.time_feat_dim, position_feat_dim=args.position_feat_dim, output_dim=args.output_dim, walk_length=args.walk_length,
                                        num_walk_heads=args.num_walk_heads, dropout=args.dropout, device=args.device)
            elif args.model_name == 'TCL':
                dynamic_backbone = TCL(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler,
                                    time_feat_dim=args.time_feat_dim, output_dim=args.output_dim, num_layers=args.num_layers, num_heads=args.num_heads,
                                    num_depths=args.num_neighbors + 1, dropout=args.dropout, device=args.device)
            elif args.model_name == 'GraphMixer':
                dynamic_backbone = GraphMixer(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler,
                                            time_feat_dim=args.time_feat_dim, output_dim=args.output_dim, num_tokens=args.num_neighbors, num_layers=args.num_layers,
                                            dropout=args.dropout, device=args.device)
            elif args.model_name == 'DyGFormer':
                dynamic_backbone = DyGFormer(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler,
                                            time_feat_dim=args.time_feat_dim, channel_embedding_dim=args.channel_embedding_dim, output_dim=args.output_dim,
                                            patch_size=args.patch_size, num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout,
                                            max_input_sequence_length=args.max_input_sequence_length, device=args.device)
            else:
                raise ValueError(f"Wrong value for model_name {args.model_name}!")
            
            # Create the link predictor
            link_predictor = MergeLayer(input_dim1=args.output_dim, input_dim2=args.output_dim, hidden_dim=args.output_dim, output_dim=edge_raw_features.shape[1])

            # Model consists of two parts: dynamic backbone "model[0]" and link predictor "model[1]"
            model = nn.Sequential(dynamic_backbone, link_predictor)
            log_patch.info(f"model:\n{model}")
            log_patch.info(f"model name: {args.model_name}, #parameters: {get_parameter_sizes(model) * 4} B, "
                        f"{get_parameter_sizes(model) * 4 / 1024} KB, {get_parameter_sizes(model) * 4 / 1024 / 1024} MB.")

            optimizer = create_optimizer(model=model, optimizer_name=args.optimizer, learning_rate=args.learning_rate, weight_decay=args.weight_decay)

            model = convert_to_gpu(model, device=args.device)

            save_model_folder = f"./saved_models/{args.model_name}/{args.dataset_name}/"
            shutil.rmtree(save_model_folder, ignore_errors=True)
            os.makedirs(save_model_folder, exist_ok=True)

            early_stopping = EarlyStopping(patience=args.patience, patience_threshold=args.patience_threshold, save_model_folder=save_model_folder,
                                        save_model_name=args.save_model_name, logger=log_patch, model_name=args.model_name)

            # loss_func = nn.BCELoss()    # BCE loss is not used as of the moment. Using L1 loss (MAE) instead.  
            evaluator = Evaluator(name=args.dataset_name)
            
            # Storage arrays for validation and test metrics
            val_metrics, test_metrics = [], []

            ######### EPOCH LOOP #########
            for epoch in range(args.num_epochs):

                log_patch.info(f"TRAINING with {args.num_epochs} epochs...")
                log_patch.info(f"********** PATCH {patch_id + 1:05d} | RUN {run + 1} | EPOCH {epoch + 1} **********\n")

                # Set model to training mode
                model.train()
                
                if args.model_name in ['DyRep', 'TGAT', 'TGN', 'CAWN', 'TCL', 'GraphMixer', 'DyGFormer']:
                    # training, only use training graph
                    model[0].set_neighbor_sampler(train_neighbor_sampler)
                if args.model_name in ['JODIE', 'DyRep', 'TGN']:
                    # reinitialize memory of memory-based models at the start of each epoch
                    model[0].memory_bank.__init_memory_bank__()

                ###### PREPARATION FOR TRAINING #######

                # Setting up the training exclusion list (python ids of the edges to be excluded from training)
                # If no edges are to be excluded, the list should be empty
                train_exclusion_list = []   

                # Storage arrays for train losses and metrics
                train_losses, train_metrics = [], [] 

                # Storage arrays for (1) train_data_indices (python indices for the training data), 
                # and (2) number of edges in each training window. 
                storage_train_window_train_data_indices = [np.array([])]       # Python indices for the training data in each training window.
                storage_train_window_num_edges = [0]                           # Number of edges in each training window. Initiate with 0 for the 0th window.
                
                # Iterating through the training windows with a stride of 1 per iteration (default)
                for train_window_idx in enumerate(range(train_data_src_node_start, train_data_dst_node_end + 1, args.temporal_window_stride)):
                    
                    log_patch.info(f"********** PATCH {patch_id + 1:05d} | RUN {run + 1} | EPOCH {epoch + 1} | TRAINING WINDOW {train_window_idx[0] + 1} **********")

                    # Get the start and end of the src and dst nodes of the temporal window in the specified iteration
                    train_window_src_node_start = train_data_src_node_start + train_window_idx[0]
                    train_window_dst_node_start = train_window_src_node_start + 1
                    train_window_dst_node_end = find_furthest_node_within_temporal_baseline(
                                                    node_mapping = node_mapping,
                                                    window_src_node_start = train_window_src_node_start,
                                                    temporal_window_width = args.temporal_window_width
                                                    )
                    train_window_src_node_end = train_window_dst_node_end - 1

                    # Get the train_data_indices for the current temporal window
                    # These are the python reference indices, NOT the exact edge ids
                    train_data_indices = np.where(((train_data.src_node_ids >= train_window_src_node_start) & 
                                                  (train_data.src_node_ids <= train_window_src_node_end)) & 
                                                  ((train_data.dst_node_ids >= train_window_dst_node_start) & 
                                                  (train_data.dst_node_ids <= train_window_dst_node_end)))[0]
                    
                    # Remove the elements from the train_data_indices that are in the training exclusion list
                    train_data_indices = train_data_indices[~np.isin(train_data_indices, train_exclusion_list)]
                    
                    # Append (1) the train_data_indices, and (2) the number of edges in the current training window to the storage arrays
                    storage_train_window_train_data_indices.append(train_data_indices)
                    storage_train_window_num_edges.append(len(train_data_indices))  

                    # Determine if the temporal sliding window should undergo training based on 2 criteria:
                    # (1) Number of total edges in the temporal window must be >= args.min_num_total_edges_valid_training_window
                    # (2) Number of new edges in the temporal window must be >= args.min_num_new_edges_valid_training_window

                    # Get numbers of total and new edges in the training window
                    train_window_num_total_edges = len(storage_train_window_train_data_indices[train_window_idx[0] + 1])
                    train_window_num_new_edges = len(set(storage_train_window_train_data_indices[(train_window_idx[0] + 1)]) - set(storage_train_window_train_data_indices[train_window_idx[0]]))

                    # Check if the current window is valid for training based on the 2 criteria
                    if (train_window_num_total_edges >= args.min_num_total_edges_valid_training_window) and \
                        (train_window_num_new_edges >= args.min_num_new_edges_valid_training_window):

                        log_patch.info(f"Training for WINDOW {train_window_idx[0] + 1} begins. | Total Edges: {train_window_num_total_edges} ({train_window_num_new_edges} new)")

                        ######## TRAINING BEGINS ########

                        # Get the batch of training data for specified temporal window only
                        batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids = \
                            train_data.src_node_ids[train_data_indices], train_data.dst_node_ids[train_data_indices], \
                            train_data.node_interact_times[train_data_indices], train_data.edge_ids[train_data_indices]    
                        
                        # Logging information about the specified temporal window
                        log_patch.info(f"train_window_idx: {train_window_idx[0] + 1}")
                        log_patch.info(f"train_window_src_node_start: {train_window_src_node_start}")
                        log_patch.info(f"train_window_src_node_end: {train_window_src_node_end}")
                        log_patch.info(f"train_window_dst_node_start: {train_window_dst_node_start}")
                        log_patch.info(f"train_window_dst_node_end: {train_window_dst_node_end}")
                        log_patch.info(f"train_data_indices:\n{train_data_indices}")
                        log_patch.info(f"batch_src_node_ids:\n{batch_src_node_ids}")
                        log_patch.info(f"batch_dst_node_ids:\n{batch_dst_node_ids}")
                        log_patch.info(f"batch_node_interact_times:\n{batch_node_interact_times}")
                        log_patch.info(f"batch_edge_ids ({len(batch_edge_ids)} edges):\n{batch_edge_ids}")

                        # We need to compute for positive and negative edges respectively, because the new sampling strategy (for evaluation) allows the negative source nodes to be
                        # different from the source nodes, this is different from previous works that just replace destination nodes with negative destination nodes
                        if args.model_name in ['TGAT', 'CAWN', 'TCL']:
                            # Get temporal embedding of source and destination nodes: 2 tensors, with shape (batch_size, output_dim)
                            batch_src_node_embeddings, batch_dst_node_embeddings = \
                                model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                                dst_node_ids=batch_dst_node_ids,
                                                                                node_interact_times=batch_node_interact_times,
                                                                                num_neighbors=args.num_neighbors)
                            # Get temporal embedding of source and destination nodes: 2 tensors, with shape (batch_size, output_dim)
                            # batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                            #     model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                            #                                                       dst_node_ids=batch_neg_dst_node_ids,
                            #                                                       node_interact_times=batch_node_interact_times,
                            #                                                       num_neighbors=args.num_neighbors)
                            
                        elif args.model_name in ['JODIE', 'DyRep', 'TGN']:
                            # Note that negative nodes do not change the memories while the positive nodes change the memories,
                            # we need to first compute the embeddings of negative nodes for memory-based models
                            # Get temporal embedding of source and destination nodes: 2 tensors, with shape (batch_size, output_dim)
                            batch_src_node_embeddings, batch_dst_node_embeddings = \
                                model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                                dst_node_ids=batch_dst_node_ids,
                                                                                node_interact_times=batch_node_interact_times,
                                                                                edge_ids=batch_edge_ids,
                                                                                edges_are_positive=True,
                                                                                num_neighbors=args.num_neighbors)
                            # Get temporal embedding of source and destination nodes: 2 tensors, with shape (batch_size, output_dim)
                            # batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                            #     model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                            #                                                       dst_node_ids=batch_neg_dst_node_ids,
                            #                                                       node_interact_times=batch_node_interact_times,
                            #                                                       edge_ids=None,
                            #                                                       edges_are_positive=False,
                            #                                                       num_neighbors=args.num_neighbors)
                            
                        elif args.model_name in ['GraphMixer']:
                            # Get temporal embedding of source and destination nodes: 2 tensors, with shape (batch_size, output_dim)
                            batch_src_node_embeddings, batch_dst_node_embeddings = \
                                model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                                dst_node_ids=batch_dst_node_ids,
                                                                                node_interact_times=batch_node_interact_times,
                                                                                num_neighbors=args.num_neighbors,
                                                                                time_gap=args.time_gap)
                            # Get temporal embedding of source and destination nodes: 2 tensors, with shape (batch_size, output_dim)
                            # batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                            #     model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                            #                                                       dst_node_ids=batch_neg_dst_node_ids,
                            #                                                       node_interact_times=batch_node_interact_times,
                            #                                                       num_neighbors=args.num_neighbors,
                            #                                                       time_gap=args.time_gap)
                            
                        elif args.model_name in ['DyGFormer']:
                            # Get temporal embedding of source and destination nodes: 2 tensors, with shape (batch_size, output_dim)
                            batch_src_node_embeddings, batch_dst_node_embeddings = \
                                model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                                dst_node_ids=batch_dst_node_ids,
                                                                                node_interact_times=batch_node_interact_times)
                            # Get temporal embedding of source and destination nodes: 2 tensors, with shape (batch_size, output_dim)
                            # batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                            #     model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                            #                                                       dst_node_ids=batch_neg_dst_node_ids,
                            #                                                       node_interact_times=batch_node_interact_times)
                            
                        else:
                            raise ValueError(f"Wrong value for model_name {args.model_name}!")
                        
                        # Get positive and negative probabilities, shape (batch_size, )
                        # positive_probabilities = model[1](input_1=batch_src_node_embeddings, input_2=batch_dst_node_embeddings).squeeze(dim=-1)
                        # negative_probabilities = model[1](input_1=batch_neg_src_node_embeddings, input_2=batch_neg_dst_node_embeddings).squeeze(dim=-1)

                        # predicts = torch.cat([positive_probabilities, negative_probabilities], dim=0)
                        # labels = torch.cat([torch.ones_like(positive_probabilities), torch.zeros_like(negative_probabilities)], dim=0)

                        # loss = loss_func(input=predicts, target=labels)
                        
                        # Predict the edge features
                        # UNCOMMENT FOR DEBUGGING: Uncomment the following lines for edge predictions with different modification operations
                        # When changing the following lines, we must also change the edge prediction lines in "evaluate_models_utils.py"
                        # (1) predictions without modification operations 
                        # (2) predictions with an external sigmoid operation (not recommended)
                        # (3) predictions with an external clamp operation (not recommended)
                        predicted_edge_feature = model[1](input_1=batch_src_node_embeddings, input_2=batch_dst_node_embeddings)
                        # predicted_edge_feature = model[1](input_1=batch_src_node_embeddings, input_2=batch_dst_node_embeddings).sigmoid()
                        # predicted_edge_feature = model[1](input_1=batch_src_node_embeddings, input_2=batch_dst_node_embeddings).clamp(min=0.0, max=1.0)

                        # Acquire the residuals_full_object, where Residuals = (Ground Truth - Prediction)
                        residuals_full_object = torch.tensor(edge_raw_features[train_data_indices + 1], dtype=torch.float32, device = args.device) - predicted_edge_feature

                        # UNCOMMENT FOR DEBUGGING: Log the Ground Truth, Prediction, Residuals, and Absolute Residuals 
                        log_patch.info(f"[TRAINING] GROUND TRUTH (Dimensions: {torch.tensor(edge_raw_features[train_data_indices + 1], dtype=torch.float32, device = args.device).shape}): \n{torch.tensor(edge_raw_features[train_data_indices + 1], dtype=torch.float32, device = args.device)}\n")
                        log_patch.info(f"[TRAINING] PREDICTION (Dimensions: {predicted_edge_feature.shape}): \n{predicted_edge_feature}\n")
                        log_patch.info(f"[TRAINING] (GROUND TRUTH - PREDICTION) (Dimensions: {residuals_full_object.shape}): \n{residuals_full_object}\n")
                        log_patch.info(f"[TRAINING] abs(GROUND TRUTH - PREDICTION) (Dimensions: {residuals_full_object.abs().shape}): \n{residuals_full_object.abs()}\n")

                        # Log the training result statistics
                        quantiles = torch.tensor([0.00, 0.01, 0.25, 0.5, 0.75, 0.99, 1.00], device=args.device)
                        log_patch.info(f"\n"
                                       f"[TRAINING] STATISTICS\n"
                                       f"[ MIN | 1st | 25th | 50th | 75th | 99th | MAX ]\n"
                                       f"GROUND TRUTH Percentiles: \n"
                                       f"{[round(v, 5) for v in torch.quantile(torch.tensor(edge_raw_features[train_data_indices + 1], dtype=torch.float32, device = args.device).flatten(), quantiles).tolist()]}\n"
                                       f"PREDICTION Percentiles: \n"
                                       f"{[round(v, 5) for v in torch.quantile(predicted_edge_feature.flatten(), quantiles).tolist()]}\n"
                                       f"LOSS Percentiles: \n"
                                       f"{[round(v, 5) for v in torch.quantile(residuals_full_object.flatten(), quantiles).tolist()]}\n"
                                       f"abs(LOSS) Percentiles: \n"
                                       f"{[round(v, 5) for v in torch.quantile(residuals_full_object.abs().flatten(), quantiles).tolist()]}\n")
                        
                        # Compute the L1 loss (MAE) between the predicted edge feature and the ground truth edge feature
                        # Note that edge_raw_features shape is [total no. of edges in full dataset +1, no. of pixels in patch], 
                        # where +1 refers to the first row padded with zeros, so all data in edge_raw_features[0,] should be ignored,
                        # so we +1 to train_data_indices below
                        # train_loss here is computed as the L1 Loss (MAE) for the entire training window and returns a single value
                        train_loss = torch.nn.functional.l1_loss(predicted_edge_feature, torch.tensor(edge_raw_features[train_data_indices + 1], dtype=torch.float32, device = args.device))

                        # Apply loss regularisations (L1 / L2 / Elastic Net), if applicable
                        # Apply L1 Regularisation (Lasso), if applicable
                        if args.l1_regularisation_lambda != 0 and args.l2_regularisation_lambda == 0:
                            log_patch.info(f"[TRAINING] Conducting L1-Regularisation (Lasso) on the Loss Function")
                            l1_norm = sum(p.abs().sum() for p in model.parameters() if p.requires_grad)
                            train_loss += args.l1_regularisation_lambda * l1_norm
                            log_patch.info(f"[TRAINING] L1-REGULARISED LOSS (Lambda = {args.l1_regularisation_lambda}): {train_loss}\n")

                        # Apply L2 Regularisation (Ridge), if applicable
                        elif args.l1_regularisation_lambda == 0 and args.l2_regularisation_lambda != 0:
                            log_patch.info(f"[TRAINING] Conducting L2-Regularisation (Ridge) on the Loss Function")
                            l2_norm = sum(p.pow(2.0).sum() for p in model.parameters() if p.requires_grad)
                            train_loss += args.l2_regularisation_lambda * l2_norm
                            log_patch.info(f"[TRAINING] L2-REGULARISED LOSS (Lambda = {args.l2_regularisation_lambda}): {train_loss}\n")
                            
                        # Apply Elastic Net Regularisation (L1 + L2 Regularisation), if applicable
                        elif args.l1_regularisation_lambda != 0 and args.l2_regularisation_lambda != 0:
                            log_patch.info(f"[TRAINING] Conducting Elastic Net Regularisation (L1 & L2 regularisation) on the Loss Function")
                            l1_norm = sum(p.abs().sum() for p in model.parameters() if p.requires_grad)
                            l2_norm = sum(p.pow(2.0).sum() for p in model.parameters() if p.requires_grad)
                            train_loss += (args.l1_regularisation_lambda * l1_norm.item()) + (args.l2_regularisation_lambda * l2_norm.item())
                            log_patch.info(f"[TRAINING] ELASTIC-NET-REGULARISED LOSS (L1-Lambda = {args.l1_regularisation_lambda}, L2-Lambda = {args.l2_regularisation_lambda}): {train_loss}\n")
                            
                        # If no regularisation takes place, retain the original loss function
                        else:
                            log_patch.info(f"[TRAINING] No regularisation was applied and the original loss function remains unchanged.\n")
                            
                        # UNCOMMENT FOR DEBUGGING: to visualize train results
                        # if epoch in [3,4,5,6,7,29] and train_window_idx[0] == 0:
                        if epoch in [19]:
                            os.makedirs(patch_dir + "/image_result", exist_ok=True)
                            for nbei, bei in enumerate(batch_edge_ids):
                                # Get dates from node and edge mappings
                                src_node_id, dst_node_id = edge_node_pairs[bei-1]
                                src_date = node_mapping.loc[node_mapping['nodeID'] == src_node_id, 'date'].values[0]
                                dst_date = node_mapping.loc[node_mapping['nodeID'] == dst_node_id, 'date'].values[0]
                                src_date_str = pd.to_datetime(src_date).strftime('%Y%m%d')
                                dst_date_str = pd.to_datetime(dst_date).strftime('%Y%m%d')
                                save_name = os.path.join(patch_dir, f"image_result/train_epoch-{epoch+1:04d}_src-{src_node_id:04d}_dst-{dst_node_id:04d}_edge-{bei:04d}_{src_date_str}-{dst_date_str}")
                                # Save results
                                gt_flat = edge_raw_features[bei,]
                                gt_img = gt_flat.reshape((args.patch_length,args.patch_length))
                                pred_flat = predicted_edge_feature[nbei,].detach().cpu().numpy()
                                pred_img = pred_flat.reshape((args.patch_length,args.patch_length))
                                np.save(save_name + "_gt.npy", gt_img)
                                np.save(save_name + "_pred.npy", pred_img)
                                # Visualise and plot distributions of results
                                evaluate_image_link_prediction_visualiser(log_patch,gt_img,pred_img,save_name+"_visual.png")
                                evaluate_image_link_prediction_plot_distributions(log_patch,gt_flat,pred_flat,gt_flat-pred_flat,np.abs(gt_flat-pred_flat),save_name+"_distributions.png")
                        
                        # Append losses to the storage arrays
                        train_losses.append(train_loss.item())
                        train_metrics.append({'Training MAE loss': train_loss.item()})

                        # Logging training loss for the specified training window
                        log_patch.info(f"TRAINING LOSS FOR PATCH {patch_id + 1:05d}, RUN {run + 1}, EPOCH {epoch + 1}, TRAINING WINDOW {train_window_idx[0] + 1}: {train_loss.item()}\n")

                        optimizer.zero_grad()   # Set the gradients of parameters to zero before backpropagation
                        train_loss.backward()   # Backpropagation
                        optimizer.step()        # Update model weights (W) and biases (b)
                        sys.stdout.flush()

                        # train_data_unique_dst_node_ids_tqdm.set_description(f'Epoch: {epoch + 1}, train for the {train_window_idx[0] + 1}-th batch, train loss: {loss.item()}')

                        if args.model_name in ['JODIE', 'DyRep', 'TGN']:
                            # Detach the memories and raw messages of nodes in the memory bank after each batch, so we don't back propagate to the start of time
                            model[0].memory_bank.detach_memory_bank()

                        # Break out of the training loop when the last destination node of the training dataset is reached
                        if train_window_dst_node_end < train_data_dst_node_end:
                            log_patch.info(f"Moving on to the next training window...\n")
                        elif train_window_dst_node_end == train_data_dst_node_end:
                            log_patch.info(f"TRAINING involving {len(train_data.edge_ids)} edges across {train_window_idx[0] + 1} temporal windows has ended. Proceeding to VALIDATION involving {len(val_data.edge_ids)} edges...\n")
                            break

                    # If the training window does not meet the 2 criteria for training, skip training for this window and move on to the next window
                    elif (train_window_num_total_edges < args.min_num_total_edges_valid_training_window) or \
                        (train_window_num_new_edges < args.min_num_new_edges_valid_training_window):

                        log_patch.warning(f"WARNING: Skipping training for WINDOW {train_window_idx[0] + 1} as there are too few total and/or new edges. | Total Edges: {train_window_num_total_edges} ({train_window_num_new_edges} new)")
                        log_patch.warning(f"WARNING: Moving on to the next training window...\n")

                        continue  # Proceed to next training window
                
                ########## VALIDATION ##########
                
                log_patch.info(f"********** PATCH {patch_id + 1:05d} | RUN {run + 1} | EPOCH {epoch + 1} | VALIDATION **********")
                visualize_flag = False
                distributions_all_flag = False
                distributions_flag = False
                
                # UNCOMMENT FOR DEBUGGING: Visualise and plot distributions of the validation results
                if epoch in [19]:
                    visualize_flag = True
                    distributions_all_flag = True
                    distributions_flag = True
                    
                # Validate as per normal without visualizing
                val_metrics = evaluate_image_link_prediction_without_dataloader(logger=log_patch,
                                                                            model_name=args.model_name,
                                                                            model=model,
                                                                            neighbor_sampler=full_neighbor_sampler,
                                                                            edge_ids_array=val_data.edge_ids,
                                                                            evaluate_data=val_data,
                                                                            eval_stage='val',
                                                                            eval_metric_name=eval_metric_name,
                                                                            evaluator=evaluator,
                                                                            evaluate_metrics=val_metrics,
                                                                            edge_raw_features=edge_raw_features,
                                                                            edge_node_pairs=edge_node_pairs,
                                                                            node_mapping=node_mapping,
                                                                            num_neighbors=args.num_neighbors,
                                                                            time_gap=args.time_gap,
                                                                            l1_regularisation_lambda=args.l1_regularisation_lambda,
                                                                            l2_regularisation_lambda=args.l2_regularisation_lambda,
                                                                            visualize=visualize_flag,
                                                                            distributions_all=distributions_all_flag,
                                                                            distributions=distributions_flag,
                                                                            epoch=epoch+1)
                    
                if args.model_name in ['JODIE', 'DyRep', 'TGN']:
                    # Backup memory bank after validating so it can be used for testing nodes (since test edges are strictly later in time than validation edges)
                    val_backup_memory_bank = model[0].memory_bank.backup_memory_bank()
                
                # Logging validation loss for the specified epoch
                log_patch.info(f"VALIDATION LOSS FOR PATCH {patch_id + 1:05d}, RUN {run + 1}, EPOCH {epoch + 1}: {val_metrics[-1]['val MAE loss']}\n\n")
                
                # Logging average training and validation losses
                log_patch.info(f'Number of epochs: {epoch + 1}, learning rate: {optimizer.param_groups[0]["lr"]}, mean train loss: {np.mean(train_losses):.4f}')
                for metric_name in train_metrics[0].keys():
                    log_patch.info(f'train mean {metric_name}, {np.mean([train_metric[metric_name] for train_metric in train_metrics]):.4f}')
                for metric_name in val_metrics[0].keys():
                    log_patch.info(f'validate mean {metric_name}, {np.mean([val_metric[metric_name] for val_metric in val_metrics]):.4f}')

                # Select the best model based on all types of validate metrics
                # For L1 MAE loss, lower is better, so use 'False' in val_metric_indicator for early stopping
                val_metric_indicator = []
                for metric_name in val_metrics[0].keys():
                    val_metric_indicator.append((metric_name, np.mean([val_metric[metric_name] for val_metric in val_metrics]), False))
                early_stop = early_stopping.step(val_metric_indicator, model)

                if early_stop:
                    log_patch.info(f"Early stopping at EPOCH {epoch + 1} for PATCH {patch_id + 1:05d} | RUN {run + 1}.\n")
                    break
            
            # Load the best model
            early_stopping.load_checkpoint(model)

            ######### TESTING #########

            log_patch.info(f"VALIDATION involving {len(val_data.edge_ids)} edges has ended. Proceeding to TESTING involving {len(test_data.edge_ids)} edges...\n")
            
            # Evaluate the best model
            log_patch.info(f'Get final performance on dataset: {args.dataset_name}...\n')

            log_patch.info(f"********** PATCH {patch_id + 1:05d} | RUN {run + 1} | TESTING **********")
            test_metrics = evaluate_image_link_prediction_without_dataloader(logger=log_patch,
                                                                            model_name=args.model_name,
                                                                            model=model,
                                                                            neighbor_sampler=full_neighbor_sampler,
                                                                            edge_ids_array=test_data.edge_ids,
                                                                            evaluate_data=test_data,
                                                                            eval_stage='test_end',
                                                                            eval_metric_name=eval_metric_name,
                                                                            evaluator=evaluator,
                                                                            evaluate_metrics=test_metrics,
                                                                            edge_raw_features=edge_raw_features,
                                                                            edge_node_pairs=edge_node_pairs,
                                                                            node_mapping=node_mapping,
                                                                            num_neighbors=args.num_neighbors,
                                                                            time_gap=args.time_gap,
                                                                            l1_regularisation_lambda=args.l1_regularisation_lambda,
                                                                            l2_regularisation_lambda=args.l2_regularisation_lambda,
                                                                            visualize=True,
                                                                            distributions_all=True,
                                                                            distributions=True,
                                                                            epoch=epoch+1)

            # exit(0)
            # store the evaluation metrics at the current run
            val_metric_dict, test_metric_dict = {}, {}

            if args.model_name not in ['JODIE', 'DyRep', 'TGN']:
                for metric_name in val_metrics[0].keys():
                    average_val_metric = np.mean([val_metric[metric_name] for val_metric in val_metrics])
                    log_patch.info(f'validate {metric_name}, {average_val_metric:.4f}')
                    val_metric_dict[metric_name] = average_val_metric

            for metric_name in test_metrics[0].keys():
                average_test_metric = np.mean([test_metric[metric_name] for test_metric in test_metrics])
                log_patch.info(f'test {metric_name}, {average_test_metric:.4f}')
                test_metric_dict[metric_name] = average_test_metric

            single_run_time = time.time() - run_start_time
            log_patch.info(f'PATCH {patch_id + 1:05d} RUN {run + 1} takes {single_run_time:.3f} seconds.')

            if args.model_name not in ['JODIE', 'DyRep', 'TGN']:
                val_metric_all_runs.append(val_metric_dict)
            test_metric_all_runs.append(test_metric_dict)

            # Save model result
            if args.model_name not in ['JODIE', 'DyRep', 'TGN']:
                result_json = {
                    "validate metrics": {metric_name: f'{val_metric_dict[metric_name]:.4f}' for metric_name in val_metric_dict},
                    "test metrics": {metric_name: f'{test_metric_dict[metric_name]:.4f}' for metric_name in test_metric_dict}
                }
            else:
                result_json = {
                    "test metrics": {metric_name: f'{test_metric_dict[metric_name]:.4f}' for metric_name in test_metric_dict}
                }
            result_json = json.dumps(result_json, indent=4)

            save_result_folder = f"./saved_results/{args.model_name}/{args.dataset_name}"
            os.makedirs(save_result_folder, exist_ok=True)
            save_result_path = os.path.join(save_result_folder, f"{args.save_model_name}.json")

            with open(save_result_path, 'w') as file:
                file.write(result_json)

        # Store the average metrics at the log of the last run
        log_patch.info(f'Metrics over {args.num_runs} RUNS for PATCH {patch_id + 1:05d}:')

        if args.model_name not in ['JODIE', 'DyRep', 'TGN']:
            for metric_name in val_metric_all_runs[0].keys():
                log_patch.info(f'validate {metric_name}, {[val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs]}')
                log_patch.info(f'AVERAGE validate {metric_name}, {np.mean([val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs]):.4f} '
                            f'± {np.std([val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs], ddof=1):.4f}')

        for metric_name in test_metric_all_runs[0].keys():
            log_patch.info(f'test {metric_name}, {[test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]}')
            log_patch.info(f'AVERAGE test {metric_name}, {np.mean([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]):.4f} '
                        f'± {np.std([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs], ddof=1):.4f}')
        
        # Stitch new patch to the merged image for each edge predicted using mean/min/max
        if args.stitch_method == 'mean':
            pred_dir = os.path.join(patch_dir, 'image_result')
            for pred_file in sorted([f for f in os.listdir(pred_dir) if f.startswith("test_end_") and f.endswith("_pred.npy")]): 
                stitchPatchMean(log_patch, merged_dir, pred_file, patch_bbox, args.patch_length)
            # Check if the merged image was created successfully
            if (merged_img_count := sum("pred_merged_img.npy" in f for f in os.listdir(merged_dir))):
                log_main.info(f'Stitched all patches using the {args.stitch_method.upper()} method to create {merged_img_count} merged image(s) in {merged_dir}\n\n')
            else:
                log_main.warning(f'WARNING: No merged image created in {merged_dir} using the {args.stitch_method.upper()} method. Please check the patch result directories.\n\n')

        os.chdir(work_dir)

    # Stitch all patches to create merged image for each edge predicted using median 
    if args.stitch_method == 'median':
        patch_dirs = sorted([d for d in os.listdir(".") if os.path.isdir(d) and re.match(r"patch_\d{4}$", d)])  # Get list of all patch folders
        for pred_file in sorted([f for f in os.listdir(f"{patch_dirs[0]}/image_result") if f.startswith("train_") and f.endswith("_pred.npy")]): # Get unique train _pred.npy files from the first patch folder as a reference
            stitchPatchMedian(log_main, merged_dir, patchList, pred_file, x, y, args.stitch_chunk_size)
        for pred_file in sorted([f for f in os.listdir(f"{patch_dirs[0]}/image_result") if f.startswith("val_") and f.endswith("_pred.npy")]): # Get unique val _pred.npy files from the first patch folder as a reference
            stitchPatchMedian(log_main, merged_dir, patchList, pred_file, x, y, args.stitch_chunk_size)
        for pred_file in sorted([f for f in os.listdir(f"{patch_dirs[0]}/image_result") if f.startswith("test_end_") and f.endswith("_pred.npy")]): # Get unique test _pred.npy files from the first patch folder as a reference
            stitchPatchMedian(log_main, merged_dir, patchList, pred_file, x, y, args.stitch_chunk_size)
        # Check if the merged image was created successfully
        if (merged_img_count := sum("pred_merged_img.npy" in f for f in os.listdir(merged_dir))):
            log_main.info(f'Stitched all patches using the {args.stitch_method.upper()} method to create {merged_img_count} merged image(s) in {merged_dir}\n\n')
        else:
            log_main.warning(f'WARNING: No merged image created in {merged_dir} using the {args.stitch_method.upper()} method. Please check the patch result directories.\n\n')
    
    # End time for the main script
    end_time_main = time.time()

    # Elapsed time for the main script
    elapsed_time_main = int(round((end_time_main - start_time_main), 0))

    # Format elapsed time to days, hours, minutes, seconds
    elapsed_days, elapsed_days_remainder = divmod(int(elapsed_time_main), 86400)
    elapsed_hours, elapsed_hours_remainder = divmod(elapsed_days_remainder, 3600)
    elapsed_minutes, elapsed_seconds = divmod(elapsed_hours_remainder, 60)

    log_main.info(f"\nTOTAL ELAPSED TIME: {elapsed_days} Days {elapsed_hours} Hours {elapsed_minutes} Minutes {elapsed_seconds} Seconds ({elapsed_time_main} Seconds)\n")

    log_main.info(f"\n\n\n~~~~~ The script '{sys.argv[0]}' has been COMPLETED SUCCESSFULLY! ~~~~~\n\n\n")

    sys.exit()
