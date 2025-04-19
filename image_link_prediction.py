import logging
import time
import sys
import os
from tqdm import tqdm
import numpy as np
import warnings
import shutil
import json
import math
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
from evaluate_models_utils import evaluate_model_link_prediction,evaluate_image_link_prediction, evaluate_image_link_prediction_without_dataloader
from utils.DataLoader import get_idx_data_loader, get_link_prediction_tgb_data, get_link_prediction_image_data, \
get_link_prediction_image_data_split_by_nodes, load_edge_node_pairs, get_sliding_window_data_loader
from utils.EarlyStopping import EarlyStopping
from utils.load_configs import get_link_prediction_args


if __name__ == "__main__":

    # Ignore warnings
    warnings.filterwarnings('ignore')

    # Input data
    data_dir = "/aria/zulun_sharing/TEST_KZ_TGNN_sharing_20250214/merged/cors_ALL" # directory containing original image data

    # get arguments from load_configs.py
    args = get_link_prediction_args(is_evaluation=False)
    
    # get list bounding boxes for each patch
    sys.setrecursionlimit(100000)  # Increase recursion limit to avoid crashes
    sub_dir = [d for d in os.listdir(data_dir) if d.startswith("cor_") and os.path.isdir(os.path.join(data_dir, d))]  # For getting image dimensions
    y, x = sizeFromXml(os.path.join(data_dir, sub_dir[0], "b01_16r4alks.cor"))   # Get size of full image
    patchList = computePatches(x, y, args.patch_length, args.patch_overlap)   # Get list of bounding boxes for each patch

    # initialize merged image directory
    work_dir = os.getcwd()
    merged_dir = os.path.join(work_dir,'merged')
    if not os.path.exists(merged_dir):
            os.makedirs(merged_dir)

    ######### PATCH LOOP #########
    # iterate over each image patch
    for patch_id, patch_bbox in enumerate(patchList):

        print (f"\n********** PATCH {patch_id + 1:04d} **********\n")

        # create sub-directory for the patch
        patch_dir = os.path.join(work_dir,'patch_' + f"{patch_id+1:04d}")
        if not os.path.exists(patch_dir):
            os.makedirs(patch_dir)

        # process dataset for the patch
        prepareData(data_dir, patch_dir, patch_bbox, base_filename='b01_16r4alks')

        # go into the patch directory
        os.chdir(patch_dir)

        # get data for training, validation and testing
        node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data, eval_neg_edge_sampler, eval_metric_name = \
            get_link_prediction_image_data_split_by_nodes(dataset_name=args.dataset_name)

        print("Dimension of node_raw_features:", node_raw_features.shape, "\n")
        print("Dimension of edge_raw_features:", edge_raw_features.shape, "\n")

        # initialize training neighbor sampler to retrieve temporal graph
        train_neighbor_sampler = get_neighbor_sampler(data=train_data, sample_neighbor_strategy=args.sample_neighbor_strategy,
                                                    time_scaling_factor=args.time_scaling_factor, seed=0)

        # initialize validation and test neighbor sampler to retrieve temporal graph
        full_neighbor_sampler = get_neighbor_sampler(data=full_data, sample_neighbor_strategy=args.sample_neighbor_strategy,
                                                    time_scaling_factor=args.time_scaling_factor, seed=1)

        # initialize train negative sampler
        # train_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=train_data.src_node_ids, dst_node_ids=train_data.dst_node_ids)

        # Print a summary of the training / validation / test nodes & edges
        print(f"train_data.src_node_ids ({len(train_data.src_node_ids)} edges):\n", train_data.src_node_ids, "\n")
        print(f"train_data.dst_node_ids ({len(train_data.dst_node_ids)} edges):\n", train_data.dst_node_ids, "\n")
        print(f"train_data.edge_ids ({len(train_data.edge_ids)} edges):\n", train_data.edge_ids, "\n")

        print(f"val_data.src_node_ids ({len(val_data.src_node_ids)} edges):\n", val_data.src_node_ids, "\n")
        print(f"val_data.dst_node_ids ({len(val_data.dst_node_ids)} edges):\n", val_data.dst_node_ids, "\n")
        print(f"val_data.edge_ids ({len(val_data.edge_ids)} edges):\n", val_data.edge_ids, "\n")
        
        print(f"test_data.src_node_ids ({len(test_data.src_node_ids)} edges):\n", test_data.src_node_ids, "\n")
        print(f"test_data.dst_node_ids ({len(test_data.dst_node_ids)} edges):\n", test_data.dst_node_ids, "\n")
        print(f"test_data.edge_ids ({len(test_data.edge_ids)} edges):\n", test_data.edge_ids, "\n")

        # Compute the desired batch size (number of edges)
        # By default, 1 date is paired with the next 10 dates, hence 11 dates in total for one temporal window.
        batch_size = math.comb(args.temporal_window_width + 1, 2)   # Max number of edges in one temporal window.

        # training nodes
        train_data_src_node_start = min(train_data.src_node_ids)
        train_data_src_node_end = max(train_data.src_node_ids)
        train_data_dst_node_start = min(train_data.dst_node_ids)
        train_data_dst_node_end = max(train_data.dst_node_ids)
            
        # validation nodes
        val_data_src_node_start = min(val_data.src_node_ids)
        val_data_src_node_end = max(val_data.src_node_ids)
        val_data_dst_node_start = min(val_data.dst_node_ids)
        val_data_dst_node_end = max(val_data.dst_node_ids)

        # testing nodes
        test_data_src_node_start = min(test_data.src_node_ids)
        test_data_src_node_end = max(test_data.src_node_ids)
        test_data_dst_node_start = min(test_data.dst_node_ids)
        test_data_dst_node_end = max(test_data.dst_node_ids)

        # Load the edge node pairs from the file
        edge_node_pairs = load_edge_node_pairs('./image_data/edge_node_pairs.txt')

        # Set up the patch logging directory
        patch_log_dir = f"./logs/{args.model_name}/{args.dataset_name}/patch_{patch_id+1:04d}"
        os.makedirs(patch_log_dir, exist_ok=True)

        # Storage arrays for metrics
        val_metric_all_runs, test_metric_all_runs = [], []

        ######### RUN LOOP #########
        for run in range(args.num_runs):

            # Set random seed. The seed is set to be the same as the run number.
            set_random_seed(seed=run)
            args.seed = run
            args.save_model_name = f'{args.model_name}_seed{args.seed}'

            # Set up logger for this run inside the patch-specific directory
            run_log_dir = os.path.join(patch_log_dir, args.save_model_name)
            os.makedirs(run_log_dir, exist_ok=True)
            logger = logging.getLogger(f"{args.model_name}_patch_{patch_id+1:04d}_run_{run+1}")
            logger.setLevel(logging.DEBUG)
            logger.handlers = []  # Clear any existing handlers
            # create file handler that logs DEBUG and higher level messages
            fh = logging.FileHandler(os.path.join(run_log_dir, f"{time.strftime('%Y%m%d_%H%M%S')}.log"))
            # fh = logging.FileHandler(os.path.join(run_log_dir, f"{str(time.time())}.log"))    # Original log nomenclature
            fh.setLevel(logging.DEBUG)
            # create console handler with a higher log level (WARNING and above)
            ch = logging.StreamHandler()
            ch.setLevel(logging.WARNING)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            fh.setFormatter(formatter)
            ch.setFormatter(formatter)
            logger.addHandler(fh)
            logger.addHandler(ch)

            # Start the timer for the run
            run_start_time = time.time()

            logger.info(f"********** PATCH {patch_id + 1:04d} | RUN {run + 1} **********\n")

            logger.info(f'CONFIGURATION: {args}\n')
            
            # create model
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
            link_predictor = MergeLayer(input_dim1=args.output_dim, input_dim2=args.output_dim, hidden_dim=args.output_dim, output_dim=edge_raw_features.shape[1])
            
            # Model consists of two parts: dynamic backbone "model[0]" and link predictor "model[1]"
            model = nn.Sequential(dynamic_backbone, link_predictor)
            logger.info(f'model -> {model}')
            logger.info(f'model name: {args.model_name}, #parameters: {get_parameter_sizes(model) * 4} B, '
                        f'{get_parameter_sizes(model) * 4 / 1024} KB, {get_parameter_sizes(model) * 4 / 1024 / 1024} MB.')

            optimizer = create_optimizer(model=model, optimizer_name=args.optimizer, learning_rate=args.learning_rate, weight_decay=args.weight_decay)

            model = convert_to_gpu(model, device=args.device)

            save_model_folder = f"./saved_models/{args.model_name}/{args.dataset_name}/{args.save_model_name}/"
            shutil.rmtree(save_model_folder, ignore_errors=True)
            os.makedirs(save_model_folder, exist_ok=True)

            early_stopping = EarlyStopping(patience=args.patience, save_model_folder=save_model_folder,
                                        save_model_name=args.save_model_name, logger=logger, model_name=args.model_name)

            loss_func = nn.BCELoss()    # BCE loss is not used as of the moment. Using L1 loss (MAE) instead.  
            evaluator = Evaluator(name=args.dataset_name)

            ######### EPOCH LOOP #########
            for epoch in range(args.num_epochs):

                print(f"TRAINING with {args.num_epochs} epochs...")

                print(f"\n********** PATCH {patch_id + 1:04d} | RUN {run + 1} | EPOCH {epoch + 1} **********\n")
                logger.info(f'********** PATCH {patch_id + 1:04d} | RUN {run + 1} | EPOCH {epoch + 1} **********\n')

                # set model to training mode
                model.train()
                
                if args.model_name in ['DyRep', 'TGAT', 'TGN', 'CAWN', 'TCL', 'GraphMixer', 'DyGFormer']:
                    # training, only use training graph
                    model[0].set_neighbor_sampler(train_neighbor_sampler)
                if args.model_name in ['JODIE', 'DyRep', 'TGN']:
                    # reinitialize memory of memory-based models at the start of each epoch
                    model[0].memory_bank.__init_memory_bank__()

                # Storage arrays for train losses and metrics
                train_losses, train_metrics = [], []        
                
                # Progress bar
                # train_data_unique_dst_node_ids_tqdm = tqdm(np.unique(train_data.dst_node_ids), ncols=120, dynamic_ncols=True)

                # for train_window_idx in enumerate (train_data_unique_dst_node_ids_tqdm):
                for train_window_idx in enumerate(range(1,train_data_dst_node_end+1)):
                    
                    print(f"\n********** PATCH {patch_id + 1:04d} | RUN {run + 1} | EPOCH {epoch + 1} | TRAINING WINDOW {train_window_idx[0] + 1} **********\n")
                    logger.info(f"********** PATCH {patch_id + 1:04d} | RUN {run + 1} | EPOCH {epoch + 1} | TRAINING WINDOW {train_window_idx[0] + 1} **********")

                    # Get the start and end nodes of the temporal window for each iteration
                    # Attempt subsetting by first assuming that there is no irregularity in node & edge features (e.g. missing dates) within the training data
                    train_window_src_node_start = train_data_src_node_start + train_window_idx[0]
                    train_window_src_node_end = train_data_src_node_start + train_window_idx[0] + (args.temporal_window_width - 1)    # -1 to account for the first date
                    train_window_dst_node_start = train_data_dst_node_start + train_window_idx[0]
                    train_window_dst_node_end = train_data_dst_node_start + train_window_idx[0] + (args.temporal_window_width - 1)    # -1 to account for the first date

                    # Get the training data indices for the current temporal window (NOT the same as edge_ids!)
                    train_data_indices = np.where(((train_data.src_node_ids >= train_window_src_node_start) & 
                                                  (train_data.src_node_ids <= train_window_src_node_end)) & 
                                                  ((train_data.dst_node_ids >= train_window_dst_node_start) & 
                                                  (train_data.dst_node_ids <= train_window_dst_node_end)))[0]
                    
                    # Check for irregularities in node & edge features (e.g. missing dates) within the training data by comparing to "batch_size"
                    # batch_size = number of edges in the temporal window (total_num_nodes choose 2)
                    if len(train_data_indices) != batch_size:
                        
                        # # [TO CHANGE] Subset the training data based on modified start and end node indices
                        # train_window_src_node_start = train_data_src_node_start + train_window_idx[0]
                        # train_window_src_node_end = train_data_src_node_start + train_window_idx[0] + (args.temporal_window_width - 1)    # -1 to account for the first date
                        # train_window_dst_node_start = train_data_dst_node_start + train_window_idx[0]
                        # train_window_dst_node_end = train_data_dst_node_start + train_window_idx[0] + (args.temporal_window_width - 1)    # -1 to account for the first date

                        # # Get the training data indices for the current temporal window
                        # train_data_indices = np.where(((train_data.src_node_ids >= train_window_src_node_start) & 
                        #                                 (train_data.src_node_ids <= train_window_src_node_end)) & 
                        #                                 ((train_data.dst_node_ids >= train_window_dst_node_start) & 
                        #                                 (train_data.dst_node_ids <= train_window_dst_node_end)))[0]

                        pass    # [PLACEHOLDER pass COMMAND TO BE REMOVED IN FUTURE]

                    # Get the batch of training data for specified temporal window only
                    batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids = \
                        train_data.src_node_ids[train_data_indices], train_data.dst_node_ids[train_data_indices], \
                        train_data.node_interact_times[train_data_indices], train_data.edge_ids[train_data_indices]    
                    
                    # Logging information about the specified temporal window
                    logger.info(f"train_window_idx: {train_window_idx[0] + 1}")
                    logger.info(f"train_window_src_node_start: {train_window_src_node_start}")
                    logger.info(f"train_window_src_node_end: {train_window_src_node_end}")
                    logger.info(f"train_window_dst_node_start: {train_window_dst_node_start}")
                    logger.info(f"train_window_dst_node_end: {train_window_dst_node_end}")
                    logger.info(f"train_data_indices: {train_data_indices}")
                    logger.info(f"batch_src_node_ids: {batch_src_node_ids}")
                    logger.info(f"batch_dst_node_ids: {batch_dst_node_ids}")
                    logger.info(f"batch_node_interact_times: {batch_node_interact_times}")
                    logger.info(f"batch_edge_ids ({len(batch_edge_ids)} edges): {batch_edge_ids}")

                    print(f"train_window_src_node_start: {train_window_src_node_start}")
                    print(f"train_window_src_node_end: {train_window_src_node_end}")
                    print(f"train_window_dst_node_start: {train_window_dst_node_start}")
                    print(f"train_window_dst_node_end: {train_window_dst_node_end}")
                    print(f"batch_edge_ids ({len(batch_edge_ids)} edges): \n{batch_edge_ids}\n")

                    # we need to compute for positive and negative edges respectively, because the new sampling strategy (for evaluation) allows the negative source nodes to be
                    # different from the source nodes, this is different from previous works that just replace destination nodes with negative destination nodes
                    if args.model_name in ['TGAT', 'CAWN', 'TCL']:
                        # get temporal embedding of source and destination nodes
                        # two Tensors, with shape (batch_size, output_dim)
                        batch_src_node_embeddings, batch_dst_node_embeddings = \
                            model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                            dst_node_ids=batch_dst_node_ids,
                                                                            node_interact_times=batch_node_interact_times,
                                                                            num_neighbors=args.num_neighbors)

                        # get temporal embedding of negative source and negative destination nodes
                        # two Tensors, with shape (batch_size, output_dim)
                        # batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                        #     model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                        #                                                       dst_node_ids=batch_neg_dst_node_ids,
                        #                                                       node_interact_times=batch_node_interact_times,
                        #                                                       num_neighbors=args.num_neighbors)
                    elif args.model_name in ['JODIE', 'DyRep', 'TGN']:
                        # note that negative nodes do not change the memories while the positive nodes change the memories,
                        # we need to first compute the embeddings of negative nodes for memory-based models
                        # get temporal embedding of negative source and negative destination nodes
                        # two Tensors, with shape (batch_size, output_dim)
                        # batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                        #     model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                        #                                                       dst_node_ids=batch_neg_dst_node_ids,
                        #                                                       node_interact_times=batch_node_interact_times,
                        #                                                       edge_ids=None,
                        #                                                       edges_are_positive=False,
                        #                                                       num_neighbors=args.num_neighbors)

                        # get temporal embedding of source and destination nodes
                        # two Tensors, with shape (batch_size, output_dim)
                        batch_src_node_embeddings, batch_dst_node_embeddings = \
                            model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                            dst_node_ids=batch_dst_node_ids,
                                                                            node_interact_times=batch_node_interact_times,
                                                                            edge_ids=batch_edge_ids,
                                                                            edges_are_positive=True,
                                                                            num_neighbors=args.num_neighbors)
                    elif args.model_name in ['GraphMixer']:
                        # get temporal embedding of source and destination nodes
                        # two Tensors, with shape (batch_size, output_dim)
                        batch_src_node_embeddings, batch_dst_node_embeddings = \
                            model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                            dst_node_ids=batch_dst_node_ids,
                                                                            node_interact_times=batch_node_interact_times,
                                                                            num_neighbors=args.num_neighbors,
                                                                            time_gap=args.time_gap)

                        # get temporal embedding of negative source and negative destination nodes
                        # # two Tensors, with shape (batch_size, output_dim)
                        # batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                        #     model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                        #                                                       dst_node_ids=batch_neg_dst_node_ids,
                        #                                                       node_interact_times=batch_node_interact_times,
                        #                                                       num_neighbors=args.num_neighbors,
                        #                                                       time_gap=args.time_gap)
                    elif args.model_name in ['DyGFormer']:
                        # get temporal embedding of source and destination nodes
                        # two Tensors, with shape (batch_size, output_dim)
                        batch_src_node_embeddings, batch_dst_node_embeddings = \
                            model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                            dst_node_ids=batch_dst_node_ids,
                                                                            node_interact_times=batch_node_interact_times)

                        # get temporal embedding of negative source and negative destination nodes
                        # two Tensors, with shape (batch_size, output_dim)
                        # batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                        #     model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                        #                                                       dst_node_ids=batch_neg_dst_node_ids,
                        #                                                       node_interact_times=batch_node_interact_times)
                    else:
                        raise ValueError(f"Wrong value for model_name {args.model_name}!")
                    # get positive and negative probabilities, shape (batch_size, )
                    # positive_probabilities = model[1](input_1=batch_src_node_embeddings, input_2=batch_dst_node_embeddings).squeeze(dim=-1).sigmoid()
                    # negative_probabilities = model[1](input_1=batch_neg_src_node_embeddings, input_2=batch_neg_dst_node_embeddings).squeeze(dim=-1).sigmoid()

                    # predicts = torch.cat([positive_probabilities, negative_probabilities], dim=0)
                    # labels = torch.cat([torch.ones_like(positive_probabilities), torch.zeros_like(negative_probabilities)], dim=0)

                    # loss = loss_func(input=predicts, target=labels)
                    
                    # Get the predicted edge feature (without applying sigmoid, as we're doing regression)
                    predicted_edge_feature = model[1](input_1=batch_src_node_embeddings, input_2=batch_dst_node_embeddings)
                    print(f"predicted_edge_feature.shape: {predicted_edge_feature.shape}\n")
                    # print("edge_raw_features[train_data_indices]", edge_raw_features[train_data_indices].shape)

                    # Compute the L1 loss (MAE) between the predicted edge feature and the ground truth edge feature.
                    loss = torch.nn.functional.l1_loss(predicted_edge_feature, torch.tensor(edge_raw_features[train_data_indices], dtype=torch.float32, device = args.device))
                    
                    # Append losses to the storage arrays
                    train_losses.append(loss.item())
                    train_metrics.append({'Training MAE loss': loss.item()})

                    # Logging training loss for the specified training window
                    print (f"TRAINING LOSS FOR PATCH {patch_id + 1:04d}, RUN {run + 1}, EPOCH {epoch + 1}, TRAINING WINDOW {train_window_idx[0] + 1}: {loss.item()}\n")
                    logger.info (f"TRAINING LOSS FOR PATCH {patch_id + 1:04d}, RUN {run + 1}, EPOCH {epoch + 1}, TRAINING WINDOW {train_window_idx[0] + 1}: {loss.item()}\n")

                    optimizer.zero_grad()   # Set the gradients of parameters to zero before backpropagation
                    loss.backward()         # Backpropagation
                    optimizer.step()        # Update model weights (W) and biases (b)
                    sys.stdout.flush()

                    # train_data_unique_dst_node_ids_tqdm.set_description(f'Epoch: {epoch + 1}, train for the {train_window_idx[0] + 1}-th batch, train loss: {loss.item()}')

                    if args.model_name in ['JODIE', 'DyRep', 'TGN']:
                        # detach the memories and raw messages of nodes in the memory bank after each batch, so we don't back propagate to the start of time
                        model[0].memory_bank.detach_memory_bank()

                    # Break out of the training loop when the last destination node of the training dataset is reached.
                    if train_window_dst_node_end < train_data_dst_node_end:
                        print(f"Moving on to the next training window...\n")
                    elif train_window_dst_node_end == train_data_dst_node_end:
                        print(f"TRAINING involving {len(train_data.edge_ids)} edges across {train_window_idx[0] + 1} temporal windows has ended. Proceeding to VALIDATION involving {len(val_data.edge_ids)} edges...\n")
                        logger.info(f"TRAINING involving {len(train_data.edge_ids)} edges across {train_window_idx[0] + 1} temporal windows has ended. Proceeding to VALIDATION involving {len(val_data.edge_ids)} edges...\n")
                        break
                    
                val_metrics = evaluate_image_link_prediction_without_dataloader(model_name=args.model_name,
                                                                                model=model,
                                                                                neighbor_sampler=full_neighbor_sampler,
                                                                                edge_ids_array=val_data.edge_ids,
                                                                                evaluate_data=val_data,
                                                                                eval_stage='val',
                                                                                eval_metric_name=eval_metric_name,
                                                                                evaluator=evaluator,
                                                                                edge_raw_features = edge_raw_features,
                                                                                num_neighbors=args.num_neighbors,
                                                                                time_gap=args.time_gap)

                # val_metrics = evaluate_image_link_prediction(model_name=args.model_name,
                #                                             model=model,
                #                                             neighbor_sampler=full_neighbor_sampler,
                #                                             evaluate_idx_data_loader=val_idx_data_loader,
                #                                             evaluate_data=val_data,
                #                                             eval_stage='val',
                #                                             eval_metric_name=eval_metric_name,
                #                                             evaluator=evaluator,
                #                                             edge_raw_features = edge_raw_features,
                #                                             num_neighbors=args.num_neighbors,
                #                                             time_gap=args.time_gap)

                if args.model_name in ['JODIE', 'DyRep', 'TGN']:
                    # backup memory bank after validating so it can be used for testing nodes (since test edges are strictly later in time than validation edges)
                    val_backup_memory_bank = model[0].memory_bank.backup_memory_bank()

                logger.info(f'Number of epochs: {epoch + 1}, learning rate: {optimizer.param_groups[0]["lr"]}, train loss: {np.mean(train_losses):.4f}')
                for metric_name in train_metrics[0].keys():
                    logger.info(f'train {metric_name}, {np.mean([train_metric[metric_name] for train_metric in train_metrics]):.4f}')
                for metric_name in val_metrics[0].keys():
                    logger.info(f'validate {metric_name}, {np.mean([val_metric[metric_name] for val_metric in val_metrics]):.4f}')

                # # perform testing once after test_interval_epochs
                # if (epoch + 1) % args.test_interval_epochs == 0:
                #     test_metrics = evaluate_image_link_prediction(model_name=args.model_name,
                #                                                 model=model,
                #                                                 neighbor_sampler=full_neighbor_sampler,
                #                                                 evaluate_idx_data_loader=test_idx_data_loader,
                #                                                 evaluate_data=test_data,
                #                                                 eval_stage='test',
                #                                                 eval_metric_name=eval_metric_name,
                #                                                 evaluator=evaluator,
                #                                                 edge_raw_features = edge_raw_features,
                #                                                 num_neighbors=args.num_neighbors,
                #                                                 time_gap=args.time_gap)

                #     if args.model_name in ['JODIE', 'DyRep', 'TGN']:
                #         # reload validation memory bank for testing nodes or saving models
                #         # note that since model treats memory as parameters, we need to reload the memory to val_backup_memory_bank for saving models
                #         model[0].memory_bank.reload_memory_bank(val_backup_memory_bank)

                #     for metric_name in test_metrics[0].keys():
                #         logger.info(f'test {metric_name}, {np.mean([test_metric[metric_name] for test_metric in test_metrics]):.4f}')

                # select the best model based on all the validate metrics
                val_metric_indicator = []
                for metric_name in val_metrics[0].keys():
                    val_metric_indicator.append((metric_name, np.mean([val_metric[metric_name] for val_metric in val_metrics]), True))
                early_stop = early_stopping.step(val_metric_indicator, model)

                if early_stop:
                    break
            
            # load the best model
            early_stopping.load_checkpoint(model)

            # TESTING
            print (f"VALIDATION involving {len(val_data.edge_ids)} edges has ended. Proceeding to TESTING involving {len(test_data.edge_ids)} edges...\n")
            logger.info(f'\nVALIDATION involving {len(val_data.edge_ids)} edges has ended. Proceeding to TESTING involving {len(test_data.edge_ids)} edges...\n')

            # evaluate the best model
            logger.info(f'Get final performance on dataset: {args.dataset_name}...')

            test_metrics = evaluate_image_link_prediction_without_dataloader(model_name=args.model_name,
                                                                            model=model,
                                                                            neighbor_sampler=full_neighbor_sampler,
                                                                            edge_ids_array=test_data.edge_ids,
                                                                            evaluate_data=test_data,
                                                                            eval_stage='test_end',
                                                                            eval_metric_name=eval_metric_name,
                                                                            evaluator=evaluator,
                                                                            edge_raw_features = edge_raw_features,
                                                                            num_neighbors=args.num_neighbors,
                                                                            time_gap=args.time_gap)

            # test_metrics = evaluate_image_link_prediction(model_name=args.model_name,
            #                                             model=model,
            #                                             neighbor_sampler=full_neighbor_sampler,
            #                                             evaluate_idx_data_loader=test_idx_data_loader,
            #                                             evaluate_data=test_data,
            #                                             eval_stage='test_end',
            #                                             eval_metric_name=eval_metric_name,
            #                                             evaluator=evaluator,
            #                                             edge_raw_features = edge_raw_features,
            #                                             num_neighbors=args.num_neighbors,
            #                                             time_gap=args.time_gap)


            # exit(0)
            # store the evaluation metrics at the current run
            val_metric_dict, test_metric_dict = {}, {}

            if args.model_name not in ['JODIE', 'DyRep', 'TGN']:
                for metric_name in val_metrics[0].keys():
                    average_val_metric = np.mean([val_metric[metric_name] for val_metric in val_metrics])
                    logger.info(f'validate {metric_name}, {average_val_metric:.4f}')
                    val_metric_dict[metric_name] = average_val_metric

            for metric_name in test_metrics[0].keys():
                average_test_metric = np.mean([test_metric[metric_name] for test_metric in test_metrics])
                logger.info(f'test {metric_name}, {average_test_metric:.4f}')
                test_metric_dict[metric_name] = average_test_metric

            single_run_time = time.time() - run_start_time
            logger.info(f'PATCH {patch_id + 1:04d} RUN {run + 1} takes {single_run_time:.3f} seconds.')

            if args.model_name not in ['JODIE', 'DyRep', 'TGN']:
                val_metric_all_runs.append(val_metric_dict)
            test_metric_all_runs.append(test_metric_dict)

            # save model result
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

        # store the average metrics at the log of the last run
        logger.info(f'Metrics over {args.num_runs} RUNS for PATCH {patch_id + 1:04d}:')

        if args.model_name not in ['JODIE', 'DyRep', 'TGN']:
            for metric_name in val_metric_all_runs[0].keys():
                logger.info(f'validate {metric_name}, {[val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs]}')
                logger.info(f'AVERAGE validate {metric_name}, {np.mean([val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs]):.4f} '
                            f'± {np.std([val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs], ddof=1):.4f}')

        for metric_name in test_metric_all_runs[0].keys():
            logger.info(f'test {metric_name}, {[test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]}')
            logger.info(f'AVERAGE test {metric_name}, {np.mean([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]):.4f} '
                        f'± {np.std([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs], ddof=1):.4f}')
        
        # stitch new patch to the merged image for each edge predicted using mean/min/max
        if args.stitch_method == 'mean':
            pred_dir = os.path.join(patch_dir, 'image_result')
            for pred_file in sorted([f for f in os.listdir(pred_dir) if f.startswith("pred_") and f.endswith(".npy")]):
                stitchPatchMean(merged_dir, pred_file, patch_bbox, args.patch_length)
            logger.info(f'Stitched patch {patch_id} using the {args.stitch_method} method to create merged image in {merged_dir}')

        os.chdir(work_dir)

    # stitch all patches to create merged image for each edge predicted using median 
    if args.stitch_method == 'median':
        for pred_file in sorted([f for f in os.listdir('patch_0001/image_result') if f.startswith("pred_") and f.endswith(".npy")]):
            stitchPatchMedian(merged_dir, patchList, pred_file, x, y, args.stitch_chunk_size)   
        logger.info(f'Stitched all patches using the {args.stitch_method} method to create merged image in {merged_dir}')
    
    print(f"\n\n\n~~~~~ The script '{sys.argv[0]}' has been COMPLETED SUCCESSFULLY ! ~~~~~\n\n\n")

    sys.exit()
