import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd
import logging
import time
import argparse
import os
import json
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors
import seaborn as sns
from skimage.exposure import match_histograms
from collections import defaultdict
from tgb.linkproppred.negative_sampler import NegativeEdgeSampler
from tgb.linkproppred.evaluate import Evaluator as LinkPredictionEvaluator
from tgb.nodeproppred.evaluate import Evaluator as NodeClassificationEvaluator

from models.EdgeBank import edge_bank_link_prediction
from models.PersistentForecast import PersistentForecast
from models.MovingAverage import MovingAverage
from utils.utils import set_random_seed
from utils.utils import NeighborSampler
from utils.DataLoader import Data
from utils.cor_logit_space import cor_to_logit, logit_to_cor, convert_images_to_reg_cor
from utils.load_configs import get_link_prediction_args


def evaluate_image_link_prediction_without_dataloader(logger: str,
                                                    model_name: str, 
                                                    model: nn.Module, 
                                                    neighbor_sampler: NeighborSampler, 
                                                    edge_ids_array: np.ndarray,
                                                    evaluate_data: Data, 
                                                    eval_stage: str,
                                                    eval_metric_name: str, 
                                                    evaluator: LinkPredictionEvaluator, 
                                                    evaluate_metrics: list,
                                                    edge_raw_features: np.ndarray, 
                                                    edge_node_pairs: list,
                                                    node_mapping: pd.DataFrame,
                                                    num_neighbors: int = 20, 
                                                    time_gap: int = 2000,
                                                    l1_regularisation_lambda: float = 0.0,
                                                    l2_regularisation_lambda: float = 0.0,
                                                    cor_logit: bool = False,
                                                    save_numpy_objects: bool = True,
                                                    visualize: bool = False,
                                                    distributions: bool = False,
                                                    distributions_all: bool = False,
                                                    epoch: int = 1):
    """
    Evaluate models on the link prediction task without requiring a data loader
    Processing all edges in a single batch
    
    :param model_name: str, name of the model
    :param model: nn.Module, the model to be evaluated
    :param neighbor_sampler: NeighborSampler, neighbor sampler
    :param edge_ids_array: np.ndarray, array of edge IDs to evaluate
    :param evaluate_data: Data, data to be evaluated
    :param eval_stage: str, specifies the evaluate stage to generate negative edges, can be 'val' or 'test'
    :param eval_metric_name: str, name of the evaluation metric
    :param evaluator: LinkPredictionEvaluator, link prediction evaluator
    :param edge_raw_features: np.ndarray, raw features of edges
    :param num_neighbors: int, number of neighbors to sample for each node
    :param time_gap: int, time gap for neighbors to compute node features
    :return: list of evaluation metrics
    """
    if model_name in ['DyRep', 'TGAT', 'TGN', 'CAWN', 'TCL', 'GraphMixer', 'DyGFormer']:
        # evaluation phase use all the graph information
        model[0].set_neighbor_sampler(neighbor_sampler)

    model.eval()

    with torch.no_grad():

        # Process all edges at once for VALIDATION and TEST stages
        # print(f"Processing all {len(edge_ids_array)} edges at once for the '{eval_stage}' stage...")
        logger.info(f"Edges used for the '{eval_stage}' stage ({len(edge_ids_array)} edges): \n{edge_ids_array}\n")
        
        # Convert edge IDs to zero-based indices e.g. [ 0  1  2  3  4  5  ...]
        # Used for subsetting the "master dataset" used for VALIDATION and TEST stages
        evaluate_data_indices = edge_ids_array - edge_ids_array[0] 

        logger.info(f"edge_ids_array for the '{eval_stage}' stage: \n{edge_ids_array}\n")
        logger.info(f"Element from edge_ids_array[0] for the '{eval_stage}' stage: {edge_ids_array[0]}\n")
        logger.info(f"evaluate_data_indices (indexing starts from zero) for the '{eval_stage}' stage:\n {evaluate_data_indices}\n") 

        # Extract data for all edges
        batch_src_node_ids = evaluate_data.src_node_ids[evaluate_data_indices]
        batch_dst_node_ids = evaluate_data.dst_node_ids[evaluate_data_indices]
        batch_node_interact_times = evaluate_data.node_interact_times[evaluate_data_indices]
        batch_edge_ids = evaluate_data.edge_ids[evaluate_data_indices]

        logger.info(f"batch_src_node_ids for the '{eval_stage}' stage: \n{batch_src_node_ids}\n")
        logger.info(f"batch_dst_node_ids for the '{eval_stage}' stage: \n{batch_dst_node_ids}\n")
        logger.info(f"batch_node_interact_times for the '{eval_stage}' stage: \n{batch_node_interact_times}\n")
        logger.info(f"batch_edge_ids for the '{eval_stage}' stage: \n{batch_edge_ids}\n")

        # Compute node embeddings based on model type
        if model_name in ['TGAT', 'CAWN', 'TCL']:
            # get temporal embedding of source and destination nodes
            src_node_embeddings, dst_node_embeddings = \
                model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                  dst_node_ids=batch_dst_node_ids,
                                                                  node_interact_times=batch_node_interact_times,
                                                                  num_neighbors=num_neighbors)
        elif model_name in ['JODIE', 'DyRep', 'TGN']:
            # get temporal embedding of source and destination nodes
            src_node_embeddings, dst_node_embeddings = \
                model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                  dst_node_ids=batch_dst_node_ids,
                                                                  node_interact_times=batch_node_interact_times,
                                                                  edge_ids=batch_edge_ids,
                                                                  edges_are_positive=True,
                                                                  num_neighbors=num_neighbors)
        elif model_name in ['GraphMixer']:
            # get temporal embedding of source and destination nodes
            src_node_embeddings, dst_node_embeddings = \
                model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                  dst_node_ids=batch_dst_node_ids,
                                                                  node_interact_times=batch_node_interact_times,
                                                                  num_neighbors=num_neighbors,
                                                                  time_gap=time_gap)
        elif model_name in ['DyGFormer']:
            # get temporal embedding of source and destination nodes
            src_node_embeddings, dst_node_embeddings = \
                model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                  dst_node_ids=batch_dst_node_ids,
                                                                  node_interact_times=batch_node_interact_times)
        else:
            raise ValueError(f"Wrong value for model_name {model_name}!")

        # Predict the positive probabilities
        # UNCOMMENT FOR DEBUGGING: Uncomment the following lines for edge predictions with different modification operations
        # When changing the following lines, we must also change the edge prediction lines in "image_link_prediction.py"
        # (1) predictions without modification operations 
        # (2) predictions with an external sigmoid operation (not recommended)
        # (3) predictions with an external clamp operation (not recommended)
        positive_probabilities = model[1](input_1=src_node_embeddings, input_2=dst_node_embeddings).squeeze(dim=-1).cpu().numpy()
        # positive_probabilities = model[1](input_1=src_node_embeddings, input_2=dst_node_embeddings).squeeze(dim=-1).sigmoid().cpu().numpy()
        # positive_probabilities = model[1](input_1=src_node_embeddings, input_2=dst_node_embeddings).squeeze(dim=-1).clamp(min=0.0, max=1.0).cpu().numpy()

        # Ground-truth edge feature
        gt_edge_feature = edge_raw_features[edge_ids_array]

        # Acquire the residuals_full_object, where Residuals = (Ground Truth - Prediction)
        residuals_full_object = gt_edge_feature - positive_probabilities

        # Compute the ground truth, predictions, and residuals for the equivalent of logit coherences in the normal coherence space
        if cor_logit:
            gt_img_reg_cor, pred_img_reg_cor, residuals_img_reg_cor, abs_residuals_img_reg_cor = convert_images_to_reg_cor(gt_edge_feature, positive_probabilities)
            # gt_regular_cor_equivalent = logit_to_cor(edge_raw_features[edge_ids_array])
            # pred_regular_cor_equivalent = logit_to_cor(positive_probabilities)
            # residuals_regular_cor_equivalent = gt_regular_cor_equivalent - pred_regular_cor_equivalent
            # abs_residuals_regular_cor_equivalent = np.abs(residuals_regular_cor_equivalent)

        # Log the Ground Truth, Prediction, Residuals, and Absolute Residuals
        logger.info(f"[{eval_stage.upper()}] GROUND TRUTH (Dimensions: {gt_edge_feature.shape}): \n{gt_edge_feature}\n")
        if cor_logit:
            logger.info(f"[{eval_stage.upper()}] GROUND TRUTH (Regular Coherence Equivalent) (Dimensions: {gt_img_reg_cor.shape}): \n{gt_img_reg_cor}\n")
        logger.info(f"[{eval_stage.upper()}] PREDICTION (Dimensions: {positive_probabilities.shape}): \n{positive_probabilities}\n")
        if cor_logit:
            logger.info(f"[{eval_stage.upper()}] PREDICTION (Regular Coherence Equivalent) (Dimensions: {pred_img_reg_cor.shape}): \n{pred_img_reg_cor}\n")
        logger.info(f"[{eval_stage.upper()}] (GROUND TRUTH - PREDICTION) (Dimensions: {residuals_full_object.shape}): \n{residuals_full_object}\n")
        if cor_logit:
            logger.info(f"[{eval_stage.upper()}] (GROUND TRUTH - PREDICTION) (Regular Coherence Equivalent) (Dimensions: {residuals_img_reg_cor.shape}): \n{residuals_img_reg_cor}\n")
        logger.info(f"[{eval_stage.upper()}] abs(GROUND TRUTH - PREDICTION) (Dimensions: {np.abs(residuals_full_object).shape}): \n{np.abs(residuals_full_object)}\n")
        if cor_logit:
            logger.info(f"[{eval_stage.upper()}] abs(GROUND TRUTH - PREDICTION) (Regular Coherence Equivalent) (Dimensions: {abs_residuals_img_reg_cor.shape}): \n{abs_residuals_img_reg_cor}\n")

        # # Log the result statistics
        # if cor_logit:
        #     logger.info(f"\n"
        #                 f"[{eval_stage.upper()}] STATISTICS\n"
        #                 f"[ MIN | 1st | 25th | 50th | 75th | 99th | MAX ]\n"
        #                 f"GROUND TRUTH (Logit-Transformed Coherence) Percentiles: \n"
        #                 f"{np.percentile(gt_edge_feature, [0, 1, 25, 50, 75, 99, 100])}\n"
        #                 f"GROUND TRUTH (Raw Coherence) Percentiles: \n"
        #                 f"{np.percentile(gt_img_reg_cor, [0, 1, 25, 50, 75, 99, 100])}\n"
        #                 f"PREDICTION (Logit-Transformed Coherence) Percentiles: \n"
        #                 f"{np.percentile(positive_probabilities, [0, 1, 25, 50, 75, 99, 100])}\n"
        #                 f"PREDICTION (Raw Coherence) Percentiles: \n"
        #                 f"{np.percentile(pred_img_reg_cor, [0, 1, 25, 50, 75, 99, 100])}\n" 
        #                 f"(GROUND TRUTH - PREDICTION) (Logit-Transformed Coherence) Percentiles: \n"
        #                 f"{np.percentile(residuals_full_object, [0, 1, 25, 50, 75, 99, 100])}\n"
        #                 f"(GROUND TRUTH - PREDICTION) (Raw Coherence) Percentiles: \n"
        #                 f"{np.percentile(residuals_img_reg_cor, [0, 1, 25, 50, 75, 99, 100])}\n"
        #                 f"abs(GROUND TRUTH - PREDICTION)(Logit-Transformed Coherence) Percentiles: \n"
        #                 f"{np.percentile(np.abs(residuals_full_object), [0, 1, 25, 50, 75, 99, 100])}\n"
        #                 f"abs(GROUND TRUTH - PREDICTION) (Raw Coherence) Percentiles: \n"
        #                 f"{np.percentile(abs_residuals_img_reg_cor, [0, 1, 25, 50, 75, 99, 100])}\n")
        # else:
        #     logger.info(f"\n"
        #                 f"[{eval_stage.upper()}] STATISTICS\n"
        #                 f"[ MIN | 1st | 25th | 50th | 75th | 99th | MAX ]\n"
        #                 f"GROUND TRUTH Percentiles: \n"
        #                 f"{np.percentile(gt_edge_feature, [0, 1, 25, 50, 75, 99, 100])}\n"
        #                 f"PREDICTION Percentiles: \n"
        #                 f"{np.percentile(positive_probabilities, [0, 1, 25, 50, 75, 99, 100])}\n"
        #                 f"(GROUND TRUTH - PREDICTION) Percentiles: \n"
        #                 f"{np.percentile(residuals_full_object, [0, 1, 25, 50, 75, 99, 100])}\n"
        #                 f"abs(GROUND TRUTH - PREDICTION) Percentiles: \n"
        #                 f"{np.percentile(np.abs(residuals_full_object), [0, 1, 25, 50, 75, 99, 100])}\n")
        
        # Create the output folder if it doesn't exist
        result_folder = os.path.join(os.getcwd(), "image_result")
        os.makedirs(result_folder, exist_ok=True)
        use_parallel = cpu_count() > 99999  # Use smaller number e.g. 4 instead of 99999 to switch on multiprocessing
        
        # Get sizes 
        n, L = positive_probabilities.shape
        H = int(np.floor(np.sqrt(L)))
        W = int(np.ceil(L / H))
        
        # Loop through each edge to get output filenames and optionally save NumPy objects
        save_names = []
        for i in range(n):
            
            # Get dates from node and edge mappings
            edge_id = edge_ids_array[i]
            src_node_id, dst_node_id = edge_node_pairs[edge_id-1]
            src_date = node_mapping.loc[node_mapping['nodeID'] == src_node_id, 'date'].values[0]
            dst_date = node_mapping.loc[node_mapping['nodeID'] == dst_node_id, 'date'].values[0]
            src_date_str = pd.to_datetime(src_date).strftime('%Y%m%d')
            dst_date_str = pd.to_datetime(dst_date).strftime('%Y%m%d')
            save_names.append(os.path.join(result_folder, f"{eval_stage}_epoch-{epoch:04d}_src-{src_node_id:04d}_dst-{dst_node_id:04d}_edge-{edge_ids_array[i]:04d}_{src_date_str}-{dst_date_str}"))

            # Save results as NumPy objects
            if save_numpy_objects: 
                
                # Reshape ground truth and prediction into 2D
                gt_2d = reshape_to_2d(gt_edge_feature[i], H, W)
                pred_2d = reshape_to_2d(positive_probabilities[i], H, W)
                res_2d = reshape_to_2d(residuals_full_object[i], H, W)
                abs_res_2d = reshape_to_2d(np.abs(residuals_full_object[i]), H, W)
                if cor_logit:
                    gt_2d_raw = reshape_to_2d(gt_img_reg_cor[i], H, W)
                    pred_2d_raw = reshape_to_2d(pred_img_reg_cor[i], H, W)
                    res_2d_raw = reshape_to_2d(residuals_img_reg_cor[i], H, W)
                    abs_res_2d_raw = reshape_to_2d(abs_residuals_img_reg_cor[i], H, W)
                
                # If logit_cors is used, save the raw cor inside "if" loop, then save the logit cor outside "if" loop
                # If logit_cors is not used, just save the raw cor outside "if" loop
                coh_type = "raw"
                if cor_logit:   
                    np.save(save_names[i] + "_gt_cor-" + coh_type + ".npy", gt_2d_raw)
                    np.save(save_names[i] + "_pred_cor-" + coh_type + ".npy", pred_2d_raw)
                    np.save(save_names[i] + "_residual_cor-" + coh_type + ".npy", res_2d_raw)
                    np.save(save_names[i] + "_residualabs_cor-" + coh_type + ".npy", abs_res_2d_raw)
                    coh_type = "logit"
                np.save(save_names[i] + "_gt_cor-" + coh_type + ".npy", gt_2d)
                np.save(save_names[i] + "_pred_cor-" + coh_type + ".npy", pred_2d)
                np.save(save_names[i] + "_residual_cor-" + coh_type + ".npy", res_2d)
                np.save(save_names[i] + "_residualabs_cor-" + coh_type + ".npy", abs_res_2d)

        # Visualizing GT, Pred, Residuals, abs(Residuals)
        if visualize:
            tasks = []
            for i in range(n):
                # Reshape ground truth and prediction into 2D
                gt_2d = reshape_to_2d(gt_edge_feature[i], H, W)
                pred_2d = reshape_to_2d(positive_probabilities[i], H, W)
                save_name = save_names[i]
                if use_parallel:  # If many CPUs are available, create tuple for multiprocessing
                    tasks.append((logger, cor_logit, gt_2d, pred_2d, save_name))
                else:  # If only few CPUs are available, use sequential processing
                    evaluate_image_link_prediction_visualiser(logger, cor_logit, gt_2d, pred_2d, save_name)
            if use_parallel:  # If many CPUs are available, use parallel processing
                with Pool(cpu_count()) as pool:
                    pool.starmap(evaluate_image_link_prediction_visualiser, tasks)
                
        # Plot distributions of each individual edge in the epoch for GT, Pred, Residuals, abs(Residuals)
        if distributions:
            tasks = []
            for i in range(n):
                # Flatten the GT, Pred, Residuals, abs(Residuals) objects for each edge for distribution plotting
                gt_flat = gt_edge_feature[i].flatten()
                pred_flat = positive_probabilities[i].flatten()
                res_flat = residuals_full_object[i].flatten()
                abs_res_flat = np.abs(residuals_full_object)[i].flatten()
                save_name = save_names[i]
                if use_parallel:  # If many CPUs are available, create tuple for multiprocessing
                    tasks.append((logger, cor_logit, gt_flat, pred_flat, res_flat, abs_res_flat, save_name))
                else:  # If only few CPUs are available, use sequential processing
                    evaluate_image_link_prediction_plot_distributions(logger, cor_logit, gt_flat, pred_flat, res_flat, abs_res_flat, save_name)
            if use_parallel:  # If many CPUs are available, use parallel processing
                with Pool(cpu_count()) as pool:
                    pool.starmap(evaluate_image_link_prediction_plot_distributions, tasks)

        # Plot distributions of all edges collectively in the epoch for GT, Pred, Residuals, abs(Residuals)
        if distributions_all:
            # Flatten the GT, Pred, Residuals, abs(Residuals) objects for distribution plotting
            gt_flat = gt_edge_feature.flatten()
            pred_flat = positive_probabilities.flatten()
            res_flat = residuals_full_object.flatten()
            abs_res_flat = np.abs(residuals_full_object).flatten()
            # Plot results
            save_name = os.path.join(result_folder, f"{eval_stage}_epoch-{epoch:04d}_all-edges_distributions")
            evaluate_image_link_prediction_plot_distributions(logger=logger,
                                                              cor_logit=cor_logit,
                                                              gt_img=gt_flat,
                                                              pred_img=pred_flat,
                                                              residuals_img=res_flat,
                                                              abs_residuals_img=abs_res_flat,
                                                              save_name=save_name)   

        # Compute loss (using MAE / L1 loss here)
        loss = np.mean(np.abs(gt_edge_feature - positive_probabilities))
        if cor_logit:
            loss_reg_cor = np.mean(abs_residuals_img_reg_cor)

        # Apply loss regularisations (L1 / L2 / Elastic Net), if applicable
        # Apply L1 Regularisation (Lasso), if applicable
        if l1_regularisation_lambda != 0 and l2_regularisation_lambda == 0:
            logger.info(f"[{eval_stage.upper()}] Conducting L1-Regularisation (Lasso) on the Loss Function")
            l1_norm = sum(p.abs().sum() for p in model.parameters() if p.requires_grad)
            loss += l1_regularisation_lambda * l1_norm.item()
            logger.info(f"[{eval_stage.upper()}] L1-REGULARISED LOSS (L1-Lambda = {l1_regularisation_lambda}): {loss}\n")

        # Apply L2 Regularisation (Ridge), if applicable
        elif l1_regularisation_lambda == 0 and l2_regularisation_lambda != 0:
            logger.info(f"[{eval_stage.upper()}] Conducting L2-Regularisation (Ridge) on the Loss Function")
            l2_norm = sum(p.pow(2.0).sum() for p in model.parameters() if p.requires_grad)
            loss += l2_regularisation_lambda * l2_norm.item()
            logger.info(f"[{eval_stage.upper()}] L2-REGULARISED LOSS (L2-Lambda = {l2_regularisation_lambda}): {loss}\n")

        # Apply Elastic Net Regularisation (L1 + L2 Regularisation), if applicable
        elif l1_regularisation_lambda != 0 and l2_regularisation_lambda != 0:
            logger.info(f"[{eval_stage.upper()}] Conducting Elastic Net Regularisation (L1 & L2 regularisation) on the Loss Function")
            l1_norm = sum(p.abs().sum() for p in model.parameters() if p.requires_grad)
            l2_norm = sum(p.pow(2.0).sum() for p in model.parameters() if p.requires_grad)
            loss += (l1_regularisation_lambda * l1_norm.item()) + (l2_regularisation_lambda * l2_norm.item())
            logger.info(f"[{eval_stage.upper()}] ELASTIC-NET-REGULARISED LOSS (L1-Lambda = {l1_regularisation_lambda}, L2-Lambda = {l2_regularisation_lambda}): {loss}\n")
        
        # If no regularisation takes place, retain the original loss function
        else:
            logger.info(f"[{eval_stage.upper()}] No regularisation was applied and the original loss function remains unchanged.\n")

        # Append the loss to evaluate_metrics 
        evaluate_metrics.append({f'{eval_stage} MAE loss': loss})

    return evaluate_metrics


def evaluate_image_link_prediction_visualiser(logger, cor_logit, gt_img, pred_img, save_name):

    # Calculate residuals (agnostic to logit-transformed or raw coherences) from the gt_img and pred_img supplied.
    residuals_img = gt_img - pred_img
    abs_residuals_img = np.abs(residuals_img)
    
    # Compute statistics & create visualisation plots for logit-transformed coherences
    if cor_logit:

        # Get the save_path (for logit_cors)
        save_path = save_name + "_visual_cor-logit.png"

        # Compute the equivalents for logit-transformed coherences in regular coherence space using "convert_images_to_reg_cor" function
        gt_img_reg_cor, pred_img_reg_cor, residuals_img_reg_cor, abs_residuals_img_reg_cor = convert_images_to_reg_cor(gt_img, pred_img)

        # # Print result statistics for logit-transformed coherences     
        # logger.info(f"\n"
        #             f"STATISTICS for: {save_path}\n"
        #             f"[ MIN | 1st | 25th | 50th | 75th | 99th | MAX ]\n"
        #             f"GROUND TRUTH (Logit-Transformed Coherence) Percentiles: \n"
        #             f"{np.percentile(gt_img, [0, 1, 25, 50, 75, 99, 100])}\n"
        #             f"GROUND TRUTH (Raw Coherence) Percentiles: \n"
        #             f"{np.percentile(gt_img_reg_cor, [0, 1, 25, 50, 75, 99, 100])}\n"
        #             f"PREDICTION (Logit-Transformed Coherence) Percentiles: \n"
        #             f"{np.percentile(pred_img, [0, 1, 25, 50, 75, 99, 100])}\n"
        #             f"PREDICTION (Raw Coherence) Percentiles: \n"
        #             f"{np.percentile(pred_img_reg_cor, [0, 1, 25, 50, 75, 99, 100])}\n"
        #             f"(GROUND TRUTH - PREDICTION) (Logit-Transformed Coherence) Percentiles: \n"
        #             f"{np.percentile(residuals_img, [0, 1, 25, 50, 75, 99, 100])}\n"
        #             f"(GROUND TRUTH - PREDICTION) (Raw Coherence) Percentiles: \n"
        #             f"{np.percentile(residuals_img_reg_cor, [0, 1, 25, 50, 75, 99, 100])}\n"
        #             f"abs(GROUND TRUTH - PREDICTION) (Logit-Transformed Coherence) Percentiles: \n"
        #             f"{np.percentile(abs_residuals_img, [0, 1, 25, 50, 75, 99, 100])}\n"
        #             f"abs(GROUND TRUTH - PREDICTION) (Raw Coherence) Percentiles: \n"
        #             f"{np.percentile(abs_residuals_img_reg_cor, [0, 1, 25, 50, 75, 99, 100])}\n")
        
        # Visualise the results: GT, Pred, Residuals, abs(Residuals) in a 2x2 subplot
        fig, axs = plt.subplots(2, 2, figsize=(16, 16))

        # General plotting configurations
        title_fontsize = 24
        cb_fontsize = 16
        cb_shrink = 0.75
        tick_label_fontsize = 16

        # Hardcoded axes limits for logit-transformed coherences. Most logit-transformed coherence values fall between -6 to +6 in logit-space.
        vmin = -6
        vmax = 6
        vmin_abs = 0.0
        vmax_abs = 12.0
        vmin_diff = -vmax_abs
        vmax_diff = vmax_abs

        # Ground Truth Image
        im0 = axs[0,0].imshow(gt_img, cmap='gray', vmin=vmin, vmax=vmax)
        axs[0,0].set_title("[Logit-Transformed Coherence]\nGround Truth", fontsize=title_fontsize)
        axs[0,0].tick_params(labelsize=tick_label_fontsize)
        cb0 = fig.colorbar(im0, ax=axs[0,0], shrink=cb_shrink)
        cb0.ax.tick_params(labelsize=cb_fontsize)
        cb0.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        # Prediction Image
        im1 = axs[0,1].imshow(pred_img, cmap='gray', vmin=vmin, vmax=vmax)
        axs[0,1].set_title("[Logit-Transformed Coherence]\nPrediction", fontsize=title_fontsize)
        axs[0,1].tick_params(labelsize=tick_label_fontsize)
        cb1 = fig.colorbar(im1, ax=axs[0,1], shrink=cb_shrink)
        cb1.ax.tick_params(labelsize=cb_fontsize)
        cb1.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        # (Ground Truth - Prediction) Image
        im2 = axs[1,0].imshow(residuals_img, cmap='seismic', vmin=vmin_diff, vmax=vmax_diff)
        axs[1,0].set_title("[Logit-Transformed Coherence]\nGround Truth - Prediction", fontsize=title_fontsize)
        axs[1,0].tick_params(labelsize=tick_label_fontsize)
        cb2 = fig.colorbar(im2, ax=axs[1,0], shrink=cb_shrink)
        cb2.ax.tick_params(labelsize=cb_fontsize)
        cb2.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        # abs(Ground Truth - Prediction) Image
        cmap_white_red_darkred = mcolors.LinearSegmentedColormap.from_list('white_red', [(0, 'white'), (0.5, 'red'), (1, '#820000')])
        im3 = axs[1,1].imshow(abs_residuals_img, cmap=cmap_white_red_darkred, vmin=vmin_abs, vmax=vmax_abs)
        axs[1,1].set_title("[Logit-Transformed Coherence]\nabs(Ground Truth - Prediction)", fontsize=title_fontsize)
        axs[1,1].tick_params(labelsize=tick_label_fontsize)
        cb3 = fig.colorbar(im3, ax=axs[1,1], shrink=cb_shrink)
        cb3.ax.tick_params(labelsize=cb_fontsize)
        cb3.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        # Save the figure
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)
        logger.info(f"Saved prediction results to: \n{save_path}\n")

        # Reassign the variables for GT, pred, residuals, and abs(residuals) for plotting raw cor images.
        gt_img, pred_img, residuals_img, abs_residuals_img = gt_img_reg_cor, pred_img_reg_cor, residuals_img_reg_cor, abs_residuals_img_reg_cor
            
    ## Processing for regular (raw) coherences
    
    # Get the save_path for raw cors
    save_path = save_name + "_visual_cor-raw.png"
    
    # # Compute statistics & create visualisation plots for raw coherences (or logit-transformed coherences converted back to regular coherence space)
    # # Print result statistics for regular coherences               
    # logger.info(f"\n"
    #             f"STATISTICS for: {save_path}\n"
    #             f"[ MIN | 1st | 25th | 50th | 75th | 99th | MAX ]\n"
    #             f"GROUND TRUTH (Raw Coherence) Percentiles: \n"
    #             f"{np.percentile(gt_img, [0, 1, 25, 50, 75, 99, 100])}\n"
    #             f"PREDICTION (Raw Coherence) Percentiles: \n"
    #             f"{np.percentile(pred_img, [0, 1, 25, 50, 75, 99, 100])}\n"
    #             f"(GROUND TRUTH - PREDICTION) (Raw Coherence) Percentiles: \n"
    #             f"{np.percentile(residuals_img, [0, 1, 25, 50, 75, 99, 100])}\n"
    #             f"abs(GROUND TRUTH - PREDICTION) (Raw Coherence) Percentiles: \n"
    #             f"{np.percentile(abs_residuals_img, [0, 1, 25, 50, 75, 99, 100])}\n")

    # Visualise the results: GT, Pred, Residuals, abs(Residuals) in a 2x2 subplot
    fig, axs = plt.subplots(2, 2, figsize=(16, 16))

    # General plotting configurations
    title_fontsize = 24
    cb_fontsize = 16
    cb_shrink = 0.75
    tick_label_fontsize = 16

    # Hardcoded axes limits for logit-transformed coherences
    vmin = 0
    vmax = 1
    vmin_abs = 0
    vmax_abs = 1
    vmin_diff = -vmax_abs
    vmax_diff = vmax_abs

    # Ground Truth Image
    im0 = axs[0,0].imshow(gt_img, cmap='gray', vmin=vmin, vmax=vmax)
    axs[0,0].set_title("[Coherence]\nGround Truth", fontsize=title_fontsize)
    axs[0,0].tick_params(labelsize=tick_label_fontsize)
    cb0 = fig.colorbar(im0, ax=axs[0,0], shrink=cb_shrink)
    cb0.ax.tick_params(labelsize=cb_fontsize)
    cb0.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    # Prediction Image
    im1 = axs[0,1].imshow(pred_img, cmap='gray', vmin=vmin, vmax=vmax)
    axs[0,1].set_title("[Coherence]\nPrediction", fontsize=title_fontsize)
    axs[0,1].tick_params(labelsize=tick_label_fontsize)
    cb1 = fig.colorbar(im1, ax=axs[0,1], shrink=cb_shrink)
    cb1.ax.tick_params(labelsize=cb_fontsize)
    cb1.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    # (Ground Truth - Prediction) Image
    im2 = axs[1,0].imshow(residuals_img, cmap='seismic', vmin=vmin_diff, vmax=vmax_diff)
    axs[1,0].set_title("[Coherence]\nGround Truth - Prediction", fontsize=title_fontsize)
    axs[1,0].tick_params(labelsize=tick_label_fontsize)
    cb2 = fig.colorbar(im2, ax=axs[1,0], shrink=cb_shrink)
    cb2.ax.tick_params(labelsize=cb_fontsize)
    cb2.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    # abs(Ground Truth - Prediction) Image
    cmap_white_red_darkred = mcolors.LinearSegmentedColormap.from_list('white_red', [(0, 'white'), (0.5, 'red'), (1, '#820000')])
    im3 = axs[1,1].imshow(abs_residuals_img, cmap=cmap_white_red_darkred, vmin=vmin_abs, vmax=vmax_abs)
    axs[1,1].set_title("[Coherence]\nabs(Ground Truth - Prediction)", fontsize=title_fontsize)
    axs[1,1].tick_params(labelsize=tick_label_fontsize)
    cb3 = fig.colorbar(im3, ax=axs[1,1], shrink=cb_shrink)
    cb3.ax.tick_params(labelsize=cb_fontsize)
    cb3.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    # Save the figure
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    logger.info(f"Saved prediction results to: \n{save_path}\n")


def evaluate_image_link_prediction_plot_distributions(logger, cor_logit, gt_img, pred_img, residuals_img, abs_residuals_img, save_name):

    # Compute percentiles from input arrays
    def get_percentile_stats(arr):
        percentiles = [0, 1, 25, 50, 75, 99, 100]
        values = np.nanpercentile(arr.flatten(), percentiles)
        stat_text = "\n".join([f"{p:>3.0f} pct: {v:.5f}" for p, v in zip(percentiles, values)])
        return stat_text
    
    # Add statistical data text to the axes
    def add_stat_text(ax, stat_text):
        ax.text(0.98, 0.98, stat_text, transform=ax.transAxes, fontsize=14, verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # # Plot graphs depending on whether logit-transformed coherences were used
    if cor_logit:

        # Initialize figure and GridSpec (2 rows x 5 columns)
        fig = plt.figure(figsize=(40, 16))
        fig.subplots_adjust(hspace=1)
        gs = GridSpec(2, 5, figure=fig)

        # [Logit-Transformed Coherences: Identity Line Plot] Ground Truth & Prediction
        ax_id1 = fig.add_subplot(gs[0:2, 0:2])
        ax_id1.scatter(gt_img.flatten(), pred_img.flatten(), alpha=0.05, s=200, color='cornflowerblue', marker='o', edgecolors='dimgrey', linewidths=0.5)
        identity_vals = np.linspace(-6, 6, 100)
        ax_id1.plot(identity_vals, identity_vals, color='black', linestyle='--', linewidth=1)
        ax_id1.set_xlim([-6, 6])
        ax_id1.set_ylim([-6, 6])
        ax_id1.set_xlabel("Ground Truth", fontsize=16)
        ax_id1.set_ylabel("Prediction", fontsize=16)
        ax_id1.set_title("[Logit-Transformed Coherence]\nGround Truth vs Prediction", fontsize=24)
        ax_id1.grid(True, linestyle='-', linewidth=0.5, alpha=0.5)
        ax_id1.tick_params(axis='both', labelsize=14)
        ax_id1.xaxis.set_major_locator(MultipleLocator(0.5))
        ax_id1.yaxis.set_major_locator(MultipleLocator(0.5))
        ax_id1.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax_id1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    
        # [Logit-Transformed Coherences: Histogram] Ground Truth & Prediction
        ax0 = fig.add_subplot(gs[0, 2])
        sns.histplot(gt_img.flatten(), bins=100, stat='percent', element='step', fill=False, color='blue', linewidth=1.5, label='Ground Truth', ax=ax0, kde=False)
        sns.histplot(pred_img.flatten(), bins=100, stat='percent', element='step', fill=False, color='red', linewidth=1.5, label='Prediction', ax=ax0, kde=False)
        ax0.set_xlim([-6.025, 6.025])
        ax0.set_xlabel("Values", fontsize=16)
        ax0.set_ylabel("Percentage", fontsize=16)
        ax0.set_title("[Logit-Transformed Coherence Histogram]\nGround Truth & Prediction", fontsize=24)
        ax0.legend(fontsize=16)
        ax0.tick_params(axis='both', labelsize=14)
        ax0.tick_params(axis='x', rotation=45)
        ax0.grid(True, linestyle='-', linewidth=0.5, alpha=0.5)
        ax0.xaxis.set_major_locator(MultipleLocator(1))
        ax0.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax0.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        stat_text0 = f"Ground Truth\n{get_percentile_stats(gt_img)}\n\nPrediction\n{get_percentile_stats(pred_img)}"
        add_stat_text(ax0, stat_text0)

        # [Logit-Transformed Coherences: Histogram] Residuals (Ground Truth - Prediction)
        ax1 = fig.add_subplot(gs[0, 3])
        sns.histplot(residuals_img.flatten(), bins=100, stat='percent', element='step', fill=False, color='green', linewidth=1.5, label='Residuals', ax=ax1, kde=False)
        ax1.set_xlim([-12.025, 12.025])
        ax1.set_xlabel("Values", fontsize=16)
        ax1.set_ylabel("Percentage", fontsize=16)
        ax1.set_title("[Logit-Transformed Coherence Histogram]\n(Ground Truth - Prediction)", fontsize=24)
        ax1.tick_params(axis='both', labelsize=14)
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, linestyle='-', linewidth=0.5, alpha=0.5)
        ax1.xaxis.set_major_locator(MultipleLocator(2))
        ax1.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        stat_text1 = f"(GT - Pred)\n{get_percentile_stats(residuals_img)}"
        add_stat_text(ax1, stat_text1)

        # [Logit-Transformed Coherences: Histogram] Absolute Residuals (abs(Ground Truth - Prediction))
        ax2 = fig.add_subplot(gs[0, 4])
        sns.histplot(abs_residuals_img.flatten(), bins=100, stat='percent', element='step', fill=False, color='brown', linewidth=1.5, label='Abs Residuals', ax=ax2, kde=False)
        ax2.set_xlim([-0.025, 12.025])
        ax2.set_xlabel("Values", fontsize=16)
        ax2.set_ylabel("Percentage", fontsize=16)
        ax2.set_title("[Logit-Transformed Coherence Histogram]\nabs(Ground Truth - Prediction)", fontsize=24)
        ax2.tick_params(axis='both', labelsize=14)
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, linestyle='-', linewidth=0.5, alpha=0.5)
        ax2.xaxis.set_major_locator(MultipleLocator(1))
        ax2.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        stat_text2 = f"abs(GT - Pred)\n{get_percentile_stats(abs_residuals_img)}"
        add_stat_text(ax2, stat_text2)

        # [Logit-Transformed Coherences: CDF] Ground Truth & Prediction
        ax3 = fig.add_subplot(gs[1, 2])
        sns.ecdfplot(gt_img.flatten(), ax=ax3, color='blue', label='Ground Truth', linewidth=1.5, stat='percent')
        sns.ecdfplot(pred_img.flatten(), ax=ax3, color='red', label='Prediction', linewidth=1.5, stat='percent')
        for line in ax3.lines:
            line.set_drawstyle('steps-post')
        ax3.set_xlim([-6.025, 6.025])
        ax3.set_ylim([0, 100])
        ax3.set_xlabel("Values", fontsize=16)
        ax3.set_ylabel("Cumulative Percentage", fontsize=16)
        ax3.set_title("[Logit-Transformed Coherence CDF]\nGround Truth & Prediction", fontsize=24)
        ax3.legend(fontsize=16)
        ax3.tick_params(axis='both', labelsize=14)
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, linestyle='-', linewidth=0.5, alpha=0.5)
        ax3.xaxis.set_major_locator(MultipleLocator(1))
        ax3.yaxis.set_major_locator(MultipleLocator(5))
        ax3.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        stat_text3 = f"Ground Truth\n{get_percentile_stats(gt_img)}\n\nPrediction\n{get_percentile_stats(pred_img)}"
        add_stat_text(ax3, stat_text3)

        # [Logit-Transformed Coherences: CDF] Residuals (Ground Truth - Prediction)
        ax4 = fig.add_subplot(gs[1, 3])
        sns.ecdfplot(residuals_img.flatten(), ax=ax4, color='green', label='Residuals', linewidth=1.5, stat='percent')
        for line in ax4.lines:
            line.set_drawstyle('steps-post')
        ax4.set_xlim([-12.025, 12.025])
        ax4.set_ylim([0, 100])
        ax4.set_xlabel("Values", fontsize=16)
        ax4.set_ylabel("Cumulative Percentage", fontsize=16)
        ax4.set_title("[Logit-Transformed Coherence CDF]\n(Ground Truth - Prediction)", fontsize=24)
        ax4.tick_params(axis='both', labelsize=14)
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, linestyle='-', linewidth=0.5, alpha=0.5)
        ax4.xaxis.set_major_locator(MultipleLocator(2))
        ax4.yaxis.set_major_locator(MultipleLocator(5))
        ax4.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        stat_text4 = f"(GT - Pred)\n{get_percentile_stats(residuals_img)}"
        add_stat_text(ax4, stat_text4)

        # [Logit-Transformed Coherences: CDF] Absolute Residuals (abs(Ground Truth - Prediction))
        ax5 = fig.add_subplot(gs[1, 4])
        sns.ecdfplot(abs_residuals_img.flatten(), ax=ax5, color='brown', label='Abs Residuals', linewidth=1.5, stat='percent')
        for line in ax5.lines:
            line.set_drawstyle('steps-post')
        ax5.set_xlim([-0.025, 12.025])
        ax5.set_ylim([0, 100])
        ax5.set_xlabel("Values", fontsize=16)
        ax5.set_ylabel("Cumulative Percentage", fontsize=16)
        ax5.set_title("[Logit-Transformed Coherence CDF]\nabs(Ground Truth - Prediction)", fontsize=24)
        ax5.tick_params(axis='both', labelsize=14)
        ax5.tick_params(axis='x', rotation=45)
        ax5.grid(True, linestyle='-', linewidth=0.5, alpha=0.5)
        ax5.xaxis.set_major_locator(MultipleLocator(1))
        ax5.yaxis.set_major_locator(MultipleLocator(5))
        ax5.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        stat_text5 = f"abs(GT - Pred)\n{get_percentile_stats(abs_residuals_img)}"
        add_stat_text(ax5, stat_text5)

        # Save distribution plots
        plt.tight_layout()
        save_path = save_name + "_distributions_cor-logit.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Saved distribution plots to: \n{save_path}\n")

        # Compute corresponding values in regular coherence space
        gt_img_reg_cor, pred_img_reg_cor, residuals_img_reg_cor, abs_residuals_img_reg_cor = convert_images_to_reg_cor(gt_img, pred_img)

        # Reassign the variables for GT, pred, residuals, and abs(residuals) for plotting raw cor images.
        gt_img, pred_img, residuals_img, abs_residuals_img = gt_img_reg_cor, pred_img_reg_cor, residuals_img_reg_cor, abs_residuals_img_reg_cor

    # Plots for regular (raw) coherences
    # Initialize figure and GridSpec (2 rows x 5 columns)
    fig = plt.figure(figsize=(40, 16))
    fig.subplots_adjust(hspace=1)
    gs = GridSpec(2, 5, figure=fig)

    # [Coherences: Identity Line Plot] Ground Truth & Prediction
    ax_id1 = fig.add_subplot(gs[0:2, 0:2])
    ax_id1.scatter(gt_img.flatten(), pred_img.flatten(), alpha=0.05, s=200, color='cornflowerblue', marker='o', edgecolors='dimgrey', linewidths=0.5)
    identity_vals = np.linspace(0, 1, 100)
    ax_id1.plot(identity_vals, identity_vals, color='black', linestyle='--', linewidth=1)
    ax_id1.set_xlim([0, 1])
    ax_id1.set_ylim([0, 1])
    ax_id1.set_xlabel("Ground Truth", fontsize=16)
    ax_id1.set_ylabel("Prediction", fontsize=16)
    ax_id1.set_title("[Coherence]\nGround Truth vs Prediction", fontsize=24)
    ax_id1.grid(True, linestyle='-', linewidth=0.5, alpha=0.5)
    ax_id1.tick_params(axis='both', labelsize=14)
    ax_id1.xaxis.set_major_locator(MultipleLocator(0.05))
    ax_id1.yaxis.set_major_locator(MultipleLocator(0.05))
    ax_id1.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax_id1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    # [Coherences: Histogram] Ground Truth & Prediction
    ax0 = fig.add_subplot(gs[0, 2])
    sns.histplot(gt_img.flatten(), bins=100, stat='percent', element='step', fill=False, color='blue', linewidth=1.5, label='Ground Truth', ax=ax0, kde=False)
    sns.histplot(pred_img.flatten(), bins=100, stat='percent', element='step', fill=False, color='red', linewidth=1.5, label='Prediction', ax=ax0, kde=False)
    ax0.set_xlim([-0.025, 1.025])
    ax0.set_xlabel("Values", fontsize=16)
    ax0.set_ylabel("Percentage", fontsize=16)
    ax0.set_title("[Histogram]\nGround Truth & Prediction", fontsize=24)
    ax0.legend(fontsize=16)
    ax0.tick_params(axis='both', labelsize=14)
    ax0.tick_params(axis='x', rotation=45)
    ax0.grid(True, linestyle='-', linewidth=0.5, alpha=0.5)
    ax0.xaxis.set_major_locator(MultipleLocator(0.05))
    ax0.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax0.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    stat_text0 = f"Ground Truth\n{get_percentile_stats(gt_img)}\n\nPrediction\n{get_percentile_stats(pred_img)}"
    add_stat_text(ax0, stat_text0)

    # [Coherences: Histogram] Residuals (Ground Truth - Prediction)
    ax1 = fig.add_subplot(gs[0, 3])
    sns.histplot(residuals_img.flatten(), bins=100, stat='percent', element='step', fill=False, color='green', linewidth=1.5, label='Residuals', ax=ax1, kde=False)
    ax1.set_xlim([-1.05, 1.05])
    ax1.set_xlabel("Values", fontsize=16)
    ax1.set_ylabel("Percentage", fontsize=16)
    ax1.set_title("[Histogram]\n(Ground Truth - Prediction)", fontsize=24)
    ax1.tick_params(axis='both', labelsize=14)
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, linestyle='-', linewidth=0.5, alpha=0.5)
    ax1.xaxis.set_major_locator(MultipleLocator(0.1))
    ax1.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    stat_text1 = f"(GT - Pred)\n{get_percentile_stats(residuals_img)}"
    add_stat_text(ax1, stat_text1)

    # [Coherences: Histogram] Absolute Residuals (abs(Ground Truth - Prediction))
    ax2 = fig.add_subplot(gs[0, 4])
    sns.histplot(abs_residuals_img.flatten(), bins=100, stat='percent', element='step', fill=False, color='brown', linewidth=1.5, label='Abs Residuals', ax=ax2, kde=False)
    ax2.set_xlim([-0.025, 1.025])
    ax2.set_xlabel("Values", fontsize=16)
    ax2.set_ylabel("Percentage", fontsize=16)
    ax2.set_title("[Histogram]\nabs(Ground Truth - Prediction)", fontsize=24)
    ax2.tick_params(axis='both', labelsize=14)
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, linestyle='-', linewidth=0.5, alpha=0.5)
    ax2.xaxis.set_major_locator(MultipleLocator(0.05))
    ax2.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    stat_text2 = f"abs(GT - Pred)\n{get_percentile_stats(abs_residuals_img)}"
    add_stat_text(ax2, stat_text2)

    # [Coherences: CDF] Ground Truth & Prediction
    ax3 = fig.add_subplot(gs[1, 2])
    sns.ecdfplot(gt_img.flatten(), ax=ax3, color='blue', label='Ground Truth', linewidth=1.5, stat='percent')
    sns.ecdfplot(pred_img.flatten(), ax=ax3, color='red', label='Prediction', linewidth=1.5, stat='percent')
    for line in ax3.lines:
        line.set_drawstyle('steps-post')
    ax3.set_xlim([-0.025, 1.025])
    ax3.set_ylim([0, 100])
    ax3.set_xlabel("Values", fontsize=16)
    ax3.set_ylabel("Cumulative Percentage", fontsize=16)
    ax3.set_title("[CDF]\nGround Truth & Prediction", fontsize=24)
    ax3.legend(fontsize=16)
    ax3.tick_params(axis='both', labelsize=14)
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, linestyle='-', linewidth=0.5, alpha=0.5)
    ax3.xaxis.set_major_locator(MultipleLocator(0.05))
    ax3.yaxis.set_major_locator(MultipleLocator(5))
    ax3.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    stat_text3 = f"Ground Truth\n{get_percentile_stats(gt_img)}\n\nPrediction\n{get_percentile_stats(pred_img)}"
    add_stat_text(ax3, stat_text3)

    # [Coherences: CDF] Residuals (Ground Truth - Prediction)
    ax4 = fig.add_subplot(gs[1, 3])
    sns.ecdfplot(residuals_img.flatten(), ax=ax4, color='green', label='Residuals', linewidth=1.5, stat='percent')
    for line in ax4.lines:
        line.set_drawstyle('steps-post')
    ax4.set_xlim([-1.05, 1.05])
    ax4.set_ylim([0, 100])
    ax4.set_xlabel("Values", fontsize=16)
    ax4.set_ylabel("Cumulative Percentage", fontsize=16)
    ax4.set_title("[CDF]\n(Ground Truth - Prediction)", fontsize=24)
    ax4.tick_params(axis='both', labelsize=14)
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, linestyle='-', linewidth=0.5, alpha=0.5)
    ax4.xaxis.set_major_locator(MultipleLocator(0.1))
    ax4.yaxis.set_major_locator(MultipleLocator(5))
    ax4.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    stat_text4 = f"(GT - Pred)\n{get_percentile_stats(residuals_img)}"
    add_stat_text(ax4, stat_text4)

    # [Coherences: CDF] Absolute Residuals (abs(Ground Truth - Prediction))
    ax5 = fig.add_subplot(gs[1, 4])
    sns.ecdfplot(abs_residuals_img.flatten(), ax=ax5, color='brown', label='Abs Residuals', linewidth=1.5, stat='percent')
    for line in ax5.lines:
        line.set_drawstyle('steps-post')
    ax5.set_xlim([-0.025, 1.025])
    ax5.set_ylim([0, 100])
    ax5.set_xlabel("Values", fontsize=16)
    ax5.set_ylabel("Cumulative Percentage", fontsize=16)
    ax5.set_title("[CDF]\nabs(Ground Truth - Prediction)", fontsize=24)
    ax5.tick_params(axis='both', labelsize=14)
    ax5.tick_params(axis='x', rotation=45)
    ax5.grid(True, linestyle='-', linewidth=0.5, alpha=0.5)
    ax5.xaxis.set_major_locator(MultipleLocator(0.05))
    ax5.yaxis.set_major_locator(MultipleLocator(5))
    ax5.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    stat_text5 = f"abs(GT - Pred)\n{get_percentile_stats(abs_residuals_img)}"
    add_stat_text(ax5, stat_text5)

    # Save distribution plots
    plt.tight_layout()
    save_path = save_name + "_distributions_cor-raw.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Saved distribution plots to: \n{save_path}\n")


def evaluate_image_link_prediction(model_name: str, model: nn.Module, neighbor_sampler: NeighborSampler, evaluate_idx_data_loader: DataLoader, evaluate_data: Data, eval_stage: str,
                                   eval_metric_name: str, evaluator: LinkPredictionEvaluator, edge_raw_features: np.ndarray, num_neighbors: int = 20, time_gap: int = 2000):
    """
    evaluate models on the link prediction task
    :param model_name: str, name of the model
    :param model: nn.Module, the model to be evaluated
    :param neighbor_sampler: NeighborSampler, neighbor sampler
    :param evaluate_idx_data_loader: DataLoader, evaluate index data loader
    :param evaluate_neg_edge_sampler: NegativeEdgeSampler, evaluate negative edge sampler from TGB
    :param evaluate_data: Data, data to be evaluated
    :param eval_stage: str, specifies the evaluate stage to generate negative edges, can be 'val' or 'test'
    :param eval_metric_name: str, name of the evaluation metric
    :param evaluator: LinkPredictionEvaluator, link prediction evaluator
    :param num_neighbors: int, number of neighbors to sample for each node
    :param time_gap: int, time gap for neighbors to compute node features
    :return:
    """
    if model_name in ['DyRep', 'TGAT', 'TGN', 'CAWN', 'TCL', 'GraphMixer', 'DyGFormer']:
        # evaluation phase use all the graph information
        model[0].set_neighbor_sampler(neighbor_sampler)

    model.eval()

    with torch.no_grad():
        # store evaluate losses and metrics
        evaluate_metrics = []
        evaluate_idx_data_loader_tqdm = tqdm(evaluate_idx_data_loader, ncols=120)
        for batch_idx, evaluate_data_indices in enumerate(evaluate_idx_data_loader_tqdm):
            evaluate_data_indices = evaluate_data_indices.numpy()
            # print("evaluate_data.src_node_ids[evaluate_data_indices]", evaluate_data.src_node_ids[evaluate_data_indices])
            # exit(0)
            batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids = \
                evaluate_data.src_node_ids[evaluate_data_indices],  evaluate_data.dst_node_ids[evaluate_data_indices], \
                evaluate_data.node_interact_times[evaluate_data_indices], evaluate_data.edge_ids[evaluate_data_indices]

           
            # follow our previous implementation, we compute for positive and negative edges respectively
            if model_name in ['TGAT', 'CAWN', 'TCL']:
                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, output_dim)
                src_node_embeddings, dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      num_neighbors=num_neighbors)

               
            elif model_name in ['JODIE', 'DyRep', 'TGN']:
                
                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, output_dim)
                src_node_embeddings, dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      edge_ids=batch_edge_ids,
                                                                      edges_are_positive=True,
                                                                      num_neighbors=num_neighbors)
            elif model_name in ['GraphMixer']:
                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, output_dim)
                src_node_embeddings, dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      num_neighbors=num_neighbors,
                                                                      time_gap=time_gap)

                
            elif model_name in ['DyGFormer']:
                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, output_dim)
                src_node_embeddings, dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times)

               
            else:
                raise ValueError(f"Wrong value for model_name {model_name}!")

            # get positive probabilities, Tensor, shape (batch_size, )
            positive_probabilities = model[1](input_1=src_node_embeddings, input_2=dst_node_embeddings).squeeze(dim=-1).sigmoid().cpu().numpy()
            # get negative probabilities, Tensor, shape (batch_size * num_negative_samples_per_node, )
            # print("positive_probabilities", positive_probabilities.shape)
            # print("edge_raw_features[edge_ids_array]", edge_raw_features[edge_ids_array].shape)
            # Assuming positive_probabilities is already a NumPy array
            loss = np.mean(np.abs(positive_probabilities - edge_raw_features[edge_ids_array]))


            # for sample_idx in range(len(batch_src_node_ids)):
            #     # compute metric
            #     input_dict = {
            #         # use slices instead of index to keep the dimension of y_pred_pos
            #         "y_pred_pos": positive_probabilities[sample_idx: sample_idx + 1],
            #         "eval_metric": [eval_metric_name],
            #     }
            #     print("positive_probabilities[sample_idx: sample_idx + 1]", positive_probabilities[sample_idx: sample_idx + 1].shape)
            #     print("edge_raw_features[sample_idx: sample_idx + 1]", edge_raw_features[sample_idx: sample_idx + 1].shape)
            #     loss = torch.nn.functional.l1_loss(positive_probabilities[sample_idx: sample_idx + 1], torch.tensor(edge_raw_features[sample_idx: sample_idx + 1], dtype=torch.float32))

            evaluate_metrics.append({f'{eval_stage} MAE loss': loss})
            if eval_stage == "test_end":
                # Create the output folder if it doesn't exist.
                result_folder = os.path.join(os.getcwd(), "image_result")
                os.makedirs(result_folder, exist_ok=True)

                # Assume:
                #   positive_probabilities: tensor of shape [n, L] (predicted embedding)
                #   ground_truth: tensor of shape [n, L] from edge_raw_features[evaluate_data_indices]
                pred_embeddings = positive_probabilities  # shape: (n, L)
                gt_embeddings = edge_raw_features[evaluate_data_indices]  # shape: (n, L)

                n, L = pred_embeddings.shape
                # Compute an approximate 2D shape for each flattened embedding.
                H = int(np.floor(np.sqrt(L)))
                W = int(np.ceil(L / H))

                def reshape_to_2d(flat_sample, H, W):
                    """
                    Pads the flattened sample if needed, reshapes it into a 2D array.
                    We do not normalize here; we rely on histogram matching later.
                    """
                    total_pixels = H * W
                    pad_size = total_pixels - flat_sample.shape[0]
                    if pad_size > 0:
                        flat_sample = np.pad(flat_sample, (0, pad_size), mode='constant', constant_values=0)
                    return flat_sample.reshape(H, W)

                def highlight_differences(pred_img, gt_img, threshold=30):
                    """
                    pred_img and gt_img should be 2D uint8 arrays of the same shape.
                    This function highlights pixels in pred_img that differ from gt_img by more than 'threshold'
                    in red. Returns an RGB array.
                    """
                    diff = np.abs(pred_img.astype(np.int16) - gt_img.astype(np.int16))
                    # Convert the predicted image to RGB.
                    rgb = np.stack([pred_img, pred_img, pred_img], axis=-1)
                    mask = diff > threshold
                    rgb[mask] = np.array([255, 0, 0], dtype=np.uint8)
                    return rgb
                
                for i in range(n):
                    # Reshape ground truth and prediction into 2D.
                    gt_2d = reshape_to_2d(gt_embeddings[i], H, W)
                    pred_2d = reshape_to_2d(pred_embeddings[i], H, W)

                    # Step 1: Convert both to float64 for histogram matching
                    gt_2d_float = gt_2d.astype(np.float64)
                    pred_2d_float = pred_2d.astype(np.float64) 

                    # Step 2: Match the histogram of pred to gt to align brightness/contrast.
                    # 'multichannel=False' because we are dealing with a single channel (grayscale).
                    pred_matched = match_histograms(pred_2d_float, gt_2d_float)    

                    # Step 3: Normalize both images to [0, 255] for display.
                    # We'll combine them to find a common min/max so they remain visually comparable.
                    combined_min = min(gt_2d_float.min(), pred_matched.min())
                    combined_max = max(gt_2d_float.max(), pred_matched.max())
                    combined_range = combined_max - combined_min + 1e-8

                    gt_norm = ((gt_2d_float - combined_min) / combined_range * 255).astype(np.uint8)
                    pred_norm = ((pred_matched - combined_min) / combined_range * 255).astype(np.uint8)

                    # Save the result
                    # CT: I think coherence values shouldn't be adjusted/normalized, they should be treated as it is because darker/lower or brighter/higher coherences have different meanings
                    np.save(os.path.join(result_folder, f"gt_{i:02d}.npy"), gt_norm)   # Save ground truth for checking only, optionally disable 
                    np.save(os.path.join(result_folder, f"pred_{i:02d}.npy"), pred_norm)

                    # Step 4: Compute the difference image.
                    diff_img = highlight_differences(pred_norm, gt_norm, threshold=0.2)

                    # Step 5: Create a figure with 1 row and 3 columns.
                    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
                    axs[0].imshow(gt_norm, cmap='gray', vmin=0, vmax=255)
                    axs[0].set_title("Ground Truth")
                    axs[0].axis('off')

                    axs[1].imshow(pred_norm, cmap='gray', vmin=0, vmax=255)
                    axs[1].set_title("Predicted (Hist Matched)")
                    axs[1].axis('off')

                    axs[2].imshow(diff_img)
                    axs[2].set_title("Differences (Red)")
                    axs[2].axis('off')

                    plt.tight_layout()

                    # Save the figure
                    save_path = os.path.join(result_folder, f"comparison_{i:02d}.png")
                    plt.savefig(save_path)
                    plt.close(fig)
                    print(f"Saved prediction results for sample {i} to {save_path}")
                
            evaluate_idx_data_loader_tqdm.set_description(f'{eval_stage} for the {batch_idx + 1}-th batch')

    return evaluate_metrics


def evaluate_model_link_prediction(model_name: str, model: nn.Module, neighbor_sampler: NeighborSampler, evaluate_idx_data_loader: DataLoader,
                                   evaluate_neg_edge_sampler: NegativeEdgeSampler, evaluate_data: Data, eval_stage: str,
                                   eval_metric_name: str, evaluator: LinkPredictionEvaluator, num_neighbors: int = 20, time_gap: int = 2000):
    """
    evaluate models on the link prediction task
    :param model_name: str, name of the model
    :param model: nn.Module, the model to be evaluated
    :param neighbor_sampler: NeighborSampler, neighbor sampler
    :param evaluate_idx_data_loader: DataLoader, evaluate index data loader
    :param evaluate_neg_edge_sampler: NegativeEdgeSampler, evaluate negative edge sampler from TGB
    :param evaluate_data: Data, data to be evaluated
    :param eval_stage: str, specifies the evaluate stage to generate negative edges, can be 'val' or 'test'
    :param eval_metric_name: str, name of the evaluation metric
    :param evaluator: LinkPredictionEvaluator, link prediction evaluator
    :param num_neighbors: int, number of neighbors to sample for each node
    :param time_gap: int, time gap for neighbors to compute node features
    :return:
    """
    if model_name in ['DyRep', 'TGAT', 'TGN', 'CAWN', 'TCL', 'GraphMixer', 'DyGFormer']:
        # evaluation phase use all the graph information
        model[0].set_neighbor_sampler(neighbor_sampler)

    model.eval()

    with torch.no_grad():
        # store evaluate losses and metrics
        evaluate_metrics = []
        evaluate_idx_data_loader_tqdm = tqdm(evaluate_idx_data_loader, ncols=120)
        for batch_idx, evaluate_data_indices in enumerate(evaluate_idx_data_loader_tqdm):
            evaluate_data_indices = evaluate_data_indices.numpy()
            batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids = \
                evaluate_data.src_node_ids[evaluate_data_indices],  evaluate_data.dst_node_ids[evaluate_data_indices], \
                evaluate_data.node_interact_times[evaluate_data_indices], evaluate_data.edge_ids[evaluate_data_indices]

            # batch_neg_dst_node_ids_list: a list of list, where each internal list contains the ids of negative destination nodes for a positive source node
            # contain batch lists, each list with length num_negative_samples_per_node (20 in the TGB evaluation)
            # we should pay attention to the mappings of node ids, reduce 1 to convert to the original node ids
            batch_neg_dst_node_ids_list = evaluate_neg_edge_sampler.query_batch(pos_src=batch_src_node_ids - 1, pos_dst=batch_dst_node_ids - 1,
                                                                                pos_timestamp=batch_node_interact_times, split_mode=eval_stage)

            # ndarray, shape (batch_size, num_negative_samples_per_node)
            # we should pay attention to the mappings of node ids, add 1 to convert to the mapped node ids in our implementation
            batch_neg_dst_node_ids = np.array(batch_neg_dst_node_ids_list) + 1

            num_negative_samples_per_node = batch_neg_dst_node_ids.shape[1]
            # ndarray, shape (batch_size * num_negative_samples_per_node, ),
            # value -> (src_node_1_id, src_node_1_id, ..., src_node_2_id, src_node_2_id, ...)
            repeated_batch_src_node_ids = np.repeat(batch_src_node_ids, repeats=num_negative_samples_per_node, axis=0)
            # ndarray, shape (batch_size * num_negative_samples_per_node, ),
            # value -> (node_1_interact_time, node_1_interact_time, ..., node_2_interact_time, node_2_interact_time, ...)
            repeated_batch_node_interact_times = np.repeat(batch_node_interact_times, repeats=num_negative_samples_per_node, axis=0)

            # follow our previous implementation, we compute for positive and negative edges respectively
            if model_name in ['TGAT', 'CAWN', 'TCL']:
                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, output_dim)
                src_node_embeddings, dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      num_neighbors=num_neighbors)

                # get temporal embedding of negative source and negative destination nodes
                # two Tensors, with shape (batch_size * num_negative_samples_per_node, output_dim)
                neg_src_node_embeddings, neg_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=repeated_batch_src_node_ids,
                                                                      dst_node_ids=batch_neg_dst_node_ids.flatten(),
                                                                      node_interact_times=repeated_batch_node_interact_times,
                                                                      num_neighbors=num_neighbors)
            elif model_name in ['JODIE', 'DyRep', 'TGN']:
                # note that negative nodes do not change the memories while the positive nodes change the memories,
                # we need to first compute the embeddings of negative nodes for memory-based models
                # get temporal embedding of negative source and negative destination nodes
                # two Tensors, with shape (batch_size * num_negative_samples_per_node, output_dim)
                neg_src_node_embeddings, neg_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=repeated_batch_src_node_ids,
                                                                      dst_node_ids=batch_neg_dst_node_ids.flatten(),
                                                                      node_interact_times=repeated_batch_node_interact_times,
                                                                      edge_ids=None,
                                                                      edges_are_positive=False,
                                                                      num_neighbors=num_neighbors)

                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, output_dim)
                src_node_embeddings, dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      edge_ids=batch_edge_ids,
                                                                      edges_are_positive=True,
                                                                      num_neighbors=num_neighbors)
            elif model_name in ['GraphMixer']:
                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, output_dim)
                src_node_embeddings, dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      num_neighbors=num_neighbors,
                                                                      time_gap=time_gap)

                # get temporal embedding of negative source and negative destination nodes
                # two Tensors, with shape (batch_size * num_negative_samples_per_node, output_dim)
                neg_src_node_embeddings, neg_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=repeated_batch_src_node_ids,
                                                                      dst_node_ids=batch_neg_dst_node_ids.flatten(),
                                                                      node_interact_times=repeated_batch_node_interact_times,
                                                                      num_neighbors=num_neighbors,
                                                                      time_gap=time_gap)
            elif model_name in ['DyGFormer']:
                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, output_dim)
                src_node_embeddings, dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times)

                # get temporal embedding of negative source and negative destination nodes
                # two Tensors, with shape (batch_size * num_negative_samples_per_node, output_dim)
                neg_src_node_embeddings, neg_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=repeated_batch_src_node_ids,
                                                                      dst_node_ids=batch_neg_dst_node_ids.flatten(),
                                                                      node_interact_times=repeated_batch_node_interact_times)
            else:
                raise ValueError(f"Wrong value for model_name {model_name}!")

            # get positive probabilities, Tensor, shape (batch_size, )
            positive_probabilities = model[1](input_1=src_node_embeddings, input_2=dst_node_embeddings).squeeze(dim=-1).sigmoid().cpu().numpy()
            # get negative probabilities, Tensor, shape (batch_size * num_negative_samples_per_node, )
            negative_probabilities = model[1](input_1=neg_src_node_embeddings, input_2=neg_dst_node_embeddings).squeeze(dim=-1).sigmoid().cpu().numpy()

            for sample_idx in range(len(batch_src_node_ids)):
                # compute metric
                input_dict = {
                    # use slices instead of index to keep the dimension of y_pred_pos
                    "y_pred_pos": positive_probabilities[sample_idx: sample_idx + 1],
                    "y_pred_neg": negative_probabilities[sample_idx * num_negative_samples_per_node: (sample_idx + 1) * num_negative_samples_per_node],
                    "eval_metric": [eval_metric_name],
                }
                evaluate_metrics.append({eval_metric_name: evaluator.eval(input_dict)[eval_metric_name]})

            evaluate_idx_data_loader_tqdm.set_description(f'{eval_stage} for the {batch_idx + 1}-th batch')

    return evaluate_metrics


def evaluate_model_node_classification(model_name: str, model: nn.Module, neighbor_sampler: NeighborSampler, evaluate_idx_data_loader: DataLoader,
                                       evaluate_data: Data, eval_stage: str, eval_metric_name: str, evaluator: NodeClassificationEvaluator,
                                       loss_func: nn.Module, num_neighbors: int = 20, time_gap: int = 2000):
    """
    evaluate models on the node classification task
    :param model_name: str, name of the model
    :param model: nn.Module, the model to be evaluated
    :param neighbor_sampler: NeighborSampler, neighbor sampler
    :param evaluate_idx_data_loader: DataLoader, evaluate index data loader
    :param evaluate_data: Data, data to be evaluated
    :param eval_stage: str, specifies the evaluate stage, can be 'val' or 'test'
    :param eval_metric_name: str, name of the evaluation metric
    :param evaluator: NodeClassificationEvaluator, node classification evaluator
    :param loss_func: nn.Module, loss function
    :param num_neighbors: int, number of neighbors to sample for each node
    :param time_gap: int, time gap for neighbors to compute node features
    :return:
    """
    if model_name in ['DyRep', 'TGAT', 'TGN', 'CAWN', 'TCL', 'GraphMixer', 'DyGFormer']:
        # evaluation phase use all the graph information
        model[0].set_neighbor_sampler(neighbor_sampler)

    model.eval()

    with torch.no_grad():
        # store evaluate losses and metrics
        evaluate_losses, evaluate_metrics = [], []
        # store the results for each timeslot, and finally compute the metric for each timeslot
        # dictionary of list, key is the timeslot, value is a list, where each element is a prediction, np.ndarray with shape (num_classes, )
        evaluate_predicts_per_timeslot_dict = defaultdict(list)
        # dictionary of list, key is the timeslot, value is a list, where each element is a label, np.ndarray with shape (num_classes, )
        evaluate_labels_per_timeslot_dict = defaultdict(list)
        evaluate_idx_data_loader_tqdm = tqdm(evaluate_idx_data_loader, ncols=120)
        for batch_idx, evaluate_data_indices in enumerate(evaluate_idx_data_loader_tqdm):
            evaluate_data_indices = evaluate_data_indices.numpy()
            batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids, batch_labels, batch_interact_types, batch_node_label_times = \
                evaluate_data.src_node_ids[evaluate_data_indices],  evaluate_data.dst_node_ids[evaluate_data_indices], \
                evaluate_data.node_interact_times[evaluate_data_indices], evaluate_data.edge_ids[evaluate_data_indices], \
                evaluate_data.labels[evaluate_data_indices], evaluate_data.interact_types[evaluate_data_indices], \
                evaluate_data.node_label_times[evaluate_data_indices]

            # split the batch data based on interaction types
            # train_idx = torch.tensor(np.where(batch_interact_types == 'train')[0])
            if eval_stage == 'val':
                eval_idx = torch.tensor(np.where(batch_interact_types == 'validate')[0])
                # other_idx = torch.tensor(np.where(batch_interact_types == 'test')[0])
            else:
                assert eval_stage == 'test', f"Wrong setting of eval_stage {eval_stage}!"
                eval_idx = torch.tensor(np.where(batch_interact_types == 'test')[0])
                # other_idx = torch.tensor(np.where(batch_interact_types == 'validate')[0])
            # just_update_idx = torch.tensor(np.where(batch_interact_types == 'just_update')[0])
            # assert len(train_idx) == len(other_idx) == 0 and len(eval_idx) + len(just_update_idx) == len(batch_interact_types), "The data are mixed!"

            # for memory-based models, we should use all the interactions to update memories (including eval_stage and 'just_update'),
            # while other memory-free methods only need to compute on eval_stage
            if model_name in ['JODIE', 'DyRep', 'TGN']:
                # get temporal embedding of source and destination nodes, note that the memories are updated during the forward process
                # two Tensors, with shape (batch_size, output_dim)
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      edge_ids=batch_edge_ids,
                                                                      edges_are_positive=True,
                                                                      num_neighbors=num_neighbors)
            else:
                if len(eval_idx) > 0:
                    if model_name in ['TGAT', 'CAWN', 'TCL']:
                        # get temporal embedding of source and destination nodes
                        # two Tensors, with shape (batch_size, output_dim)
                        batch_src_node_embeddings, batch_dst_node_embeddings = \
                            model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                              dst_node_ids=batch_dst_node_ids,
                                                                              node_interact_times=batch_node_interact_times,
                                                                              num_neighbors=num_neighbors)

                    elif model_name in ['GraphMixer']:
                        # get temporal embedding of source and destination nodes
                        # two Tensors, with shape (batch_size, output_dim)
                        batch_src_node_embeddings, batch_dst_node_embeddings = \
                            model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                              dst_node_ids=batch_dst_node_ids,
                                                                              node_interact_times=batch_node_interact_times,
                                                                              num_neighbors=num_neighbors,
                                                                              time_gap=time_gap)
                    elif model_name in ['DyGFormer']:
                        # get temporal embedding of source and destination nodes
                        # two Tensors, with shape (batch_size, output_dim)
                        batch_src_node_embeddings, batch_dst_node_embeddings = \
                            model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                              dst_node_ids=batch_dst_node_ids,
                                                                              node_interact_times=batch_node_interact_times)
                    else:
                        raise ValueError(f"Wrong value for model_name {model_name}!")
                else:
                    batch_src_node_embeddings = None

            if len(eval_idx) > 0:
                # get predicted probabilities, shape (batch_size, num_classes)
                predicts = model[1](x=batch_src_node_embeddings).squeeze(dim=-1)
                labels = torch.from_numpy(batch_labels).float().to(predicts.device)

                loss = loss_func(input=predicts[eval_idx], target=labels[eval_idx])

                evaluate_losses.append(loss.item())
                # append the predictions and labels to evaluate_predicts_per_timeslot_dict and evaluate_labels_per_timeslot_dict
                for idx in eval_idx:
                    evaluate_predicts_per_timeslot_dict[batch_node_label_times[idx]].append(predicts[idx].softmax(dim=0).cpu().detach().numpy())
                    evaluate_labels_per_timeslot_dict[batch_node_label_times[idx]].append(labels[idx].cpu().detach().numpy())

                evaluate_idx_data_loader_tqdm.set_description(f'{eval_stage} for the {batch_idx + 1}-th batch, loss: {loss.item()}')

        # compute the evaluation metric for each timeslot
        for time_slot in tqdm(evaluate_predicts_per_timeslot_dict):
            time_slot_predictions = np.stack(evaluate_predicts_per_timeslot_dict[time_slot], axis=0)
            time_slot_labels = np.stack(evaluate_labels_per_timeslot_dict[time_slot], axis=0)
            # compute metric
            input_dict = {
                "y_true": time_slot_labels,
                "y_pred": time_slot_predictions,
                "eval_metric": [eval_metric_name],
            }
            evaluate_metrics.append({eval_metric_name: evaluator.eval(input_dict)[eval_metric_name]})

    return evaluate_losses, evaluate_metrics


def evaluate_edge_bank_link_prediction(args: argparse.Namespace, train_data: Data, val_data: Data, test_data: Data,
                                       val_idx_data_loader: DataLoader, test_idx_data_loader: DataLoader,
                                       evaluate_neg_edge_sampler: NegativeEdgeSampler, eval_metric_name: str, dataset_name: str,):
    """
    evaluate the EdgeBank model for link prediction
    :param args: argparse.Namespace, configuration
    :param train_data: Data, train data
    :param val_data: Data, validation data
    :param test_data: Data, test data
    :param val_idx_data_loader: DataLoader, validate index data loader
    :param test_idx_data_loader: DataLoader, test index data loader
    :param evaluate_neg_edge_sampler: NegativeEdgeSampler, evaluate negative edge sampler from TGB
    :param eval_metric_name: str, name of the evaluation metric
    :param dataset_name: str, dataset name
    :return:
    """
    # generate the train_validation split of the data: needed for constructing the memory for EdgeBank
    train_val_data = Data(src_node_ids=np.concatenate([train_data.src_node_ids, val_data.src_node_ids], axis=0),
                          dst_node_ids=np.concatenate([train_data.dst_node_ids, val_data.dst_node_ids], axis=0),
                          node_interact_times=np.concatenate([train_data.node_interact_times, val_data.node_interact_times], axis=0),
                          edge_ids=np.concatenate([train_data.edge_ids, val_data.edge_ids], axis=0),
                          labels=np.concatenate([train_data.labels, val_data.labels], axis=0))

    val_metric_all_runs, test_metric_all_runs = [], []

    for run in range(args.num_runs):

        set_random_seed(seed=run)

        args.seed = run
        if args.edge_bank_memory_mode == "time_window_memory":
            args.save_result_name = f'eval_{args.model_name}_{args.edge_bank_memory_mode}_{args.time_window_mode}_seed{args.seed}'
        else:
            args.save_result_name = f'eval_{args.model_name}_{args.edge_bank_memory_mode}_seed{args.seed}'

        # set up logger
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        os.makedirs(f"./logs/{args.model_name}/{args.dataset_name}/{args.save_result_name}/", exist_ok=True)
        # create file handler that logs debug and higher level messages
        fh = logging.FileHandler(f"./logs/{args.model_name}/{args.dataset_name}/{args.save_result_name}/{str(time.time())}.log")
        fh.setLevel(logging.DEBUG)
        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)
        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        # add the handlers to logger
        logger.addHandler(fh)
        logger.addHandler(ch)

        run_start_time = time.time()
        logger.info(f"********** Run {run + 1} starts. **********")

        logger.info(f'configuration is {args}')

        # evaluate EdgeBank
        logger.info(f'get final performance on dataset {args.dataset_name}...')

        evaluator = LinkPredictionEvaluator(name=dataset_name)

        val_metrics, test_metrics = [], []
        val_idx_data_loader_tqdm = tqdm(val_idx_data_loader, ncols=120)
        test_idx_data_loader_tqdm = tqdm(test_idx_data_loader, ncols=120)

        for batch_idx, val_data_indices in enumerate(val_idx_data_loader_tqdm):
            val_data_indices = val_data_indices.numpy()
            batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times = \
                val_data.src_node_ids[val_data_indices], val_data.dst_node_ids[val_data_indices], \
                val_data.node_interact_times[val_data_indices]

            # batch_neg_dst_node_ids_list: a list of list, where each internal list contains the ids of negative destination nodes for a positive source node
            # contain batch lists, each list with length num_negative_samples_per_node (20 in the TGB evaluation)
            # we should pay attention to the mappings of node ids, reduce 1 to convert to the original node ids
            batch_neg_dst_node_ids_list = evaluate_neg_edge_sampler.query_batch(pos_src=batch_src_node_ids - 1, pos_dst=batch_dst_node_ids - 1,
                                                                                pos_timestamp=batch_node_interact_times, split_mode="val")

            # ndarray, shape (batch_size, num_negative_samples_per_node)
            # we should pay attention to the mappings of node ids, add 1 to convert to the mapped node ids in our implementation
            batch_neg_dst_node_ids = np.array(batch_neg_dst_node_ids_list) + 1

            num_negative_samples_per_node = batch_neg_dst_node_ids.shape[1]
            # ndarray, shape (batch_size * num_negative_samples_per_node, ),
            # value -> (src_node_1_id, src_node_1_id, ..., src_node_2_id, src_node_2_id, ...)
            repeated_batch_src_node_ids = np.repeat(batch_src_node_ids, repeats=num_negative_samples_per_node, axis=0)

            positive_edges = (batch_src_node_ids, batch_dst_node_ids)
            negative_edges = (repeated_batch_src_node_ids, batch_neg_dst_node_ids.flatten())

            # incorporate the testing data before the current batch to history_data, which is similar to memory-based models
            history_data = Data(src_node_ids=np.concatenate([train_data.src_node_ids, val_data.src_node_ids[: val_data_indices[0]]], axis=0),
                                dst_node_ids=np.concatenate([train_data.dst_node_ids, val_data.dst_node_ids[: val_data_indices[0]]], axis=0),
                                node_interact_times=np.concatenate([train_data.node_interact_times, val_data.node_interact_times[: val_data_indices[0]]], axis=0),
                                edge_ids=np.concatenate([train_data.edge_ids, val_data.edge_ids[: val_data_indices[0]]], axis=0),
                                labels=np.concatenate([train_data.labels, val_data.labels[: val_data_indices[0]]], axis=0))

            # perform link prediction for EdgeBank
            # positive_probabilities, Tensor, shape (batch_size, )
            # negative_probabilities, Tensor, shape (batch_size * num_negative_samples_per_node, )
            positive_probabilities, negative_probabilities = edge_bank_link_prediction(history_data=history_data,
                                                                                       positive_edges=positive_edges,
                                                                                       negative_edges=negative_edges,
                                                                                       edge_bank_memory_mode=args.edge_bank_memory_mode,
                                                                                       time_window_mode=args.time_window_mode,
                                                                                       time_window_proportion=0.15)

            for sample_idx in range(len(batch_src_node_ids)):
                # compute MRR
                input_dict = {
                    # use slices instead of index to keep the dimension of y_pred_pos
                    "y_pred_pos": positive_probabilities[sample_idx: sample_idx + 1],
                    "y_pred_neg": negative_probabilities[sample_idx * num_negative_samples_per_node: (sample_idx + 1) * num_negative_samples_per_node],
                    "eval_metric": [eval_metric_name],
                }
                val_metrics.append({eval_metric_name: evaluator.eval(input_dict)[eval_metric_name]})

            val_idx_data_loader_tqdm.set_description(f'validate for the {batch_idx + 1}-th batch')

        for batch_idx, test_data_indices in enumerate(test_idx_data_loader_tqdm):
            test_data_indices = test_data_indices.numpy()
            batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times = \
                test_data.src_node_ids[test_data_indices], test_data.dst_node_ids[test_data_indices], \
                test_data.node_interact_times[test_data_indices]

            # batch_neg_dst_node_ids_list: a list of list, where each internal list contains the ids of negative destination nodes for a positive source node
            # contain batch lists, each list with length num_negative_samples_per_node (20 in the TGB evaluation)
            # we should pay attention to the mappings of node ids, reduce 1 to convert to the original node ids
            batch_neg_dst_node_ids_list = evaluate_neg_edge_sampler.query_batch(pos_src=batch_src_node_ids - 1, pos_dst=batch_dst_node_ids - 1,
                                                                                pos_timestamp=batch_node_interact_times, split_mode="test")

            # ndarray, shape (batch_size, num_negative_samples_per_node)
            # we should pay attention to the mappings of node ids, add 1 to convert to the mapped node ids in our implementation
            batch_neg_dst_node_ids = np.array(batch_neg_dst_node_ids_list) + 1

            num_negative_samples_per_node = batch_neg_dst_node_ids.shape[1]
            # ndarray, shape (batch_size * num_negative_samples_per_node, ),
            # value -> (src_node_1_id, src_node_1_id, ..., src_node_2_id, src_node_2_id, ...)
            repeated_batch_src_node_ids = np.repeat(batch_src_node_ids, repeats=num_negative_samples_per_node, axis=0)

            positive_edges = (batch_src_node_ids, batch_dst_node_ids)
            negative_edges = (repeated_batch_src_node_ids, batch_neg_dst_node_ids.flatten())

            # incorporate the testing data before the current batch to history_data, which is similar to memory-based models
            history_data = Data(src_node_ids=np.concatenate([train_val_data.src_node_ids, test_data.src_node_ids[: test_data_indices[0]]], axis=0),
                                dst_node_ids=np.concatenate([train_val_data.dst_node_ids, test_data.dst_node_ids[: test_data_indices[0]]], axis=0),
                                node_interact_times=np.concatenate([train_val_data.node_interact_times, test_data.node_interact_times[: test_data_indices[0]]], axis=0),
                                edge_ids=np.concatenate([train_val_data.edge_ids, test_data.edge_ids[: test_data_indices[0]]], axis=0),
                                labels=np.concatenate([train_val_data.labels, test_data.labels[: test_data_indices[0]]], axis=0))

            # perform link prediction for EdgeBank
            # positive_probabilities, Tensor, shape (batch_size, )
            # negative_probabilities, Tensor, shape (batch_size * num_negative_samples_per_node, )
            positive_probabilities, negative_probabilities = edge_bank_link_prediction(history_data=history_data,
                                                                                       positive_edges=positive_edges,
                                                                                       negative_edges=negative_edges,
                                                                                       edge_bank_memory_mode=args.edge_bank_memory_mode,
                                                                                       time_window_mode=args.time_window_mode,
                                                                                       time_window_proportion=0.15)

            for sample_idx in range(len(batch_src_node_ids)):
                # compute MRR
                input_dict = {
                    # use slices instead of index to keep the dimension of y_pred_pos
                    "y_pred_pos": positive_probabilities[sample_idx: sample_idx + 1],
                    "y_pred_neg": negative_probabilities[sample_idx * num_negative_samples_per_node: (sample_idx + 1) * num_negative_samples_per_node],
                    "eval_metric": [eval_metric_name],
                }
                test_metrics.append({eval_metric_name: evaluator.eval(input_dict)[eval_metric_name]})

            test_idx_data_loader_tqdm.set_description(f'test for the {batch_idx + 1}-th batch')

        # store the evaluation metrics at the current run
        val_metric_dict, test_metric_dict = {}, {}

        for metric_name in val_metrics[0].keys():
            average_val_metric = np.mean([val_metric[metric_name] for val_metric in val_metrics])
            logger.info(f'validate {metric_name}, {average_val_metric:.4f}')
            val_metric_dict[metric_name] = average_val_metric

        for metric_name in test_metrics[0].keys():
            average_test_metric = np.mean([test_metric[metric_name] for test_metric in test_metrics])
            logger.info(f'test {metric_name}, {average_test_metric:.4f}')
            test_metric_dict[metric_name] = average_test_metric

        single_run_time = time.time() - run_start_time
        logger.info(f'Run {run + 1} cost {single_run_time:.2f} seconds.')

        val_metric_all_runs.append(val_metric_dict)
        test_metric_all_runs.append(test_metric_dict)

        # avoid the overlap of logs
        if run < args.num_runs - 1:
            logger.removeHandler(fh)
            logger.removeHandler(ch)

        # save model result
        result_json = {
            "validate metrics": {metric_name: f'{val_metric_dict[metric_name]:.4f}' for metric_name in val_metric_dict},
            "test metrics": {metric_name: f'{test_metric_dict[metric_name]:.4f}' for metric_name in test_metric_dict}
        }
        result_json = json.dumps(result_json, indent=4)

        save_result_folder = f"./saved_results/{args.model_name}/{args.dataset_name}"
        os.makedirs(save_result_folder, exist_ok=True)
        save_result_path = os.path.join(save_result_folder, f"{args.save_result_name}.json")
        with open(save_result_path, 'w') as file:
            file.write(result_json)
        logger.info(f'save results at {save_result_path}')

    # store the average metrics at the log of the last run
    logger.info(f'metrics over {args.num_runs} runs:')

    for metric_name in val_metric_all_runs[0].keys():
        logger.info(f'validate {metric_name}, {[val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs]}')
        logger.info(f'average validate {metric_name}, {np.mean([val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs]):.4f} '
                    f'± {np.std([val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs], ddof=1):.4f}')

    for metric_name in test_metric_all_runs[0].keys():
        logger.info(f'test {metric_name}, {[test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]}')
        logger.info(f'average test {metric_name}, {np.mean([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]):.4f} '
                    f'± {np.std([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs], ddof=1):.4f}')


def evaluate_parameter_free_node_classification(args: argparse.Namespace, train_data: Data, val_data: Data, test_data: Data,
                                                train_idx_data_loader: DataLoader, val_idx_data_loader: DataLoader,
                                                test_idx_data_loader: DataLoader, eval_metric_name: str, num_classes: int):
    """
    evaluate parameter-free models (PersistentForecast and MovingAverage) on the node classification task
    :param args: argparse.Namespace, configuration
    :param train_data: Data, train data
    :param val_data: Data, validation data
    :param test_data: Data, test data
    :param train_idx_data_loader: DataLoader, train index data loader
    :param val_idx_data_loader: DataLoader, validate index data loader
    :param test_idx_data_loader: DataLoader, test index data loader
    :param eval_metric_name: str, name of the evaluation metric
    :param num_classes: int, number of label classes
    :return:
    """
    train_metric_all_runs, val_metric_all_runs, test_metric_all_runs = [], [], []

    for run in range(args.num_runs):

        set_random_seed(seed=run)

        args.seed = run
        args.save_result_name = f'eval_{args.model_name}_seed{args.seed}'

        # set up logger
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        os.makedirs(f"./logs/{args.model_name}/{args.dataset_name}/{args.save_result_name}/", exist_ok=True)
        # create file handler that logs debug and higher level messages
        fh = logging.FileHandler(f"./logs/{args.model_name}/{args.dataset_name}/{args.save_result_name}/{str(time.time())}.log")
        fh.setLevel(logging.DEBUG)
        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)
        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        # add the handlers to logger
        logger.addHandler(fh)
        logger.addHandler(ch)

        run_start_time = time.time()
        logger.info(f"********** Run {run + 1} starts. **********")

        logger.info(f'configuration is {args}')

        # create model
        if args.model_name == 'PersistentForecast':
            model = PersistentForecast(num_classes=num_classes)
        else:
            assert args.model_name == 'MovingAverage', f"Wrong setting of model_name {args.model_name}!"
            model = MovingAverage(num_classes=num_classes, window_size=args.moving_average_window_size)
        logger.info(f'model -> {model}')

        loss_func = nn.CrossEntropyLoss()
        evaluator = NodeClassificationEvaluator(name=args.dataset_name)

        # evaluate the best model
        logger.info(f'get final performance on dataset {args.dataset_name}...')

        def inner_evaluate_parameter_free_node_classification(evaluate_idx_data_loader: DataLoader, evaluate_data: Data, stage: str):
            """
            inner function to evaluate parameter-free models (PersistentForecast and MovingAverage) on the node classification task,
            note that we need compute on the train data because it can modify the memory to improve validation and test performance
            :param evaluate_idx_data_loader: DataLoader, evaluate index data loader
            :param evaluate_data: Data, data to be evaluated
            :param stage: str, specifies the stage, can be 'train', 'val' or 'test'
            :return:
            """
            # store evaluate losses and metrics
            evaluate_losses, evaluate_metrics = [], []
            # store the results for each timeslot, and finally compute the metric for each timeslot
            # dictionary of list, key is the timeslot, value is a list, where each element is a prediction, np.ndarray with shape (num_classes, )
            evaluate_predicts_per_timeslot_dict = defaultdict(list)
            # dictionary of list, key is the timeslot, value is a list, where each element is a label, np.ndarray with shape (num_classes, )
            evaluate_labels_per_timeslot_dict = defaultdict(list)
            evaluate_idx_data_loader_tqdm = tqdm(evaluate_idx_data_loader, ncols=120)
            for batch_idx, evaluate_data_indices in enumerate(evaluate_idx_data_loader_tqdm):
                evaluate_data_indices = evaluate_data_indices.numpy()
                batch_src_node_ids, batch_labels, batch_interact_types, batch_node_label_times = \
                    evaluate_data.src_node_ids[evaluate_data_indices], evaluate_data.labels[evaluate_data_indices], \
                    evaluate_data.interact_types[evaluate_data_indices], evaluate_data.node_label_times[evaluate_data_indices]

                # split the batch data based on interaction types
                if stage == 'train':
                    eval_idx = torch.tensor(np.where(batch_interact_types == 'train')[0])
                    # other_idx_1 = torch.tensor(np.where(batch_interact_types == 'validate')[0])
                    # other_idx_2 = torch.tensor(np.where(batch_interact_types == 'test')[0])
                elif stage == 'val':
                    eval_idx = torch.tensor(np.where(batch_interact_types == 'validate')[0])
                    # other_idx_1 = torch.tensor(np.where(batch_interact_types == 'train')[0])
                    # other_idx_2 = torch.tensor(np.where(batch_interact_types == 'test')[0])
                else:
                    assert stage == 'test', f"Wrong setting of stage {stage}!"
                    eval_idx = torch.tensor(np.where(batch_interact_types == 'test')[0])
                    # other_idx_1 = torch.tensor(np.where(batch_interact_types == 'train')[0])
                    # other_idx_2 = torch.tensor(np.where(batch_interact_types == 'validate')[0])
                # just_update_idx = torch.tensor(np.where(batch_interact_types == 'just_update')[0])
                # assert len(other_idx_1) == len(other_idx_2) == 0 and len(eval_idx) + len(just_update_idx) == len(batch_interact_types), "The data are mixed!"

                # only use the interactions in stage to update memories, since the labels of 'just_update' are meaningless
                if len(eval_idx) > 0:
                    predicts, label_times, labels = [], [], []
                    for idx in eval_idx:
                        predict = model.get_memory(node_id=batch_src_node_ids[idx])
                        predicts.append(predict)
                        label_times.append(batch_node_label_times[idx])
                        labels.append(batch_labels[idx])
                        model.update_memory(node_id=batch_src_node_ids[idx], node_label=batch_labels[idx])
                    predicts = torch.from_numpy(np.stack(predicts, axis=0)).float()
                    labels = torch.from_numpy(np.stack(labels, axis=0)).float()

                    loss = loss_func(input=predicts, target=labels)

                    evaluate_losses.append(loss.item())
                    # append the predictions and labels to evaluate_predicts_per_timeslot_dict and evaluate_labels_per_timeslot_dict
                    for predict, label_time, label in zip(predicts, label_times, labels):
                        evaluate_predicts_per_timeslot_dict[label_time].append(predict.numpy())
                        evaluate_labels_per_timeslot_dict[label_time].append(label.numpy())

                    evaluate_idx_data_loader_tqdm.set_description(f'{stage} for the {batch_idx + 1}-th batch, loss: {loss.item()}')

            # compute the evaluation metric for each timeslot
            for time_slot in tqdm(evaluate_predicts_per_timeslot_dict):
                time_slot_predictions = np.stack(evaluate_predicts_per_timeslot_dict[time_slot], axis=0)
                time_slot_labels = np.stack(evaluate_labels_per_timeslot_dict[time_slot], axis=0)
                # compute metric
                input_dict = {
                    "y_true": time_slot_labels,
                    "y_pred": time_slot_predictions,
                    "eval_metric": [eval_metric_name],
                }
                evaluate_metrics.append({eval_metric_name: evaluator.eval(input_dict)[eval_metric_name]})

            return evaluate_losses, evaluate_metrics

        train_losses, train_metrics = inner_evaluate_parameter_free_node_classification(evaluate_idx_data_loader=train_idx_data_loader,
                                                                                        evaluate_data=train_data,
                                                                                        stage='train')

        val_losses, val_metrics = inner_evaluate_parameter_free_node_classification(evaluate_idx_data_loader=val_idx_data_loader,
                                                                                    evaluate_data=val_data,
                                                                                    stage='val')

        test_losses, test_metrics = inner_evaluate_parameter_free_node_classification(evaluate_idx_data_loader=test_idx_data_loader,
                                                                                      evaluate_data=test_data,
                                                                                      stage='test')

        # store the evaluation metrics at the current run
        train_metric_dict, val_metric_dict, test_metric_dict = {}, {}, {}

        logger.info(f'train loss: {np.mean(train_losses):.4f}')
        for metric_name in train_metrics[0].keys():
            average_train_metric = np.mean([train_metric[metric_name] for train_metric in train_metrics])
            logger.info(f'train {metric_name}, {average_train_metric:.4f}')
            train_metric_dict[metric_name] = average_train_metric

        logger.info(f'validate loss: {np.mean(val_losses):.4f}')
        for metric_name in val_metrics[0].keys():
            average_val_metric = np.mean([val_metric[metric_name] for val_metric in val_metrics])
            logger.info(f'validate {metric_name}, {average_val_metric:.4f}')
            val_metric_dict[metric_name] = average_val_metric

        logger.info(f'test loss: {np.mean(test_losses):.4f}')
        for metric_name in test_metrics[0].keys():
            average_test_metric = np.mean([test_metric[metric_name] for test_metric in test_metrics])
            logger.info(f'test {metric_name}, {average_test_metric:.4f}')
            test_metric_dict[metric_name] = average_test_metric

        single_run_time = time.time() - run_start_time
        logger.info(f'Run {run + 1} cost {single_run_time:.2f} seconds.')

        train_metric_all_runs.append(train_metric_dict)
        val_metric_all_runs.append(val_metric_dict)
        test_metric_all_runs.append(test_metric_dict)

        # avoid the overlap of logs
        if run < args.num_runs - 1:
            logger.removeHandler(fh)
            logger.removeHandler(ch)

        result_json = {
            "train metrics": {metric_name: f'{train_metric_dict[metric_name]:.4f}' for metric_name in train_metric_dict},
            "validate metrics": {metric_name: f'{val_metric_dict[metric_name]:.4f}' for metric_name in val_metric_dict},
            "test metrics": {metric_name: f'{test_metric_dict[metric_name]:.4f}' for metric_name in test_metric_dict}
        }
        result_json = json.dumps(result_json, indent=4)

        save_result_folder = f"./saved_results/{args.model_name}/{args.dataset_name}"
        os.makedirs(save_result_folder, exist_ok=True)
        save_result_path = os.path.join(save_result_folder, f"{args.save_result_name}.json")
        with open(save_result_path, 'w') as file:
            file.write(result_json)
        logger.info(f'save results at {save_result_path}')

    # store the average metrics at the log of the last run
    logger.info(f'metrics over {args.num_runs} runs:')

    for metric_name in train_metric_all_runs[0].keys():
        logger.info(f'train {metric_name}, {[train_metric_single_run[metric_name] for train_metric_single_run in train_metric_all_runs]}')
        logger.info(f'average train {metric_name}, {np.mean([train_metric_single_run[metric_name] for train_metric_single_run in train_metric_all_runs]):.4f} '
                    f'± {np.std([train_metric_single_run[metric_name] for train_metric_single_run in train_metric_all_runs], ddof=1):.4f}')

    for metric_name in val_metric_all_runs[0].keys():
        logger.info(f'validate {metric_name}, {[val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs]}')
        logger.info(f'average validate {metric_name}, {np.mean([val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs]):.4f} '
                    f'± {np.std([val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs], ddof=1):.4f}')

    for metric_name in test_metric_all_runs[0].keys():
        logger.info(f'test {metric_name}, {[test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]}')
        logger.info(f'average test {metric_name}, {np.mean([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]):.4f} '
                    f'± {np.std([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs], ddof=1):.4f}')


def reshape_to_2d(flat_sample, H, W):
    """
    Pads the flattened sample if needed, reshapes it into a 2D array.
    """
    total_pixels = H * W
    pad_size = total_pixels - flat_sample.shape[0]
    if pad_size > 0:
        flat_sample = np.pad(flat_sample, (0, pad_size), mode='constant', constant_values=0)
    return flat_sample.reshape(H, W)