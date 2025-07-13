import numpy as np
import torch
import torch.nn as nn

from models.modules import TimeEncoder, TransformerEncoder
from utils.utils import NeighborSampler


class TCL(nn.Module):

    def __init__(self, node_raw_features: np.ndarray, edge_raw_features: np.ndarray, neighbor_sampler: NeighborSampler,
                 time_feat_dim: int, output_dim: int = 172, num_layers: int = 2, num_heads: int = 2, num_depths: int = 20, dropout: float = 0.1, device: str = 'cpu'):
        """
        TCL model.
        :param node_raw_features: ndarray, shape (num_nodes + 1, node_feat_dim)
        :param edge_raw_features: ndarray, shape (num_edges + 1, edge_feat_dim)
        :param neighbor_sampler: neighbor sampler
        :param time_feat_dim: int, dimension of time features (encodings)
        :param output_dim: int, dimension of the output embedding
        :param num_layers: int, number of transformer layers
        :param num_heads: int, number of attention heads
        :param num_depths: int, number of depths, identical to the number of sampled neighbors plus 1 (involving the target node)
        :param dropout: float, dropout rate
        :param device: str, device
        """
        super(TCL, self).__init__()

        self.node_raw_features = torch.from_numpy(node_raw_features.astype(np.float32)).to(device)
        self.edge_raw_features = torch.from_numpy(edge_raw_features.astype(np.float32)).to(device)

        self.neighbor_sampler = neighbor_sampler
        self.node_feat_dim = self.node_raw_features.shape[1]
        self.edge_feat_dim = self.edge_raw_features.shape[1]
        self.time_feat_dim = time_feat_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_depths = num_depths
        self.dropout = dropout
        self.device = device

        self.time_encoder = TimeEncoder(time_dim=time_feat_dim)
        self.depth_embedding = nn.Embedding(num_embeddings=num_depths, embedding_dim=self.output_dim)

        self.projection_layer = nn.ModuleDict({
            'node': nn.Linear(in_features=self.node_feat_dim, out_features=self.output_dim, bias=True),
            'edge': nn.Linear(in_features=self.edge_feat_dim, out_features=self.output_dim, bias=True),
            'time': nn.Linear(in_features=self.time_feat_dim, out_features=self.output_dim, bias=True)
        })

        self.transformers = nn.ModuleList([
            TransformerEncoder(attention_dim=self.output_dim, num_heads=self.num_heads, dropout=self.dropout)
            for _ in range(self.num_layers)
        ])

        self.output_layer = nn.Linear(in_features=self.output_dim, out_features=self.output_dim, bias=True)

    def compute_src_dst_node_temporal_embeddings(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray,
                                                 node_interact_times: np.ndarray, num_neighbors: int = 20):
        """
        compute source and destination node temporal embeddings
        :param src_node_ids: ndarray, shape (batch_size, )
        :param dst_node_ids: ndarray, shape (batch_size, )
        :param node_interact_times: ndarray, shape (batch_size, )
        :param num_neighbors: int, number of neighbors to sample for each node
        :return:
        """
        # GET FEATURES: There are 3 options below to get the features of the source and destination nodes
        # (1) Recent node neighbors (backward only) + node itself, original TCL approach
        # (2) All node neighbors (forward and backward) + node itself
        # (3) No neighbors, node itself only
        # GET FEATURES: (1) Recent node neighbors (backward only) + node itself, original TCL approach
        # # get temporal neighbors of source nodes, including neighbor ids, edge ids and time information
        # # src_neighbor_node_ids, ndarray, shape (batch_size, num_neighbors)
        # # src_neighbor_edge_ids, ndarray, shape (batch_size, num_neighbors)
        # # src_neighbor_times, ndarray, shape (batch_size, num_neighbors)
        # src_neighbor_node_ids, src_neighbor_edge_ids, src_neighbor_times = \
        #     self.neighbor_sampler.get_historical_neighbors(node_ids=src_node_ids,
        #                                                    node_interact_times=node_interact_times,
        #                                                    num_neighbors=num_neighbors)

        # # get temporal neighbors of destination nodes, including neighbor ids, edge ids and time information
        # # dst_neighbor_node_ids, ndarray, shape (batch_size, num_neighbors)
        # # dst_neighbor_edge_ids, ndarray, shape (batch_size, num_neighbors)
        # # dst_neighbor_times, ndarray, shape (batch_size, num_neighbors)
        # dst_neighbor_node_ids, dst_neighbor_edge_ids, dst_neighbor_times = \
        #     self.neighbor_sampler.get_historical_neighbors(node_ids=dst_node_ids,
        #                                                    node_interact_times=node_interact_times,
        #                                                    num_neighbors=num_neighbors)
        
        # def fill_initial_zeros(arr):
        #     result = arr.copy()
        #     for i in range(result.shape[0]):
        #         first = result[i, 0]
        #         for j in range(1, result.shape[1]):
        #             if result[i, j] == 0:
        #                 result[i, j] = first
        #             else:
        #                 break  # stop at the first non-zero (different) value
        #     return result

        # # src_neighbor_node_ids, ndarray, shape (batch_size, num_neighbors + 1)
        # src_neighbor_node_ids = np.concatenate((src_node_ids[:, np.newaxis], src_neighbor_node_ids), axis=1)
        # # src_neighbor_edge_ids, ndarray, shape (batch_size, num_neighbors + 1)
        # src_neighbor_edge_ids = np.concatenate((np.zeros((len(src_node_ids), 1)).astype(np.longlong), src_neighbor_edge_ids), axis=1)
        # # src_neighbor_times, ndarray, shape (batch_size, num_neighbors + 1)
        # src_neighbor_times = np.concatenate((node_interact_times[:, np.newaxis], src_neighbor_times), axis=1)
        # # src_neighbor_times = fill_initial_zeros(src_neighbor_times)  # Fill empty values with node_interact_times so that recency will be zero in get_features

        # # dst_neighbor_node_ids, ndarray, shape (batch_size, num_neighbors + 1)
        # dst_neighbor_node_ids = np.concatenate((dst_node_ids[:, np.newaxis], dst_neighbor_node_ids), axis=1)
        # # dst_neighbor_edge_ids, ndarray, shape (batch_size, num_neighbors + 1)
        # dst_neighbor_edge_ids = np.concatenate((np.zeros((len(dst_node_ids), 1)).astype(np.longlong), dst_neighbor_edge_ids), axis=1)
        # # dst_neighbor_times, ndarray, shape (batch_size, num_neighbors + 1)
        # dst_neighbor_times = np.concatenate((node_interact_times[:, np.newaxis], dst_neighbor_times), axis=1)
        # # dst_neighbor_times = fill_initial_zeros(dst_neighbor_times)  # Fill empty values with node_interact_times so that recency will be zero in get_features

        # # pad the features of the sequence of source and destination nodes
        # # src_nodes_neighbor_node_raw_features, Tensor, shape (batch_size, num_neighbors + 1, node_feat_dim)
        # # src_nodes_edge_raw_features, Tensor, shape (batch_size, num_neighbors + 1, edge_feat_dim)
        # # src_nodes_neighbor_time_features, Tensor, shape (batch_size, num_neighbors + 1, time_feat_dim)
        # # src_nodes_neighbor_depth_features, Tensor, shape (num_neighbors + 1, output_dim)
        # src_nodes_neighbor_node_raw_features, src_nodes_edge_raw_features, src_nodes_neighbor_time_features, src_nodes_neighbor_depth_features = \
        #     self.get_features(node_interact_times=node_interact_times, nodes_neighbor_ids=src_neighbor_node_ids,
        #                       nodes_edge_ids=src_neighbor_edge_ids, nodes_neighbor_times=src_neighbor_times, time_encoder=self.time_encoder)

        # # dst_nodes_neighbor_node_raw_features, Tensor, shape (batch_size, num_neighbors + 1, node_feat_dim)
        # # dst_nodes_edge_raw_features, Tensor, shape (batch_size, num_neighbors + 1, edge_feat_dim)
        # # dst_nodes_neighbor_time_features, Tensor, shape (batch_size, num_neighbors + 1, time_feat_dim)
        # # dst_nodes_neighbor_depth_features, Tensor, shape (num_neighbors + 1, output_dim)
        # dst_nodes_neighbor_node_raw_features, dst_nodes_edge_raw_features, dst_nodes_neighbor_time_features, dst_nodes_neighbor_depth_features = \
        #     self.get_features(node_interact_times=node_interact_times, nodes_neighbor_ids=dst_neighbor_node_ids,
        #                       nodes_edge_ids=dst_neighbor_edge_ids, nodes_neighbor_times=dst_neighbor_times, time_encoder=self.time_encoder)
        
        # # Tensor, shape (batch_size, num_neighbors + 1, output_dim)
        # src_nodes_neighbor_node_raw_features = self.projection_layer['node'](src_nodes_neighbor_node_raw_features)
        # src_nodes_edge_raw_features = self.projection_layer['edge'](src_nodes_edge_raw_features)
        # src_nodes_neighbor_time_features = self.projection_layer['time'](src_nodes_neighbor_time_features)

        # # Tensor, shape (batch_size, num_neighbors + 1, output_dim)
        # dst_nodes_neighbor_node_raw_features = self.projection_layer['node'](dst_nodes_neighbor_node_raw_features)
        # dst_nodes_edge_raw_features = self.projection_layer['edge'](dst_nodes_edge_raw_features)
        # dst_nodes_neighbor_time_features = self.projection_layer['time'](dst_nodes_neighbor_time_features)
        
        
        # GET FEATURES: (2) All node neighbors (forward and backward) + node itself
        # def find_common_edges(nodes, src_node_ids, dst_node_ids, num_neighbors):
        #     neighbors = np.empty((len(nodes), num_neighbors+1), dtype=object)
        #     for idx, node in enumerate(nodes):
        #         # Get indices where node appears in either src or dst node IDs
        #         indices = np.where((src_node_ids == node) | (dst_node_ids == node))[0]
        #         # Get the src & dst tuple for those indices 
        #         pairs = [(src_node_ids[i], dst_node_ids[i]) for i in indices]
        #         # Use only the first num_neighbors-th 
        #         if len(pairs) > num_neighbors:
        #             pairs = pairs[:num_neighbors] 
        #         padding = [(0, 0)] * (num_neighbors+1 - len(pairs))
        #         pairs = padding + pairs
        #         neighbors[idx,] = pairs
        #     return neighbors
        # # get all neighbors within the batch for source nodes
        # src_neighbor_node_pairs = find_common_edges(src_node_ids, src_node_ids, dst_node_ids, num_neighbors)
        # # get all neighbors within the batch for source nodes
        # dst_neighbor_node_pairs = find_common_edges(dst_node_ids, src_node_ids, dst_node_ids, num_neighbors)
        
        # # get edge, time and depth features
        # src_nodes_edge_raw_features, src_nodes_neighbor_time_features, src_nodes_neighbor_depth_features = \
        #     self.get_features_all(nodes_neighbor_ids=src_neighbor_node_pairs, time_encoder=self.time_encoder)
        # dst_nodes_edge_raw_features, dst_nodes_neighbor_time_features, dst_nodes_neighbor_depth_features = \
        #     self.get_features_all(nodes_neighbor_ids=dst_neighbor_node_pairs, time_encoder=self.time_encoder)
        
        # # Tensor, shape (batch_size, num_neighbors + 1, output_dim)
        # src_nodes_edge_raw_features = self.projection_layer['edge'](src_nodes_edge_raw_features)
        # src_nodes_neighbor_time_features = self.projection_layer['time'](src_nodes_neighbor_time_features)
        
        # # Tensor, shape (batch_size, num_neighbors + 1, output_dim)
        # dst_nodes_edge_raw_features = self.projection_layer['edge'](dst_nodes_edge_raw_features)
        # dst_nodes_neighbor_time_features = self.projection_layer['time'](dst_nodes_neighbor_time_features)
        
        
        # GET FEATURES: (3) No neighbors, node itself only
        src_nodes_raw_features, src_nodes_time_features = self.get_features_no_neighbors(node_ids=src_node_ids, time_encoder=self.time_encoder)
        dst_nodes_raw_features, dst_nodes_time_features = self.get_features_no_neighbors(node_ids=dst_node_ids, time_encoder=self.time_encoder)
        
        # Tensor, shape (batch_size, output_dim)
        src_nodes_raw_features = self.projection_layer['node'](src_nodes_raw_features)
        src_nodes_time_features = self.projection_layer['time'](src_nodes_time_features)
        
        # Tensor, shape (batch_size, output_dim)
        dst_nodes_raw_features = self.projection_layer['node'](dst_nodes_raw_features)
        dst_nodes_time_features = self.projection_layer['time'](dst_nodes_time_features)


        # Concatenate features
        # Tensor, shape (batch_size, num_neighbors + 1, output_dim)
        # src_node_features = src_nodes_neighbor_node_raw_features + src_nodes_edge_raw_features + src_nodes_neighbor_time_features + src_nodes_neighbor_depth_features
        # src_node_features = src_nodes_edge_raw_features + src_nodes_neighbor_time_features + src_nodes_neighbor_depth_features
        # Tensor, shape (batch_size, num_neighbors + 1, output_dim)
        # dst_node_features = dst_nodes_neighbor_node_raw_features + dst_nodes_edge_raw_features + dst_nodes_neighbor_time_features + dst_nodes_neighbor_depth_features
        # dst_node_features = dst_nodes_edge_raw_features + dst_nodes_neighbor_time_features + dst_nodes_neighbor_depth_features
        # # Tensor, shape (batch_size, 1, output_dim)
        # src_node_features = src_nodes_raw_features + src_nodes_time_features
        src_node_features = src_nodes_time_features
        src_neighbor_node_ids = np.expand_dims(src_node_ids, axis=1)
        # # Tensor, shape (batch_size, 1, output_dim)
        # dst_node_features = dst_nodes_raw_features + dst_nodes_time_features
        dst_node_features = dst_nodes_time_features
        dst_neighbor_node_ids = np.expand_dims(dst_node_ids, axis=1)

        # Run model
        for transformer in self.transformers:
            # Self-attention block
            # Tensor, shape (batch_size, num_neighbors + 1, output_dim)
            src_node_features = transformer(inputs_query=src_node_features, inputs_key=src_node_features,
                                            inputs_value=src_node_features, neighbor_masks=src_neighbor_node_ids)
            # Tensor, shape (batch_size, num_neighbors + 1, output_dim)
            dst_node_features = transformer(inputs_query=dst_node_features, inputs_key=dst_node_features,
                                            inputs_value=dst_node_features, neighbor_masks=dst_neighbor_node_ids)
            # Cross-attention block
            # Tensor, shape (batch_size, num_neighbors + 1, output_dim)
            src_node_embeddings = transformer(inputs_query=src_node_features, inputs_key=dst_node_features,
                                              inputs_value=dst_node_features, neighbor_masks=dst_neighbor_node_ids)
            # Tensor, shape (batch_size, num_neighbors + 1, output_dim)
            dst_node_embeddings = transformer(inputs_query=dst_node_features, inputs_key=src_node_features,
                                              inputs_value=src_node_features, neighbor_masks=src_neighbor_node_ids)

            src_node_features, dst_node_features = src_node_embeddings, dst_node_embeddings

        # Retrieve the embedding of the corresponding target node, which is at the first position of the sequence
        # Tensor, shape (batch_size, output_dim)
        src_node_embeddings = self.output_layer(src_node_embeddings[:, 0, :])
        # src_node_embeddings = self.output_layer(src_node_embeddings[:, 0, :]) + src_node_features # (optional: add the input to output to counter degradation problem)
        # Tensor, shape (batch_size, output_dim)
        dst_node_embeddings = self.output_layer(dst_node_embeddings[:, 0, :])
        # dst_node_embeddings = self.output_layer(dst_node_embeddings[:, 0, :]) + dst_node_features # (optional: add the input to output to counter degradation problem)

        return src_node_embeddings, dst_node_embeddings

    def get_features_no_neighbors(self, node_ids: np.ndarray, time_encoder: TimeEncoder):
        """
        get node (either src or dst) and time features
        :param node_ids: ndarray, shape (batch_size, 1)
        :param time_encoder: TimeEncoder, time encoder
        :return:
        """
        # Load mappings for edge-node pairs and id-date information
        edge_node_pairs = np.loadtxt("image_data/edge_node_pairs.txt", delimiter="\t", dtype=np.int64)
        node_mapping = np.loadtxt("image_data/node_mapping.txt", dtype={'names': ('nodeID', 'date', 'doy'), 'formats': ('i8', 'U10', 'i8')})
        id_to_timestamp = {row['nodeID']: row['date'] for row in node_mapping}
        id_to_doy = {row['nodeID']: row['doy'] for row in node_mapping}
        
        # Tensor, shape (batch_size, 1, node_feat_dim)
        nodes_raw_features = self.node_raw_features[torch.from_numpy(node_ids[:, np.newaxis])]
        
        # Tensor, shape (batch_size, 1, time_feat_dim)
        doys = np.zeros_like(node_ids, dtype=np.int64)
        for i in range(node_ids.shape[0]):
            doys[i] = id_to_doy.get(node_ids[i], 0)
        nodes_time_features = time_encoder(timestamps=torch.from_numpy(doys[:, np.newaxis]).float().to(self.device))
        
        return nodes_raw_features, nodes_time_features
    
    def get_features(self, node_interact_times: np.ndarray, nodes_neighbor_ids: np.ndarray, nodes_edge_ids: np.ndarray,
                        nodes_neighbor_times: np.ndarray, time_encoder: TimeEncoder):
        """
        get node, edge, time and depth features
        :param node_interact_times: ndarray, shape (batch_size, )
        :param nodes_neighbor_ids: ndarray, shape (batch_size, num_neighbors + 1)
        :param nodes_edge_ids: ndarray, shape (batch_size, num_neighbors + 1)
        :param nodes_neighbor_times: ndarray, shape (batch_size, num_neighbors + 1)
        :param time_encoder: TimeEncoder, time encoder
        :return:
        """
        # Load mappings for edge-node pairs and id-date information
        edge_node_pairs = np.loadtxt("image_data/edge_node_pairs.txt", delimiter="\t", dtype=np.int64)
        node_mapping = np.loadtxt("image_data/node_mapping.txt", dtype={'names': ('nodeID', 'date', 'doy'), 'formats': ('i8', 'U10', 'i8')})
        id_to_timestamp = {row['nodeID']: row['date'] for row in node_mapping}
        id_to_doy = {row['nodeID']: row['doy'] for row in node_mapping}
        
        # Tensor, shape (batch_size, num_neighbors + 1, node_feat_dim)
        nodes_neighbor_node_raw_features = self.node_raw_features[torch.from_numpy(nodes_neighbor_ids)]
        
        # Tensor, shape (batch_size, num_neighbors + 1, edge_feat_dim)
        nodes_edge_raw_features = self.edge_raw_features[torch.from_numpy(nodes_edge_ids)]
        
        # Tensor, shape (batch_size, num_neighbors + 1, time_feat_dim)
        # Original: time feature is recency i.e. the time difference between current interaction and neighbor interactions "how long ago did this node interact with its neighbors relative to now?"
        # CT: Not sure if this logic makes sense for coherences? e.g. for coherence 6-10, current time=10 because 6-10 only occurs at time=10, which makes time difference relative to now=0,
        # but isn't what's more important is the time difference between 6 and 10 themselves which is 4? It doesn't matter what time 6-10 occurred?
        # nodes_neighbor_time_features = time_encoder(timestamps=torch.from_numpy(node_interact_times[:, np.newaxis] - nodes_neighbor_times).float().to(self.device))
        
        # Alternative: time feature is recency, but using time difference in no. of days instead of no. of IDs
        from datetime import datetime
        recency_time_diffs = np.zeros_like(nodes_edge_ids, dtype=np.int64)
        recency = node_interact_times[:, np.newaxis] - nodes_neighbor_times
        for i in range(recency.shape[0]):
            current_time = node_interact_times[i]
            current_time_date = datetime.strptime(node_mapping[node_mapping['nodeID']==current_time]['date'][0], "%Y%m%d")
            for j in range(recency.shape[1]):
                if recency[i, j] == 0:
                    recency_time_diffs[i, j] = 0
                    continue
                if recency[i, j] == current_time:
                    recency_time_diffs[i, j] = 9999 # Set to a large value if the recency is equal to the current time
                    continue
                past_time = current_time - recency[i, j]
                past_time_date = datetime.strptime(node_mapping[node_mapping['nodeID']==past_time]['date'][0], "%Y%m%d")
                recency_time_diffs[i, j] = abs((current_time_date - past_time_date).days)
        nodes_neighbor_time_features = time_encoder(timestamps=torch.from_numpy(recency_time_diffs).float().to(self.device))

        # Alternative: time feature is the time difference between the src and dst nodes which form the edge
        # from datetime import datetime
        # edge_time_diffs = np.zeros_like(nodes_edge_ids, dtype=np.int64)
        # for i in range(nodes_edge_ids.shape[0]):
        #     for j in range(nodes_edge_ids.shape[1]):
        #         edge_id = nodes_edge_ids[i, j]
        #         if edge_id == 0: # Skip or set to 0 if there is no edge
        #             edge_time_diffs[i, j] = 0
        #             continue
        #         src_id, dst_id = edge_node_pairs[edge_id-1]
        #         src_id_date = datetime.strptime(node_mapping[node_mapping['nodeID']==src_id]['date'][0], "%Y%m%d")
        #         dst_id_date = datetime.strptime(node_mapping[node_mapping['nodeID']==dst_id]['date'][0], "%Y%m%d")
        #         edge_time_diffs[i, j] = abs((src_id_date - dst_id_date).days)
        # nodes_neighbor_time_features = time_encoder(timestamps=torch.from_numpy(edge_time_diffs).float().to(self.device))
        
        # Alternative: time feature is simply the time of the node neighbour
        # doys = np.zeros_like(nodes_neighbor_ids, dtype=np.int64)
        # for i in range(nodes_neighbor_ids.shape[0]):
        #     current_time = node_interact_times[i]
        #     for j in range(nodes_neighbor_ids.shape[1]):
        #         node_id = nodes_neighbor_ids[i, j]
        #         if node_id == 0: # Skip or set to 0 if there is no node
        #             doys[i, j] = 0
        #             continue
        #         if node_id == current_time:
        #             doys[i, j] = 9999 # Set to a large value if the node ID is equal to the current time
        #             continue
        #         doys[i, j] = id_to_doy.get(node_id, 0)
        # nodes_neighbor_time_features = time_encoder(timestamps=torch.from_numpy(doys).float().to(self.device))
        
        # Ensure the number of neighbors assigned per node matches the number of depth embeddings available
        assert nodes_neighbor_ids.shape[1] == self.depth_embedding.weight.shape[0]
        
        # Tensor, shape (num_neighbors + 1, output_dim)
        nodes_neighbor_depth_features = self.depth_embedding(torch.tensor(range(nodes_neighbor_ids.shape[1])).to(self.device))

        return nodes_neighbor_node_raw_features, nodes_edge_raw_features, nodes_neighbor_time_features, nodes_neighbor_depth_features
    
    def get_features_all(self, nodes_neighbor_ids: np.ndarray, time_encoder: TimeEncoder):
        """
        get edge, time and depth features
        :param nodes_neighbor_ids: ndarray, shape (batch_size, num_neighbors + 1) 
        :param time_encoder: TimeEncoder, time encoder
        :return:
        """
        # Load mappings for edge-node pairs and id-date information
        edge_node_pairs = np.loadtxt("image_data/edge_node_pairs.txt", delimiter="\t", dtype=np.int64)
        node_mapping = np.loadtxt("image_data/node_mapping.txt", dtype={'names': ('nodeID', 'date', 'doy'), 'formats': ('i8', 'U10', 'i8')})
        id_to_timestamp = {row['nodeID']: row['date'] for row in node_mapping}
        id_to_doy = {row['nodeID']: row['doy'] for row in node_mapping}
        
        # Get corresponding edge IDs for nodes_neighbor_ids
        from datetime import datetime
        edges_neighbor_ids = np.zeros_like(nodes_neighbor_ids, dtype=np.int64)
        edge_time_diffs = np.zeros_like(nodes_neighbor_ids, dtype=np.int64)
        for i in range(edges_neighbor_ids.shape[0]):
            for j in range(edges_neighbor_ids.shape[1]):
                pair = np.array(nodes_neighbor_ids[i, j])
                if np.array_equal(pair,[0,0]):  
                    edges_neighbor_ids[i, j] = 0  # Skip or set to 0 if there is no node
                    edge_time_diffs[i, j] = 0
                else:
                    matches = np.all(edge_node_pairs == pair, axis=1)
                    edges_neighbor_ids[i, j] = np.where(matches)[0] + 1
                    src_id_date = datetime.strptime(node_mapping[node_mapping['nodeID']==pair[0]]['date'][0], "%Y%m%d")
                    dst_id_date = datetime.strptime(node_mapping[node_mapping['nodeID']==pair[1]]['date'][0], "%Y%m%d")
                    edge_time_diffs[i, j] = abs((src_id_date - dst_id_date).days)
        
        # Tensor, shape (batch_size, num_neighbors + 1, edge_feat_dim)
        nodes_edge_raw_features = self.edge_raw_features[torch.from_numpy(edges_neighbor_ids)]
        
        # Tensor, shape (batch_size, num_neighbors + 1, time_feat_dim)
        # Time feature is the time difference between the src and dst nodes which form the edge
        nodes_neighbor_time_features = time_encoder(timestamps=torch.from_numpy(edge_time_diffs).float().to(self.device))
        
        # Ensure the number of neighbors assigned per node matches the number of depth embeddings available
        assert nodes_neighbor_ids.shape[1] == self.depth_embedding.weight.shape[0]
        
        # Tensor, shape (num_neighbors + 1, output_dim)
        nodes_neighbor_depth_features = self.depth_embedding(torch.tensor(range(nodes_neighbor_ids.shape[1])).to(self.device))

        return nodes_edge_raw_features, nodes_neighbor_time_features, nodes_neighbor_depth_features

    def set_neighbor_sampler(self, neighbor_sampler: NeighborSampler):
        """
        set neighbor sampler to neighbor_sampler and reset the random state (for reproducing the results for uniform and time_interval_aware sampling)
        :param neighbor_sampler: NeighborSampler, neighbor sampler
        :return:
        """
        self.neighbor_sampler = neighbor_sampler
        if self.neighbor_sampler.sample_neighbor_strategy in ['uniform', 'time_interval_aware']:
            assert self.neighbor_sampler.seed is not None
            self.neighbor_sampler.reset_random_state()
