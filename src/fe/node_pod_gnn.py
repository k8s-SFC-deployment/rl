import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv, GATConv
from torch_geometric.nn import MessagePassing

import torch
from typing import Tuple
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class EdgeAttrMessagePassing(MessagePassing):
    def __init__(self, in_channels, output_dim, edge_dim):
        super(EdgeAttrMessagePassing, self).__init__(aggr='add')
        self.lin = nn.Linear(in_channels + edge_dim, output_dim)
    
    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
    
    def message(self, x_j, edge_attr):
        return F.relu(self.lin(torch.cat([x_j, edge_attr], dim=-1)))

class ALLGNNFE(BaseFeaturesExtractor):
    def __init__(self,  observation_space: spaces.Dict, hidden_dim: int = 16, output_dim: int = 32):
        super().__init__(observation_space, output_dim)

        self.in_channels_dict = {
            "node": observation_space["node_vertex_attr"].shape[1],
            "pod": observation_space["pod_vertex_attr"].shape[1],
        }

        self.node_num = observation_space["node_vertex_attr"].shape[0]
        self.pod_num = observation_space["pod_vertex_attr"].shape[0]

        self.edge_dim_dict = {
            ("node", "communicate", "node"): observation_space.spaces["node_edge_attr"].shape[1],
        }

        # in_channels -> hidden_dim
        self.conv1 = HeteroConv({
            ("node", "communicate", "node"): EdgeAttrMessagePassing(self.in_channels_dict["node"], hidden_dim, self.edge_dim_dict[("node", "communicate", "node")]),
            ("pod", "is_in", "node"): SAGEConv(self.in_channels_dict["node"], hidden_dim),
            ("node", "have", "pod"): SAGEConv(self.in_channels_dict["pod"], hidden_dim),
        }, aggr="sum")

        # hidden_channel -> output_dim
        self.conv2 = HeteroConv({
            ("node", "communicate", "node"): EdgeAttrMessagePassing(hidden_dim, output_dim, self.edge_dim_dict[("node", "communicate", "node")]),
            ("pod", "is_in", "node"): SAGEConv(hidden_dim, output_dim),
            ("node", "have", "pod"): SAGEConv(hidden_dim, output_dim),
        }, aggr="sum")

        self.lin = nn.Linear(output_dim * 2, output_dim)

    def _format(self, observation: dict) -> Tuple[torch.Tensor]:
        vertex_attr_dict = {
            "node": observation["node_vertex_attr"].to(torch.float32).squeeze(),
            "pod": observation["pod_vertex_attr"].to(torch.float32).squeeze()
        }

        edge_attr_dict = {
            ("node", "communicate", "node"): observation["node_edge_attr"].to(torch.float32).squeeze(),
        }

        edge_index_dict = {
            ("node", "communicate", "node"): observation["node_edge_index"].to(torch.long).squeeze().T.contiguous(),
            ("pod", "is_in", "node"): observation["pod_node_edge"].to(torch.long).squeeze().T.contiguous(),
            ("node", "have", "pod"): observation["pod_node_edge"].to(torch.long).squeeze().T.flip(0).contiguous(),
        }

        return vertex_attr_dict, edge_index_dict, edge_attr_dict

    def _preprocess_observation(self, observation: dict) -> dict:
        observation["node_edge_index"] = observation["node_edge_index"].reshape(self.node_num * self.node_num, 2, self.node_num).argmax(-1)
        observation["service_edge_index"] = observation["service_edge_index"].reshape(self.service_num * self.service_num, 2, self.service_num).argmax(-1)

        pod_idxs = observation["pod_node_edge"].reshape(self.pod_num, self.pod_num + self.node_num)[:, :self.pod_num].argmax(-1)
        node_idxs = observation["pod_node_edge"].reshape(self.pod_num, self.pod_num + self.node_num)[:, self.pod_num:].argmax(-1)
        observation["pod_node_edge"] = torch.stack((pod_idxs, node_idxs), dim=-1)
        
        pod_idxs = observation["pod_service_edge"].reshape(self.pod_num, self.pod_num + self.service_num)[:, :self.pod_num].argmax(-1)
        service_idxs = observation["pod_service_edge"].reshape(self.pod_num, self.pod_num + self.service_num)[:, self.pod_num:].argmax(-1)
        observation["pod_service_edge"] = torch.stack((pod_idxs, service_idxs), dim=-1)

        pod_mask = observation["pod_vertex_attr"].sum(axis=-1) != 0
        observation["pod_vertex_attr"] = observation["pod_vertex_attr"][pod_mask]
        observation["pod_node_edge"] = observation["pod_node_edge"][pod_mask]
        observation["pod_service_edge"] = observation["pod_service_edge"][pod_mask]

        service_edge_mask = observation["service_edge_attr"].sum(axis=-1) != 0 #
        observation["service_edge_attr"] = observation["service_edge_attr"][service_edge_mask]
        observation["service_edge_index"] = observation["service_edge_index"][service_edge_mask]

        observation["target_service_idx"] = observation["target_service_idx"].argmax(-1)

        return observation

    def forward(self, observations: dict) -> torch.Tensor:
        batch_size = observations["node_vertex_attr"].shape[0]
        output = []
        for b in range(batch_size):
            obs = {key: x[b] for key, x in observations.items()}
            preprocessed_obs = self._preprocess_observation(obs)    
            vertex_attr_dict, edge_index_dict, edge_attr_dict, target_service_idx = self._format(preprocessed_obs)

            vertex_attr_dict = self.conv1(vertex_attr_dict, edge_index_dict, edge_attr_dict)
            vertex_attr_dict = {key: F.relu(x) for key, x in vertex_attr_dict.items()}
            vertex_attr_dict = self.conv2(vertex_attr_dict, edge_index_dict, edge_attr_dict)

            node_embedding = []
            for i in range(vertex_attr_dict["node"].shape[0]):
                node_embedding.append(
                    torch.cat((
                        vertex_attr_dict["node"][i], # (output_dim)
                        vertex_attr_dict["service"][target_service_idx].view(-1), # (output_dim)
                    )) # (output_dim x 2)
                )
            node_embedding = torch.stack(node_embedding) # (Node, output_dim x 2)
            output.append(node_embedding)
        output = torch.stack(output) # (Batch, Node, output_dim x 2)
        output = F.relu(self.lin(output)) # (Batch, Node, output_dim)

        return output
