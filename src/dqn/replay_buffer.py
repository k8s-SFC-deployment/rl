import torch
import random


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, observations, action, reward, next_observations, done):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = (observations, action, reward, next_observations, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int):
        samples = random.sample(self.memory, batch_size)
        tot_observations = {
            "node_vertex_attr": [],
            "node_edge_attr": [],
            "node_edge_index": [],
            "service_vertex_attr": [],
            "service_edge_attr": [],
            "service_edge_index": [],
            "pod_vertex_attr": [],
            "pod_node_edge": [],
            "pod_service_edge": [],
            "target_service_idx": [],
        }
        actions = []
        rewards = []
        tot_next_observations = {
            "node_vertex_attr": [],
            "node_edge_attr": [],
            "node_edge_index": [],
            "service_vertex_attr": [],
            "service_edge_attr": [],
            "service_edge_index": [],
            "pod_vertex_attr": [],
            "pod_node_edge": [],
            "pod_service_edge": [],
            "target_service_idx": [],
        }
        dones = []
        for sample in samples:
            observations, action, reward, next_observations, done = sample
            tot_observations["node_vertex_attr"].append(observations["node_vertex_attr"])
            tot_observations["node_edge_attr"].append(observations["node_edge_attr"])
            tot_observations["node_edge_index"].append(observations["node_edge_index"])
            tot_observations["service_vertex_attr"].append(observations["service_vertex_attr"])
            tot_observations["service_edge_attr"].append(observations["service_edge_attr"])
            tot_observations["service_edge_index"].append(observations["service_edge_index"])
            tot_observations["pod_vertex_attr"].append(observations["pod_vertex_attr"])
            tot_observations["pod_node_edge"].append(observations["pod_node_edge"])
            tot_observations["pod_service_edge"].append(observations["pod_service_edge"])
            tot_observations["target_service_idx"].append(observations["target_service_idx"])
            actions.append(action)
            rewards.append(torch.tensor(reward))
            tot_next_observations["node_vertex_attr"].append(next_observations["node_vertex_attr"])
            tot_next_observations["node_edge_attr"].append(next_observations["node_edge_attr"])
            tot_next_observations["node_edge_index"].append(next_observations["node_edge_index"])
            tot_next_observations["service_vertex_attr"].append(next_observations["service_vertex_attr"])
            tot_next_observations["service_edge_attr"].append(next_observations["service_edge_attr"])
            tot_next_observations["service_edge_index"].append(next_observations["service_edge_index"])
            tot_next_observations["pod_vertex_attr"].append(next_observations["pod_vertex_attr"])
            tot_next_observations["pod_node_edge"].append(next_observations["pod_node_edge"])
            tot_next_observations["pod_service_edge"].append(next_observations["pod_service_edge"])
            tot_next_observations["target_service_idx"].append(next_observations["target_service_idx"])
            dones.append(torch.tensor(done))


        
        if len(tot_observations["node_vertex_attr"]) > 0:
            tot_observations["node_vertex_attr"] = torch.concat(tot_observations["node_vertex_attr"]).float()
            tot_observations["node_edge_attr"] = torch.concat(tot_observations["node_edge_attr"]).float()
            tot_observations["node_edge_index"] = torch.concat(tot_observations["node_edge_index"]).float()
            tot_observations["service_vertex_attr"] = torch.concat(tot_observations["service_vertex_attr"]).float()
            tot_observations["service_edge_attr"] = torch.concat(tot_observations["service_edge_attr"]).float()
            tot_observations["service_edge_index"] = torch.concat(tot_observations["service_edge_index"]).float()
            tot_observations["pod_vertex_attr"] = torch.concat(tot_observations["pod_vertex_attr"]).float()
            tot_observations["pod_node_edge"] = torch.concat(tot_observations["pod_node_edge"]).float()
            tot_observations["pod_service_edge"] = torch.concat(tot_observations["pod_service_edge"]).float()
            tot_observations["target_service_idx"] = torch.concat(tot_observations["target_service_idx"]).long()
            action = torch.concat(actions).float()
            reward = torch.concat(rewards).float()
            tot_next_observations["node_vertex_attr"] = torch.concat(tot_next_observations["node_vertex_attr"]).float()
            tot_next_observations["node_edge_attr"] = torch.concat(tot_next_observations["node_edge_attr"]).float()
            tot_next_observations["node_edge_index"] = torch.concat(tot_next_observations["node_edge_index"]).float()
            tot_next_observations["service_vertex_attr"] = torch.concat(tot_next_observations["service_vertex_attr"]).float()
            tot_next_observations["service_edge_attr"] = torch.concat(tot_next_observations["service_edge_attr"]).float()
            tot_next_observations["service_edge_index"] = torch.concat(tot_next_observations["service_edge_index"]).float()
            tot_next_observations["pod_vertex_attr"] = torch.concat(tot_next_observations["pod_vertex_attr"]).float()
            tot_next_observations["pod_node_edge"] = torch.concat(tot_next_observations["pod_node_edge"]).float()
            tot_next_observations["pod_service_edge"] = torch.concat(tot_next_observations["pod_service_edge"]).float()
            tot_next_observations["target_service_idx"] = torch.concat(tot_next_observations["target_service_idx"]).long()
            done = torch.concat(dones).bool()

        return tot_observations, action, reward, next_observations, done


    def __len__(self):
        return len(self.memory)
    