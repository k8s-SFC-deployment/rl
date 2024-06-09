import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import List

from src.apis.k8s import K8sApi
from src.apis.prom import PromApi
from src.traffic_generator import TrafficGenerator
from src.env.vnfs.vnf import VNF


class DummyClusterEnv(gym.Env):
    def __init__(self, k8sApi: K8sApi, promApi: PromApi, default_tg: TrafficGenerator, step_tg: TrafficGenerator,
                 vnfs: List[VNF], namespace: str, observation_duration: int):
        
        self.vnfs = vnfs
        self.node_num = 4
        self.node_edge_num = self.node_num * self.node_num

        self.service_num = len(self.vnfs)
        self.service_edge_num = self.service_num * self.service_num

        self.max_pod_num = self.service_num * self.node_num * 2
        
        self.max_step = (self.node_num - 2) * self.service_num
        self.cur_step = 0

        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        # action is probability of node selection.
        self.action_space = spaces.Box(low=0, high=1, shape=(self.node_num,), dtype=np.float32)

        # MultiDiscrete is a special space that represents multiple discrete spaces. But, in stable-baselines3, 2D MultiDiscrete is not supported.
        # So, we need to flatten the 2D MultiDiscrete space to 1D MultiDiscrete space.
        # Example for using node embedding as input we will normalize itself:
        node_edge_index_space = np.array([[self.node_num, self.node_num] for _ in range(self.node_edge_num)], dtype=np.int64)
        node_edge_index_space = node_edge_index_space.reshape(-1)

        service_edge_index_space = np.array([[self.service_num, self.service_num] for _ in range(self.service_edge_num)], dtype=np.int64)
        service_edge_index_space = service_edge_index_space.reshape(-1)

        pod_service_edge_space = np.array([[self.max_pod_num, self.service_num] for _ in range(self.max_pod_num)], dtype=np.int64)
        pod_service_edge_space = pod_service_edge_space.reshape(-1)

        pod_node_edge_space = np.array([[self.max_pod_num, self.node_num] for _ in range(self.max_pod_num)], dtype=np.int64)
        pod_node_edge_space = pod_node_edge_space.reshape(-1)

        self.observation_space = spaces.Dict({
            "node_vertex_attr": spaces.Box(low=0, high=np.inf, shape=(self.node_num, 4), dtype=np.float32),
            "node_edge_attr": spaces.Box(low=0, high=np.inf, shape=(self.node_edge_num, 3), dtype=np.float32),
            "node_edge_index": spaces.MultiDiscrete(node_edge_index_space, dtype=np.int64),
            "service_vertex_attr": spaces.Box(low=0, high=np.inf, shape=(self.service_num, 4), dtype=np.float32),
            "service_edge_attr": spaces.Box(low=0, high=np.inf, shape=(self.service_edge_num, 3), dtype=np.float32),
            "service_edge_index": spaces.MultiDiscrete(service_edge_index_space, dtype=np.int64),
            "pod_vertex_attr": spaces.Box(low=0, high=np.inf, shape=(self.max_pod_num, 4), dtype=np.float32),
            "pod_service_edge": spaces.MultiDiscrete(pod_service_edge_space, dtype=np.int64),
            "pod_node_edge": spaces.MultiDiscrete(pod_node_edge_space, dtype=np.int64),
            "target_service_idx": spaces.Discrete(n=self.service_num),
        })


    def format_index_to_multidiscrete(self, index):
        return index.reshape(-1)

    def format_multi_discrete_to_index(self, multi_discrete):
        return multi_discrete.reshape(-1, 2)

    def step(self, action):
        self.cur_step += 1
        observation = self._gen_dummy_observation()
        reward = 0.
        terminated = self.cur_step >= self.max_step
        truncated = False
        info = {
            "e2eLatency": { "total": 3214.321, "cnt": 18 },
            "powerConsumptions":  [{"nodename": "ed-node1", "value": 1}, {"nodename": "ed-node2", "value": 2}, {"nodename": "ed-node3", "value": 3}, {"nodename": "ed-node4", "value": 4}],
            "avgE2ELatency": 3214.321 / 18,
            "avgPowerConsumption": (1 + 2 + 3 + 4 ) / 4,
        }
        self.mask = np.ones(self.node_num)
        self.mask[0] = 0
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        self.cur_step = 0
        observation = self._gen_dummy_observation()
        info = {}
        self.mask = np.ones(self.node_num)
        self.mask[0] = 0
        return observation, info 

    def render(self):
        pass

    def close(self):
        pass

    def valid_action_mask(self):
        return self.mask

    def _gen_dummy_observation(self):
        obeservation = {
            "node_vertex_attr": np.random.randn(self.node_num, 4).astype(dtype=np.float32),
            "node_edge_attr": np.random.randn(self.node_edge_num, 3).astype(dtype=np.float32),
            "node_edge_index": self.format_index_to_multidiscrete(np.array([[i, j] for i in range(self.node_num) for j in range(self.node_num)])),
            "service_vertex_attr": np.random.randn(self.service_num, 4).astype(dtype=np.float32),
            "service_edge_attr": np.random.randn(self.service_edge_num, 3).astype(dtype=np.float32),
            "service_edge_index": self.format_index_to_multidiscrete(np.array([[i, j] for i in range(self.service_num) for j in range(self.service_num)])),
            "pod_vertex_attr": np.random.randn(self.max_pod_num, 4).astype(dtype=np.float32),
            "pod_service_edge": self.format_index_to_multidiscrete(np.array([[i, np.random.choice(self.service_num)] for i in range(self.max_pod_num)])),
            "pod_node_edge": self.format_index_to_multidiscrete(np.array([[i, np.random.choice(self.node_num)] for i in range(self.max_pod_num)])),
            "target_service_idx": np.random.choice(self.service_num),
        }

        # block minus value
        for key, value in obeservation.items():
            if "index" in key:
                continue
            if "idx" in key:
                continue
            obeservation[key] = value - value.min()

        pod_num = self.service_num + self.cur_step
        # mask out the unused pods
        obeservation["pod_vertex_attr"][pod_num:] = 0.
        obeservation["pod_service_edge"][pod_num:] = 0.
        obeservation["pod_node_edge"][pod_num:] = 0.

        # mask out the unused service edges
        service_edge_mask = np.zeros(self.service_edge_num)
        service_edge_mask[np.random.choice(self.service_edge_num, int(self.service_edge_num / 2))] = 1
        obeservation["service_edge_attr"] *= service_edge_mask.reshape(-1, 1)

        return obeservation