import threading
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import List

from stable_baselines3.common.env_checker import check_env

from src.collector import Collector
from src.apis.k8s import initApi as initK8sApi, K8sApi
from src.apis.prom import initApi as initPromApi, PromApi
from src.traffic_generator import TrafficGenerator
from src.env.vnfs.vnf import VNF
from src.env.vnfs.factory import create_vnfs
from src.server import BackgroundHTTPServer
from src.utils import softmax, sample_from_prob

class ClusterEnv(gym.Env):
    """Cluster Environment that follows gym interface."""

    metadata = {"render_modes": None}
   
    default_tg_key = "default"
    step_tg_key = "step"

    def __init__(self, k8sApi: K8sApi, promApi: PromApi, default_tg: TrafficGenerator, step_tg: TrafficGenerator,
                 vnfs: List[VNF], namespace: str, observation_duration: int):
        super().__init__()
        self.k8sApi = k8sApi
        self.promApi = promApi
        self.collector = Collector(k8sApi, promApi, namespace, observation_duration)
        self.default_tg = default_tg
        self.step_tg = step_tg
        self.vnfs = vnfs
        self.namespace = namespace
        self.observation_duration = observation_duration

        nodeVertexDf, _ = self.collector.getNodes()
        self.node_names = nodeVertexDf["hostname"].tolist()
        self.node_num = len(self.node_names)
        self.node_edge_num = self.node_num * self.node_num

        self.service_names = [vnf.name for vnf in vnfs]
        self.service_num = len(vnfs)
        self.service_edge_num = self.service_num * self.service_num

        self.max_pod_num = self.service_num * self.node_num * 2
        

        self.server = BackgroundHTTPServer()
        
        self.step_n = (self.node_num - 2) * self.service_num
        self.scenario = []

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

    def step(self, action):
        # 1. send response
        if action is not None:
            print(f"Action: {action}")
            self._send_response(action)
        # 2. send traffic.
        vnf_names = [f"http://{vnf.name}.{self.namespace}.svc/loadv2" for vnf in self.vnfs]
        print(f"Traffic generator is generating traffic. This lasts {self.observation_duration}s.")
        self.step_tg.generate(self.step_tg_key, vnf_names, [1, 0.5, 0.3], self.observation_duration)
        # 3. get reward
        e2eLatencies = self.step_tg.get_latencies(self.step_tg_key)
        self.step_tg.clear_latencies(self.step_tg_key)
        avgE2ELatency = e2eLatencies["latency"] / e2eLatencies["count"]
        powerConsumptions = self.collector.getPowerConsumptions()
        avgPowerConsumption = 0
        for pc in powerConsumptions:
            avgPowerConsumption += pc["value"] / len(powerConsumptions)
        reward = -(avgE2ELatency * avgPowerConsumption)
        # 4. check if terminated.
        terminated = len(self.scenario) == 0
        info = {
            "e2eLatency": e2eLatencies,
            "powerConsumptions": powerConsumptions,
            "avgE2ELatency": avgE2ELatency,
            "avgPowerConsumption": avgPowerConsumption,
        }
        truncated = False
        if terminated:
            # 4.1. if terminated, get observation and clear namespace.
            observation, dfs = self._observe()
            availability = self._get_availability(dfs)
            reward = reward / availability
            print(f"Reward: {reward}")
            info["availability"] = availability
            self._clear_namespace()
            return observation, reward, terminated, truncated, info
        # 5. get vnf from scenario.
        vnf_name = self.scenario.pop(0)
        # 6. scale up vnf.
        vnf = [vnf for vnf in self.vnfs if vnf.name == vnf_name][0]
        vnf.scaleUp(1)
        # 7. wait request from scheduler.
        req_data = self._wait_request_from_client()
        self.target_service_name = self.collector.getServicenameByPodlabels(req_data["podLabels"])
        self.target_namespaces = req_data["namespaces"]
        filteredNodeNames = req_data["nodeNames"]
        # 8. make mask for action space.
        mask = np.zeros(self.node_num)
        mask[self._nodenames_to_nodelist(filteredNodeNames)] = 1
        self.mask = mask
        # 3. get observation.
        observation, dfs = self._observe()
        availability = self._get_availability(dfs)
        reward = reward / availability
        print(f"Reward: {reward}")
        info["availability"] = availability
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        print("Start reset")
        if len(self.scenario) != 0:
            self._send_response([1 for m in self.mask if m != 0])
        np.random.seed(seed)
        # 0. reset vnfs cur_replica
        for vnf in self.vnfs: vnf.cur_replicas = 1
        # 1. if traffic exist, delete all.
        # 2. delete existing service, deployment, etc in the namespace.
        # 3. delete namespace.
        self._clear_namespace()
        # 4. create new namespace.
        # 5. create new services in the namespace.
        # 6. wait until ready.
        # 7. create new default traffic.
        self._init_namespace()
        # 8. generate replica generation scenario.
        self._create_scenario()
        # 9. make a step with no action.
        observation, _reward, _terminated, _truncated, info = self.step(None)
        print("End reset")
        return observation, info

    def render(self):
        pass

    def close(self):
        self._clear_namespace()

    def valid_action_mask(self):
        return self.mask
    
    def format_index_to_multidiscrete(self, index):
        return index.reshape(-1)

    def format_multi_discrete_to_index(self, multi_discrete):
        return multi_discrete.reshape(-1, 2)

    def _clear_namespace(self):
        if not self.default_tg.done:
            self.default_tg.finish()
            self.default_tg.clear_latencies(self.default_tg_key)
        if not self.step_tg.done:
            self.step_tg.finish()
            self.step_tg.clear_latencies(self.default_tg_key)
        self.k8sApi.deleteAll(self.namespace)
        print("Wait until existing namespace is deleted.")
        self.k8sApi.deleteNamespace(self.namespace)
        self.server.close()

    def _init_namespace(self):
        self.server.run_server()
        self.k8sApi.createNamespace(self.namespace)
        for vnf in self.vnfs:
            vnf.create()
            req, queue = self.server.wait_request()
            self.server.send_response([0 for _ in req["nodeNames"]], queue)
        print("Wait until services are ready.")
        for vnf in self.vnfs:
           self.k8sApi.isAtleastOnePodReadyInDeployment(vnf.name, self.namespace)
        self.default_tg_thread = threading.Thread(target=self.default_tg.generate, args=(self.default_tg_key, [f"http://{vnf.name}.{self.namespace}.svc/loadv2" for vnf in self.vnfs], [1], -1))
        self.default_tg_thread.start()

    # determine the order of scaling up.
    def _create_scenario(self):
        self.scenario = []
        vnf_names = [vnf.name for vnf in self.vnfs]
        for _ in range(self.node_num - 2): # install vnfs that have node_num replias. except master node(-1). except default pod(-1).
            np.random.shuffle(vnf_names)
            for vnf_name in vnf_names:
                self.scenario.append(vnf_name)
    
    def _node_name_to_idx(self, node_name: str):
        return self.node_names.index(node_name)
    
    def _node_edge_to_idx(self, src_name: str, dst_name: str):
        return self._node_name_to_idx(src_name) * self.node_num + self._node_name_to_idx(dst_name)

    def _service_name_to_idx(self, service_name: str):
        return self.service_names.index(service_name)
    
    def _service_edge_to_idx(self, src_name: str, dst_name: str):
        return self._service_name_to_idx(src_name) * self.service_num + self._service_name_to_idx(dst_name)

    def _observe(self):
        dfs = self.collector.getGraph()
        if self.target_service_name is not None:
            target_service_idx = self._service_name_to_idx(self.target_service_name)
        else:
            target_service_idx = -1

        # self.node_name을 기준으로 dfs["nodeVertexDf"]의 값을 재정렬하기

        node_vertex_attr = np.zeros((self.node_num, 4))
        for _, row in dfs["nodeVertexDf"].iterrows():
            node_idx = self._node_name_to_idx(row["hostname"])
            node_vertex_attr[node_idx] = row[["cpu_util", "mem_util", "receive_bytes", "transmit_bytes"]]

        # src, dst를 기반으로 dfs["nodeEdgeDf"]의 값을 node_edge_attr에 채우기
        node_edge_index = np.array([[i, j] for i in range(self.node_num) for j in range(self.node_num)])
        node_edge_attr = np.zeros((self.node_edge_num, 3))
        for _, row in dfs["nodeEdgeDf"].iterrows():
            edge_idx = self._node_edge_to_idx(row["src"], row["dst"])
            node_edge_attr[edge_idx] = row[["receive_bytes", "transmit_bytes", "latency_microseconds"]]

        # self.service_names을 기준으로 dfs["serviceVertexDf"]의 값을 재정렬하기
        service_vertex_attr = np.zeros((self.service_num, 4))
        for _, row in dfs["serviceVertexDf"].iterrows():
            service_idx = self._service_name_to_idx(row["name"])
            service_vertex_attr[service_idx] = row[["cpu_util", "mem_util", "receive_bytes", "transmit_bytes"]]
        
        # src, dst를 기반으로 dfs["serviceEdgeDf"]의 값을 service_edge_attr에 채우기
        service_edge_index = np.array([[i, j] for i in range(self.service_num) for j in range(self.service_num)])
        service_edge_attr = np.zeros((self.service_edge_num, 3))
        for _, row in dfs["serviceEdgeDf"].iterrows():
            edge_idx = self._service_edge_to_idx(row["src"], row["dst"])
            service_edge_attr[edge_idx] = row[["receive_bytes", "transmit_bytes", "duration"]]
        
        # self.max_pod_num으로 pod_vertex_attr를 만들고, 위에서부터 dfs["podVertexDf"]로 채우기
        pod_vertex_attr = np.zeros((self.max_pod_num, 4))
        pod_vertex_attr[:dfs["podVertexDf"].shape[0]] = dfs["podVertexDf"][["cpu_util", "mem_util", "receive_bytes", "transmit_bytes"]].values
        
        def _pod_name_to_idx(pod_name: str): return dfs["podVertexDf"][dfs["podVertexDf"]["name"] == pod_name].index[0]

        # pod, service를 기반으로 dfs["serviceSelectPodEdgeDf"]의 값을 pod_service_edge에 채우기
        pod_service_edge = np.zeros((self.max_pod_num, 2))
        for _, row in dfs["serviceSelectPodEdgeDf"].iterrows():
            try:
                pod_idx = _pod_name_to_idx(row["pod"])
                service_idx = self._service_name_to_idx(row["service"])
                pod_service_edge[pod_idx] = [pod_idx, service_idx]
            except Exception as e:
                print("pod(%s) is not found in podVertexDf." % row["pod"])
            

        pod_node_edge = np.zeros((self.max_pod_num, 2))
        for _, row in dfs["podInNodeEdgeDf"].iterrows():
            try:
                pod_idx = _pod_name_to_idx(row["pod"])
                node_idx = self._node_name_to_idx(row["node"])
                pod_node_edge[pod_idx] = [pod_idx, node_idx]
            except Exception as e:
                print("pod(%s) is not found in podVertexDf." % row["pod"])
        
        observation = {
            "node_vertex_attr": np.nan_to_num( node_vertex_attr.astype(np.float32)),
            "node_edge_attr": np.nan_to_num(node_edge_attr.astype(np.float32)),
            "node_edge_index": np.nan_to_num(self.format_index_to_multidiscrete(node_edge_index.astype(np.int64))),
            "service_vertex_attr": np.nan_to_num(service_vertex_attr.astype(np.float32)),
            "service_edge_attr": np.nan_to_num(service_edge_attr.astype(np.float32)),
            "service_edge_index": np.nan_to_num(self.format_index_to_multidiscrete(service_edge_index.astype(np.int64))),
            "pod_vertex_attr": np.nan_to_num(pod_vertex_attr.astype(np.float32)),
            "pod_service_edge": np.nan_to_num(self.format_index_to_multidiscrete(pod_service_edge.astype(np.int64))),
            "pod_node_edge": np.nan_to_num(self.format_index_to_multidiscrete(pod_node_edge.astype(np.int64))),
            "target_service_idx": np.nan_to_num(target_service_idx),
        }

        return observation, dfs
    
    def _wait_request_from_client(self):
        req, queue = self.server.wait_request()
        self.queue = queue
        return req
    
    def _send_response(self, action):
        action = np.array([a for a, m in zip(action, self.mask) if m == 1])
        # action = self.choose_prob_action(action)
        action = self.choose_max_action(action)

        self.server.send_response(action, self.queue)

    def choose_max_action(self, action):
        max_idx = action.argmax()
        action = [0 if i != max_idx else 1 for i in range(len(action))]
        return action
    
    def choose_prob_action(self, action):
        if action.sum() != 1:
            action = softmax(action)
        return sample_from_prob(action)

    def _nodenames_to_nodelist(self, nodeNames):
        return [self.node_names.index(nodeName) for nodeName in nodeNames]

    def _get_availability(self, dfs):
        # inner join podInNodeEdgeDf and serviceSelectPodEdgeDf with pod name.
        df = dfs["podInNodeEdgeDf"].merge(dfs["serviceSelectPodEdgeDf"], on="pod")
        # count unique node num
        node_num = len(df["node"].unique())
        # count unique service num
        service_num = len(df["service"].unique())
        # count each service's located node num
        service_node_num = df.groupby("service")["node"].nunique()

        availability = service_node_num.sum() / (node_num * service_num)

        return availability


        

if __name__ == "__main__":
    namespace = "rl-testbed"
    observation_duration = 30
    k8sApi = initK8sApi(config_file='/home/dpnm/projects/rl-server/kube-config.yaml')
    promApi = initPromApi(url='http://sfc-testbed.duckdns.org:31237/prometheus', disable_ssl=True)
    default_tg = TrafficGenerator("http://sfc-testbed.duckdns.org:31237/sfc-e2e-collector")
    step_tg = TrafficGenerator("http://sfc-testbed.duckdns.org:31237/sfc-e2e-collector")
    vnfs = create_vnfs(k8sApi, namespace)

    env = ClusterEnv(k8sApi, promApi, default_tg, step_tg, vnfs, namespace, observation_duration)
    
    