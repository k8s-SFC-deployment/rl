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
from src.utils import softmax, sample_from_prob

from src.env.env import ClusterEnv

class BaslineClusterEnv(ClusterEnv):
    """Cluster Environment that follows gym interface."""

    metadata = {"render_modes": None}
   
    default_tg_key = "default"
    step_tg_key = "step"

    def __init__(self, k8sApi: K8sApi, promApi: PromApi, default_tg: TrafficGenerator, step_tg: TrafficGenerator,
                 vnfs: List[VNF], namespace: str, observation_duration: int):
        super().__init__(k8sApi, promApi, default_tg, step_tg, vnfs, namespace, observation_duration)

    def _init_namespace(self):
        self.k8sApi.createNamespace(self.namespace)
        for vnf in self.vnfs:
            vnf.create()
        print("Wait until services are ready.")
        for vnf in self.vnfs:
           self.k8sApi.isAtleastOnePodReadyInDeployment(vnf.name, self.namespace)
        self.default_tg_thread = threading.Thread(target=self.default_tg.generate, args=(self.default_tg_key, [f"http://{vnf.name}.{self.namespace}.svc/loadv2" for vnf in self.vnfs], [1], -1))
        self.default_tg_thread.start()

    def _wait_request_from_client(self): pass

    def _send_response(self, action): pass


if __name__ == "__main__":
    namespace = "rl-testbed"
    observation_duration = 30
    k8sApi = initK8sApi(config_file='/home/dpnm/projects/rl-server/kube-config.yaml')
    promApi = initPromApi(url='http://sfc-testbed.duckdns.org:31237/prometheus', disable_ssl=True)
    default_tg = TrafficGenerator("http://sfc-testbed.duckdns.org:31237/sfc-e2e-collector")
    step_tg = TrafficGenerator("http://sfc-testbed.duckdns.org:31237/sfc-e2e-collector")
    vnfs = create_vnfs(k8sApi, namespace)

    env = BaslineClusterEnv(k8sApi, promApi, default_tg, step_tg, vnfs, namespace, observation_duration)
    check_env(env)