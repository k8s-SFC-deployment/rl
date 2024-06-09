from src.traffic_generator import TrafficGenerator
from src.env.env import ClusterEnv
from src.env.vnfs.factory import create_vnfs
from src.apis.k8s import initApi as initK8sApi
from src.apis.prom import initApi as initPromApi
from src.sac.model import GNNMASAC
# from src.td3.model import GNNMATD3

import json
import torch
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.preprocessing import preprocess_obs

if __name__ == "__main__":
    namespace = "rl-testbed"
    k8sApi = initK8sApi(config_file='/home/dpnm/projects/rl/kube-config.yaml')
    promApi = initPromApi(url='http://<prometheus-url>', disable_ssl=True)
    default_tg = TrafficGenerator("http://<sfc-e2e-collector-url>")
    step_tg = TrafficGenerator("http://<sfc-e2e-collector-url>")
    vnfs = create_vnfs(k8sApi, namespace)

    env = ClusterEnv(k8sApi, promApi, default_tg, step_tg, vnfs, namespace, observation_duration=60)

    vec_env = DummyVecEnv([lambda: env])
    vec_env = VecNormalize( # normalize features and reward.
        vec_env, 
        norm_obs=True,
        norm_reward=True,
        clip_obs=10., # max value of observation features.
        norm_obs_keys=[ # except "*_index".
            "node_vertex_attr",
            "node_edge_attr",
            "service_vertex_attr",
            "service_edge_attr",
            "pod_vertex_attr",
        ],
    )

    model = GNNMASAC(env.observation_space, 16, 1, 1e-3, mem_capacity=1000)
    # model = GNNMATD3(env.observation_space, 16, 1, 1e-3, 1000)

    model.load("501")
    vec_env.load(f"{model.path_prefix}vec_env_501.pkl", vec_env)

    all_infos = []
    all_rewards = []
    for episode in range(1, 11):
        episode_reward = []
        episode_info = []
        done = False
        observations = vec_env.reset()
        for key, value in observations.items():
            observations[key] = torch.tensor(value)
        observations = preprocess_obs(observations, env.observation_space)
        # episode_info.append(info)
        step = 1
        while not done:
            action = model.select_action(observations)
            vec_env.step_async(action.detach().numpy())
            next_observations, rewards, done, info = vec_env.step_wait()
            print(f"Episode: {episode} Step: {step}")
            print(f"Observations: {observations}")
            print(f"Action: {action}")
            print(f"Rewards: {rewards}")
            print(f"Done: {done}")
            print(f"Info: {info}")

            for key, value in next_observations.items():
                next_observations[key] = torch.tensor(value)
            next_observations = preprocess_obs(next_observations, env.observation_space)
            observations = next_observations

            episode_info.append(info)
            episode_reward.append(rewards)
            step += 1
        all_rewards.append(episode_reward)
        all_infos.append(episode_info)
    
    for infos in all_infos:
        for info in infos:
            if "terminal_observation" in info[0].keys():
                del info[0]["terminal_observation"]
    for idx, rewards in enumerate(all_rewards):
        for iidx, reward in enumerate(rewards):
            all_rewards[idx][iidx] = reward.tolist()
    json.dump(all_rewards, open("rewards.json", "w"))
    json.dump(all_infos, open("infos.json", "w"))