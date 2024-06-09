from src.traffic_generator import TrafficGenerator
from src.env.env import ClusterEnv
from src.env.vnfs.factory import create_vnfs
from src.apis.k8s import initApi as initK8sApi
from src.apis.prom import initApi as initPromApi
from src.td3.model import GNNMATD3

import torch
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.preprocessing import preprocess_obs

url_prefix = "http://euidong.duckdns.org:31418"


if __name__ == "__main__":
    namespace = "rl-testbed"
    k8sApi = initK8sApi(config_file='/home/dpnm/projects/rl/kube-config.yaml')
    promApi = initPromApi(url=f'{url_prefix}/prometheus', disable_ssl=True)
    default_tg = TrafficGenerator(f"{url_prefix}/sfc-e2e-collector")
    step_tg = TrafficGenerator(f"{url_prefix}/sfc-e2e-collector")
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

    model = GNNMATD3(env.observation_space, 16, 1, 1e-3, mem_capacity=1000)

    batch_size = 32

    max_episodes = 50
    tot_step = 1
    save_interval = 100

    for episode in range(max_episodes):
        observations = vec_env.reset()
        for key, value in observations.items():
            observations[key] = torch.tensor(value)
        observations = preprocess_obs(observations, env.observation_space)
        done = False
        step = 1
        while not done:
            action = model.select_action(observations)
            vec_env.step_async(action.detach().numpy())
            next_observations, rewards, done, info = vec_env.step_wait()
            for key, value in next_observations.items():
                next_observations[key] = torch.tensor(value)
            next_observations = preprocess_obs(next_observations, env.observation_space)
            model.save_transition(observations, action, torch.tensor(rewards), next_observations, torch.tensor(done))
            observations = next_observations
            model.update(batch_size)
            if tot_step % save_interval == 1:
                model.save(tot_step)
                vec_env.save(f"{model.path_prefix}_gat_vec_env_{tot_step}.pkl")
            step += 1
            tot_step += 1
            print(f"Episode {episode}, Step {step}, Reward {rewards}, Done {done}, Info {info}")
    model.save("final")
    vec_env.save(f"{model.path_prefix}_gat_vec_env_final.pkl")
