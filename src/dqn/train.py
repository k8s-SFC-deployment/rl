import torch
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.preprocessing import preprocess_obs

from src.traffic_generator import TrafficGenerator
from src.env.env import ClusterEnv
from src.env.vnfs.factory import create_vnfs
from src.apis.k8s import initApi as initK8sApi
from src.apis.prom import initApi as initPromApi

from src.dqn.model import DQN

if __name__ == "__main__":
    namespace = "rl-testbed"
    k8sApi = initK8sApi(config_file='/home/dpnm/projects/rl-server/kube-config.yaml')
    promApi = initPromApi(url='http://sfc-testbed.duckdns.org:31237/prometheus', disable_ssl=True)
    default_tg = TrafficGenerator("http://sfc-testbed.duckdns.org:31237/sfc-e2e-collector")
    step_tg = TrafficGenerator("http://sfc-testbed.duckdns.org:31237/sfc-e2e-collector")
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

    model = DQN(env.observation_space)

    batch_size=32

    max_episodes = 50
    tot_step = 1
    target_update_interval = 32
    def calc_epsilon(): 
        if tot_step < batch_size:
            return 1.0
        return 1.0 - (tot_step - batch_size / (max_episodes * env.step_n) - batch_size)
    for episode in range(max_episodes):
        observations = vec_env.reset()
        for key, value in observations.items():
            observations[key] = torch.tensor(value)
        observations = preprocess_obs(observations, env.observation_space)
        done = False
        step = 1
        while not done:
            action = model.act(observations, epsilon=calc_epsilon())
            vec_env.step_async(action.detach().numpy())
            next_observations, rewards, done, info = vec_env.step_wait()
            print(f"Episode: {episode} Step: {step}")
            print(f"Observations: {observations}")
            print(f"Action: {action}")
            print(f"Rewards: {rewards}")
            print(f"Done: {done}")
            print(f"Info: {info}")
            print()
            model.update(batch_size, 0.9)
            for key, value in next_observations.items():
                next_observations[key] = torch.tensor(value)
            next_observations = preprocess_obs(next_observations, env.observation_space)
            model.memory.push(observations, action, rewards, next_observations, done)
            observations = next_observations
            if tot_step % target_update_interval == 0:
                model.update_target()
                model.save(f"DQN_model/model_{tot_step}.pth")
                vec_env.save(f"DQN_model/vec_env_{tot_step}.pkl")
                # model.memory.save(f"DQN_model/replay_buffer_{tot_step}.json")
            step += 1
            tot_step += 1