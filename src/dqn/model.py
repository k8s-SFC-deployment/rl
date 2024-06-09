import random
from gymnasium import spaces
from torch import nn
import torch
from typing import Dict, List

from src.dqn.dqn_fe import DQNFE
from src.dqn.replay_buffer import ReplayBuffer

class QNetwork(nn.Module):
    def __init__(self, observation_space: spaces.Dict, hidden_dims: List[int] = [16, 16, 16]):
        super(QNetwork, self).__init__()
        self.feature_extractor = DQNFE(observation_space, hidden_dims[0], hidden_dims[1])
        self.fc1 = nn.Linear(hidden_dims[1], hidden_dims[2])
        self.fc2 = nn.Linear(hidden_dims[2], 1)

    def forward(self, observations: spaces.Dict):
        x = self.feature_extractor(observations)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.squeeze(-1)
        x = nn.functional.softmax(x, dim=-1)
        
        return x

class DQN(nn.Module):
    def __init__(self, observation_space: spaces.Dict, hidden_dims: List[int] = [16, 16, 16]):
        super(DQN, self).__init__()
        self.q_network = QNetwork(observation_space, hidden_dims)
        self.target_q_network = QNetwork(observation_space, hidden_dims)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = torch.optim.Adam(self.q_network.parameters())
        self.memory = ReplayBuffer(10000)
        self.node_n = observation_space["node_vertex_attr"].shape[0]

    def forward(self, observations: spaces.Dict):
        return self.q_network(observations)

    # batch_size must be 1.
    # epsilon-greedy policy.
    def act(self, observations: spaces.Dict, epsilon: float = 0.1):
        if random.random() < epsilon:
            idx = random.randint(0, self.node_n - 1)
            action = torch.zeros(self.node_n)
            action[idx] = 1
            return action.unsqueeze(0)
        q_values = self.q_network(observations)
        idx = q_values.argmax(1)
        action = torch.zeros(self.node_n)
        action[idx] = 1
        return action.unsqueeze(0)

    def target_forward(self, observations: spaces.Dict):
        return self.target_q_network(observations)
    
    def update(self, batch_size: int, gamma: float):
        if len(self.memory) < batch_size:
            return
        state, action, reward, next_state, done = self.memory.sample(batch_size)
        
        done = torch.tensor(done).float()

        q_values = self.q_network(state)
        with torch.no_grad():
            next_q_values = self.target_q_network(next_state)
            next_q_value = next_q_values.max(1).values
        action_idx = action.argmax(1).unsqueeze(1)
        q_value = q_values.gather(1, action_idx).squeeze(1)
        expected_q_value = reward + gamma * next_q_value * (1 - done)
        
        loss = nn.functional.mse_loss(q_value, expected_q_value)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def update_target(self):
        self.target_q_network.load_state_dict(self.q_network.state_dict())

    def save(self, path):
        torch.save(self.q_network.state_dict(), path)

    def load(self, path):
        self.q_network.load_state_dict(torch.load(path))
        self.update_target()
    