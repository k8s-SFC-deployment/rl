import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces

from src.fe.all_gnn_v2 import ALLGNNFE
from src.sac.replay_buffer import ReplayBuffer

torch.autograd.set_detect_anomaly(True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
min_Val = torch.tensor(1e-7).float().to(device)

# Actor(s_dim=16, a_dim=4, hidden_dims=[32, 32])
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=[], min_log_std=-10, max_log_std=2):
        super(Actor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std

        self.lins = nn.Sequential()
        dims = [state_dim] + hidden_dims
        for i in range(1, len(dims)):
            self.lins.add_module(f'linear{i}', nn.Linear(dims[i-1], dims[i]))
            self.lins.add_module(f'relu{i}', nn.ReLU())
        self.mu_head = nn.Linear(dims[-1], action_dim)
        self.log_std_head = nn.Linear(dims[-1], action_dim)

    # input :
    #   - x : (Batch, Node, state_dim)
    # output:
    #   - mu : (Batch, Node)
    #   - log_std : (Batch, Node)
    def forward(self, x):
        x = self.lins(x)
        mu = self.mu_head(x)
        log_std = F.relu(self.log_std_head(x), inplace=False)
        log_std_clamp = torch.clamp(log_std, self.min_log_std, self.max_log_std)
        
        return mu.squeeze(-1), log_std_clamp.squeeze(-1)

# Critic(s_dim=16, hidden_dims=[32, 32])
class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dims=[]):
        super(Critic, self).__init__()
        self.state_dim = state_dim
        self.hidden_dims = hidden_dims
        self.lins = nn.Sequential()
        dims = [state_dim] + hidden_dims + [1]
        for i in range(1, len(dims)):
            self.lins.add_module(f'linear{i}', nn.Linear(dims[i-1], dims[i]))
            if i != len(dims) - 1:
                self.lins.add_module(f'relu{i}', nn.ReLU())
    
    # input :
    #   - x : (Batch, Node, state_dim)
    # output :
    #   - output : (Batch, Node)
    def forward(self, x): 
        return self.lins(x).squeeze(-1)

class Q(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=[]):
        super(Q, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims
        self.lins = nn.Sequential()
        dims = [state_dim + action_dim] + hidden_dims + [1]
        for i in range(1, len(dims)):
            self.lins.add_module(f'linear{i}', nn.Linear(dims[i-1], dims[i]))
            if i != len(dims) - 1:
                self.lins.add_module(f'relu{i}', nn.ReLU())
    
    # input : 
    #   - x : (Batch, Node, state_dim)
    #   - a : (Batch, Node)
    # output :
    #   - output : (Batch, Node)
    def forward(self, x, a):
        a = a.unsqueeze(2)
        x = torch.cat([x, a], dim=2)
        return self.lins(x).squeeze(-1)

# GNNMASAC(state_dim=4, action_dim=1, learning_rate=1e-3, mem_capacity=1000)
class GNNMASAC(nn.Module):
    def __init__(self, observation_space, state_dim, action_dim, learning_rate, mem_capacity=1000):
        super(GNNMASAC, self).__init__()

        self.observation_space = observation_space
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate

        self.feature_extractor = ALLGNNFE(observation_space, 32, state_dim).to(device)

        self.policy_net = Actor(state_dim, action_dim, [32, 16]).to(device)
        self.value_net = Critic(state_dim, [32, 16]).to(device)
        self.target_value_net = Critic(state_dim, [32, 16]).to(device)
        self.q1_net = Q(state_dim, action_dim, [32, 16]).to(device)
        self.q2_net = Q(state_dim, action_dim, [32, 16]).to(device)

        self.feature_extractor_optimizer = torch.optim.Adam(self.feature_extractor.parameters(), lr=learning_rate)
        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=learning_rate)
        self.q1_optimizer = torch.optim.Adam(self.q1_net.parameters(), lr=learning_rate)
        self.q2_optimizer = torch.optim.Adam(self.q2_net.parameters(), lr=learning_rate)

        self.replay_buffer = ReplayBuffer(mem_capacity)
        self.num_training = 0

        self.value_criterion = nn.MSELoss()
        self.q1_criterion = nn.MSELoss()
        self.q2_criterion = nn.MSELoss()

        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(param.data)

        self.path_prefix = './SAC_model/'
        os.makedirs(self.path_prefix, exist_ok=True)


    def save_transition(self, observations, action, rewards, next_observations, done):
        self.replay_buffer.push(observations, action, rewards, next_observations, done)
    
    # input :
    #   - observation : Dict[str, torch.Tensor]
    # output :
    #   - action : (1, Node)
    def select_action(self, observations):
        state = self.feature_extractor(observations)
        mu, log_sigma = self.policy_net(state)
        sigma = torch.exp(log_sigma)
        dist = torch.distributions.Normal(mu, sigma)
        z = dist.sample()
        action = torch.tanh(z).detach().cpu()
        return action

    def update(self, batch_size, tot_step, gradient_steps=1, gamma=0.99, grad_clip=0.5, tau=0.005):
        if self.replay_buffer.cursor < batch_size:
            return
        for _ in range(gradient_steps):
            obs, act, rew, next_obs, dones = self.replay_buffer.sample(batch_size)
            sts = self.feature_extractor(obs) # (Batch, Node, state_dim)
            next_sts = self.feature_extractor(next_obs) # (Batch, Node, state_dim)

            target_value = self.target_value_net(next_sts) # (Batch, Node)
            # next_q_value = rew + (1 - dones) * gamma * target_value # (Batch, Node)
            # ! (1 - dones) is not suitable for our endless scheduling.
            next_q_value = rew.unsqueeze(1) + gamma * target_value # (Batch, Node)

            expected_value = self.value_net(sts) # (Batch, Node)
            expected_Q1 = self.q1_net(sts, act) # (Batch, Node)
            expected_Q2 = self.q2_net(sts, act) # (Batch, Node)
            # sample_action : (Batch, Node)
            # log_prob : (Batch, Node)
            sample_action, log_prob, z, mu, log_sigma = self._evaluate(sts)

            new_q1 = self.q1_net(sts, sample_action) # (Batch, Node)
            new_q2 = self.q2_net(sts, sample_action) # (Batch, Node)
            
            expected_new_Q = torch.min(new_q1.view(-1), new_q2.view(-1)) # (Batch x Node)
            log_prob = log_prob.view(-1) # (Batch x Node)
            next_value = expected_new_Q - log_prob # (Batch x Node)
            
            expected_value = expected_value.view(-1) # (Batch x Node)
            expected_Q1 = expected_Q1.view(-1) # (Batch x Node)
            expected_Q2 = expected_Q2.view(-1) # (Batch x Node)
            next_q_value = next_q_value.view(-1) # (Batch x Node)

            log_policy_target = expected_new_Q - expected_value

            V_loss = self.value_criterion(expected_value, next_value.detach()).mean()

            Q1_loss = self.q1_criterion(expected_Q1, next_q_value.detach()).mean()
            Q2_loss = self.q2_criterion(expected_Q2, next_q_value.detach()).mean()

            pi_loss = log_prob * (log_prob - log_policy_target).detach()
            pi_loss = pi_loss.mean()

            print('Loss/V_loss', V_loss)
            print('Loss/Q1_loss', Q1_loss)
            print('Loss/Q2_loss', Q2_loss)
            print('Loss/policy_loss', pi_loss)

            self.feature_extractor_optimizer.zero_grad()
            self.value_optimizer.zero_grad()
            self.q1_optimizer.zero_grad()
            self.q2_optimizer.zero_grad()
            self.policy_optimizer.zero_grad()

            V_loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(self.value_net.parameters(), grad_clip)
            Q1_loss.backward(retain_graph = True)
            nn.utils.clip_grad_norm_(self.q1_net.parameters(), grad_clip)
            Q2_loss.backward(retain_graph = True)
            nn.utils.clip_grad_norm_(self.q2_net.parameters(), grad_clip)
            pi_loss.backward(retain_graph = True)
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), grad_clip)
            
            self.feature_extractor_optimizer.step()
            self.policy_optimizer.step()
            self.value_optimizer.step()
            self.q1_optimizer.step()
            self.q2_optimizer.step()


            for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
                target_param.data.copy_(target_param * (1 - tau) + param * tau)
            
            self.num_training += 1


    def save(self, step):
        torch.save(self.feature_extractor.state_dict(), os.path.join(self.path_prefix, f'feature_extractor{step}.pth'))
        torch.save(self.policy_net.state_dict(), os.path.join(self.path_prefix, f'policy_net{step}.pth'))
        torch.save(self.value_net.state_dict(), os.path.join(self.path_prefix, f'value_net{step}.pth'))
        torch.save(self.q1_net.state_dict(), os.path.join(self.path_prefix, f'q1_net{step}.pth'))
        torch.save(self.q2_net.state_dict(), os.path.join(self.path_prefix, f'q2_net{step}.pth'))

    def load(self, step):
        self.feature_extractor.load_state_dict(torch.load(os.path.join(self.path_prefix, f'feature_extractor{step}.pth')))
        self.policy_net.load_state_dict(torch.load(os.path.join(self.path_prefix, f'policy_net{step}.pth')))
        self.value_net.load_state_dict(torch.load(os.path.join(self.path_prefix, f'value_net{step}.pth')))
        self.q1_net.load_state_dict(torch.load(os.path.join(self.path_prefix, f'q1_net{step}.pth')))
        self.q2_net.load_state_dict(torch.load(os.path.join(self.path_prefix, f'q2_net{step}.pth')))

    # input :
    #   - state : (Batch, Node, state_dim)
    # output : 
    #   - action : (Batch, Node)
    #   - log_prob : (Batch, Node)
    #   - z : 1,
    #   - mu : (Batch, Node)
    #   - log_sigma : (Batch, Node)
    def _evaluate(self, state):
        mu, log_sigma = self.policy_net(state) # (Batch, Node), (Batch, Node)
        sigma = torch.exp(log_sigma) # (Batch, Node)
        dist = torch.distributions.Normal(mu, sigma)
        noise = torch.distributions.Normal(0, 1)

        z = noise.sample() # Scalar
        action = torch.tanh(mu + sigma * z).cpu() # (Batch, Node)
        log_prob = dist.log_prob(mu + sigma * z.to(device)) - torch.log(1 - action.pow(2) + min_Val)
        return action, log_prob, z, mu, log_sigma