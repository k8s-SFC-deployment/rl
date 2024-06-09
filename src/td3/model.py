import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces

from src.td3.replay_buffer import ReplayBuffer
from src.fe.all_gnn_v3 import ALLGNNFE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    def __init__(self, state_dim, hidden_dims=[]):
        super(Actor, self).__init__()

        self.state_dim = state_dim
        self.hidden_dims = hidden_dims

        self.lins = nn.Sequential()
        dims = [state_dim] + hidden_dims + [1]
        for i in range(1, len(dims)):
            self.lins.add_module(f'linear{i}', nn.Linear(dims[i-1], dims[i]))
            if i != len(dims) - 1:
                self.lins.add_module(f'relu{i}', nn.ReLU())
            else:
                self.lins.add_module(f'tanh{len(dims) + 1}', nn.Tanh())

    # input : 
    #   - state : (Batch, Node, state_dim)
    # output :
    #   - output : (Batch, Node)
    def forward(self, state):
        output = self.lins(state) # (Batch, Node, 1)
        return output.squeeze(-1)

class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dims=[]):
        super(Critic, self).__init__()

        self.state_dim = state_dim
        self.hidden_dims = hidden_dims
        self.lins = nn.Sequential()
        dims = [state_dim + 1] + hidden_dims + [1]
        for i in range(1, len(dims)):
            self.lins.add_module(f'linear{i}', nn.Linear(dims[i-1], dims[i]))
            if i != len(dims) - 1:
                self.lins.add_module(f'relu{i}', nn.ReLU())
    
    # input : 
    #   - state : (Batch, Node, state_dim)
    #   - action : (Batch, Node)
    # output :
    #   - output : (Batch, Node)
    def forward(self, state, action):
        action = action.unsqueeze(2) # (Batch, Node, 1)
        state_action = torch.cat([state, action], dim=2) # (Batch, Node, state_dim + 1)
        output = self.lins(state_action) # (Batch, Node, 1)
        return output.squeeze(-1) # (Batch, Node)

class GNNMATD3(nn.Module):
    def __init__(self, observation_space, state_dim, action_dim, learning_rate, mem_capacity=1000):
        super(GNNMATD3, self).__init__()

        self.observation_space = observation_space
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.mem_capacity = mem_capacity

        self.feature_extractor = ALLGNNFE(observation_space, 32, state_dim).to(device)

        self.actor = Actor(state_dim, [32, 16]).to(device)
        self.actor_target = Actor(state_dim, [32, 16]).to(device)

        self.critic_1 = Critic(state_dim, [32, 16]).to(device)
        self.critic_1_target = Critic(state_dim, [32, 16]).to(device)

        self.critic_2 = Critic(state_dim, [32, 16]).to(device)
        self.critic_2_target = Critic(state_dim, [32, 16]).to(device)

        self.feature_extractor_optimizer = optim.Adam(self.feature_extractor.parameters(), lr=learning_rate)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=learning_rate)
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=learning_rate)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())

        self.replay_buffer = ReplayBuffer(mem_capacity)
        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0
        self.num_training = 0

        self.path_prefix = './TD3_model/'
        os.makedirs(self.path_prefix, exist_ok=True)

    def save_transition(self, observation, action, reward, next_observation, done):
        self.replay_buffer.push(observation, action, reward, next_observation, done)

    def select_action(self, observation):
        state = self.feature_extractor(observation)
        action = self.actor(state)
        return action

    def update(self, batch_size, num_iteration = 1, policy_noise = 0.2, noise_clip = 0.5, gamma = 0.99, policy_delay = 2, tau=0.005):
        if self.replay_buffer.cursor < batch_size:
            return
        for i in range(num_iteration):
            obs, act, rew, next_obs, dones = self.replay_buffer.sample(batch_size)
            
            sts = self.feature_extractor(obs) # (Batch, Node, state_dim)
            next_sts = self.feature_extractor(next_obs) # (Batch, Node, state_dim)

            # Select next action according to target policy:
            noise = torch.ones_like(act).data.normal_(0, policy_noise).to(device) # (Batch, Node)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (self.actor_target(next_sts) + noise) # (Batch, Node)
            

            # Compute target Q-value:
            target_Q1 = self.critic_1_target(next_sts, next_action) # (Batch, Node)
            target_Q2 = self.critic_2_target(next_sts, next_action) # (Batch, Node)
            target_Q = torch.min(target_Q1, target_Q2) # (Batch, Node)
            # target_Q = rew + ((1 - done) * args.gamma * target_Q).detach()
            # ! (1 - dones) is not suitable for our environment.
            target_Q = rew.unsqueeze(1).detach() + (gamma * target_Q)

            # Optimize Critic 1:
            current_Q1 = self.critic_1(sts, act) # (Batch, Node)
            loss_Q1 = F.mse_loss(current_Q1, target_Q.detach())
            print('Loss/Q1_loss: ', loss_Q1)

            # Optimize Critic 2:
            current_Q2 = self.critic_2(sts, act) # (Batch, Node)
            loss_Q2 = F.mse_loss(current_Q2, target_Q.detach())
            print('Loss/Q2_loss: ', loss_Q2)
            

            self.feature_extractor_optimizer.zero_grad()
            self.critic_1_optimizer.zero_grad()
            self.critic_2_optimizer.zero_grad()

            loss_Q1.backward(retain_graph = True)
            loss_Q2.backward()
            
            self.feature_extractor_optimizer.step()
            self.critic_1_optimizer.step()
            self.critic_2_optimizer.step()
            
            # Delayed policy updates:
            if i % policy_delay == 0:
                # Compute actor loss:
                actor_loss = - self.critic_1(sts.detach(), self.actor(sts.detach())).mean() # Scalar

                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                print('Loss/actor_loss: ', actor_loss)

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(((1- tau) * target_param.data) + tau * param.data)

                for param, target_param in zip(self.critic_1.parameters(), self.critic_1_target.parameters()):
                    target_param.data.copy_(((1 - tau) * target_param.data) + tau * param.data)

                for param, target_param in zip(self.critic_2.parameters(), self.critic_2_target.parameters()):
                    target_param.data.copy_(((1 - tau) * target_param.data) + tau * param.data)

                self.num_actor_update_iteration += 1
        self.num_critic_update_iteration += 1
        self.num_training += 1

    def save(self, step):
        torch.save(self.feature_extractor.state_dict(), os.path.join(self.path_prefix, f'feature_extractor_{step}.pth'))
        torch.save(self.actor.state_dict(), os.path.join(self.path_prefix, f'actor_{step}.pth'))
        torch.save(self.actor_target.state_dict(), os.path.join(self.path_prefix,  f"actor_target_{step}.pth"))
        torch.save(self.critic_1.state_dict(), os.path.join(self.path_prefix,  f"critic_1_{step}.pth"))
        torch.save(self.critic_1_target.state_dict(), os.path.join(self.path_prefix,  f"critic_1_target_{step}.pth"))
        torch.save(self.critic_2.state_dict(), os.path.join(self.path_prefix,  f"critic_2_{step}.pth"))
        torch.save(self.critic_2_target.state_dict(), os.path.join(self.path_prefix,  f"critic_2_target_{step}.pth"))
        print("====================================")
        print("Model has been saved...")
        print("====================================")

    def load(self, step):
        self.feature_extractor.load_state_dict(torch.load(os.path.join(self.path_prefix, f"feature_extractor_{step}.pth")))
        self.actor.load_state_dict(torch.load(os.path.join(self.path_prefix, f'actor_{step}.pth')))
        self.actor_target.load_state_dict(torch.load(os.path.join(self.path_prefix, f"actor_target_{step}.pth")))
        self.critic_1.load_state_dict(torch.load(os.path.join(self.path_prefix, f"critic_1_{step}.pth")))
        self.critic_1_target.load_state_dict(torch.load(os.path.join(self.path_prefix, f"critic_1_target_{step}.pth")))
        self.critic_2.load_state_dict(torch.load(os.path.join(self.path_prefix, f"critic_2_{step}.pth")))
        self.critic_2_target.load_state_dict(torch.load(os.path.join(self.path_prefix, f"critic_2_target_{step}.pth")))

        print("====================================")
        print("model has been loaded...")
        print("====================================")