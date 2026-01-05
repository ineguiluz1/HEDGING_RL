"""
Recurrent TD3 (RTD3) Agent for Hedging
Implementation handling partial observability with LSTM/GRU layers
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from collections import deque
from config import CONFIG, get_model_config, get_exploration_config

# Set device
device = torch.device("cuda" if torch.cuda.is_available() and CONFIG.get("use_gpu", False) else "cpu")

class RecurrentReplayBuffer:
    """Experience Replay Buffer for Recurrent TD3
    Stores full episodes to allow sampling sequences.
    """
    
    def __init__(self, capacity=None, seq_len=None):
        if capacity is None:
            capacity = CONFIG.get("replay_buffer_size", 100000)
        self.capacity = capacity
        self.seq_len = seq_len if seq_len is not None else CONFIG.get("seq_len", 20)
        
        self.buffer = deque(maxlen=capacity) # Stores episodes
        self.current_episode = []
        self.total_transitions = 0
    
    def push(self, state, action, reward, next_state, done):
        """Add experience to current episode"""
        self.current_episode.append((state, action, reward, next_state, done))
        
        if done:
            # Store full episode
            self.buffer.append(list(self.current_episode))
            self.total_transitions += len(self.current_episode)
            self.current_episode = []
            
    def sample(self, batch_size=None):
        """Sample a batch of sequences"""
        if batch_size is None:
            batch_size = CONFIG.get("batch_size", 64)
            
        # Filter episodes that are long enough
        # Optimization: We could maintain a separate list of valid episodes, 
        # but for now we filter on the fly as the buffer size (in episodes) is usually manageable.
        valid_episodes = [ep for ep in self.buffer if len(ep) >= self.seq_len]
        
        if not valid_episodes:
            return None
            
        sampled_episodes = random.choices(valid_episodes, k=batch_size)
        
        batch_states = []
        batch_actions = []
        batch_rewards = []
        batch_next_states = []
        batch_dones = []
        
        for ep in sampled_episodes:
            # Sample a random start index for the sequence
            start_idx = random.randint(0, len(ep) - self.seq_len)
            seq = ep[start_idx : start_idx + self.seq_len]
            
            # Extract data from sequence
            batch_states.append(np.array([e[0] for e in seq]))
            batch_actions.append(np.array([e[1] for e in seq]))
            batch_rewards.append(np.array([e[2] for e in seq]))
            batch_next_states.append(np.array([e[3] for e in seq]))
            batch_dones.append(np.array([e[4] for e in seq]))
            
        # Convert to tensors: (batch_size, seq_len, dim)
        # Note: We use float32 for all tensors
        states = torch.FloatTensor(np.array(batch_states)).to(device)
        actions = torch.FloatTensor(np.array(batch_actions)).to(device)
        rewards = torch.FloatTensor(np.array(batch_rewards)).unsqueeze(2).to(device)
        next_states = torch.FloatTensor(np.array(batch_next_states)).to(device)
        dones = torch.FloatTensor(np.array(batch_dones)).unsqueeze(2).to(device)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)

class RecurrentActor(nn.Module):
    """Recurrent Actor Network - Outputs continuous action"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=None, max_action=None):
        super(RecurrentActor, self).__init__()
        
        if hidden_dim is None:
            hidden_dim = CONFIG.get("hidden_dim", 256)
        if max_action is None:
            max_action = CONFIG.get("max_action", 1.0)
            
        self.max_action = max_action
        self.hidden_dim = hidden_dim
        
        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state, hidden_state=None):
        # state shape: (batch_size, seq_len, state_dim)
        
        x = F.relu(self.l1(state))
        
        # LSTM
        # x shape: (batch_size, seq_len, hidden_dim)
        # hidden_state is (h_0, c_0)
        x, next_hidden_state = self.lstm(x, hidden_state)
        
        x = F.relu(self.l2(x))
        x = torch.tanh(self.l3(x)) * self.max_action # Use tanh for bounded actions
        
        return x, next_hidden_state

class RecurrentCritic(nn.Module):
    """Recurrent Twin Critic Networks"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=None):
        super(RecurrentCritic, self).__init__()
        
        if hidden_dim is None:
            hidden_dim = CONFIG.get("hidden_dim", 256)
            
        self.hidden_dim = hidden_dim
        
        # Q1 Architecture
        self.l1_1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.lstm1 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.l1_2 = nn.Linear(hidden_dim, hidden_dim)
        self.l1_3 = nn.Linear(hidden_dim, 1)
        
        # Q2 Architecture
        self.l2_1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.l2_2 = nn.Linear(hidden_dim, hidden_dim)
        self.l2_3 = nn.Linear(hidden_dim, 1)
        
    def forward(self, state, action, hidden_state1=None, hidden_state2=None):
        sa = torch.cat([state, action], dim=-1)
        
        # Q1
        q1 = F.relu(self.l1_1(sa))
        q1, next_hidden1 = self.lstm1(q1, hidden_state1)
        q1 = F.relu(self.l1_2(q1))
        q1 = self.l1_3(q1)
        
        # Q2
        q2 = F.relu(self.l2_1(sa))
        q2, next_hidden2 = self.lstm2(q2, hidden_state2)
        q2 = F.relu(self.l2_2(q2))
        q2 = self.l2_3(q2)
        
        return q1, q2, next_hidden1, next_hidden2
    
    def Q1(self, state, action, hidden_state=None):
        sa = torch.cat([state, action], dim=-1)
        
        q1 = F.relu(self.l1_1(sa))
        q1, next_hidden = self.lstm1(q1, hidden_state)
        q1 = F.relu(self.l1_2(q1))
        q1 = self.l1_3(q1)
        
        return q1, next_hidden

class OUNoise:
    """Ornstein-Uhlenbeck Process for exploration noise"""
    def __init__(self, action_dim, theta=0.15, sigma=0.2, mu=0.0):
        self.action_dim = action_dim
        self.theta = theta
        self.sigma = sigma
        self.mu = mu
        self.state = np.ones(action_dim) * mu
        
    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu
    
    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.action_dim)
        self.state += dx
        return self.state

class RTD3Agent:
    """Recurrent TD3 Agent"""
    
    def __init__(self, state_dim, action_dim, config=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Load configuration
        model_config = get_model_config()
        exploration_config = get_exploration_config()
        
        if config is not None:
            model_config.update(config)
            
        self.max_action = model_config.get("max_action", 1.0)
        self.tau = model_config.get("tau", 0.005)
        self.gamma = model_config.get("gamma", 0.99)
        self.policy_noise = model_config.get("policy_noise", 0.2)
        self.noise_clip = model_config.get("noise_clip", 0.5)
        self.policy_freq = model_config.get("policy_freq", 2)
        self.batch_size = model_config.get("batch_size", 64)
        self.seq_len = model_config.get("seq_len", 20)
        
        # Exploration
        self.initial_noise = exploration_config.get("initial_noise", 0.1)
        self.final_noise = exploration_config.get("final_noise", 0.0)
        self.noise_decay_steps = exploration_config.get("noise_decay_steps", 100000)
        self.min_noise = exploration_config.get("min_noise", 0.0)
        self.current_noise = self.initial_noise
        
        # Networks
        hidden_dim = model_config.get("hidden_dim", 256)
        
        self.actor = RecurrentActor(state_dim, action_dim, hidden_dim, self.max_action).to(device)
        self.actor_target = RecurrentActor(state_dim, action_dim, hidden_dim, self.max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        self.critic = RecurrentCritic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target = RecurrentCritic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=model_config.get("actor_lr", 3e-4))
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=model_config.get("critic_lr", 3e-4))
        
        # Replay Buffer
        self.replay_buffer = RecurrentReplayBuffer(model_config.get("replay_buffer_size", 100000), self.seq_len)
        
        # Noise
        self.ou_noise = OUNoise(action_dim)
        
        # Hidden states for inference
        self.actor_hidden = None
        
        self.total_steps = 0
        self.update_counter = 0
        
        self.actor_losses = []
        self.critic_losses = []
        
    def select_action(self, state, add_noise=True):
        """Select action with hidden state maintenance"""
        # state: (state_dim,) -> (1, 1, state_dim) for LSTM
        state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(device)
        
        self.actor.eval()
        with torch.no_grad():
            action, self.actor_hidden = self.actor(state_tensor, self.actor_hidden)
            action = action.cpu().numpy().flatten()
        self.actor.train()
        
        if add_noise:
            noise = self.ou_noise.sample() * self.current_noise
            action = action + noise
            
        return np.clip(action, -self.max_action, self.max_action)
    
    def reset_hidden_state(self):
        """Reset hidden state at start of episode"""
        self.actor_hidden = None
        self.ou_noise.reset()
        
    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)
        
    def update_noise(self):
        if self.total_steps < self.noise_decay_steps:
            decay_ratio = self.total_steps / self.noise_decay_steps
            self.current_noise = self.initial_noise - (self.initial_noise - self.final_noise) * decay_ratio
        else:
            self.current_noise = max(self.final_noise, self.min_noise)
            
    def train_step(self):
        batch = self.replay_buffer.sample(self.batch_size)
        if batch is None:
            return None, None
            
        states, actions, rewards, next_states, dones = batch
        
        self.total_steps += 1
        self.update_noise()
        
        # Target Q calculation
        with torch.no_grad():
            # Add noise to target action
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            
            # Target Actor: we need to pass hidden states. 
            # For simplicity in training, we initialize hidden states to zero (None) at start of sequence.
            # This is an approximation.
            next_actions, _ = self.actor_target(next_states)
            next_actions = (next_actions + noise).clamp(-self.max_action, self.max_action)
            
            # Target Critics
            target_q1, target_q2, _, _ = self.critic_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + (1 - dones) * self.gamma * target_q
            
        # Current Q
        current_q1, current_q2, _, _ = self.critic(states, actions)
        
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        actor_loss = None
        self.update_counter += 1
        
        if self.update_counter % self.policy_freq == 0:
            # Actor update
            # We need to re-compute actions with current actor to get gradients
            new_actions, _ = self.actor(states)
            
            # Get Q1 for these actions
            q1_pred, _ = self.critic.Q1(states, new_actions)
            actor_loss = -q1_pred.mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_optimizer.step()
            
            # Soft updates
            self._soft_update(self.critic, self.critic_target)
            self._soft_update(self.actor, self.actor_target)
            
            actor_loss = actor_loss.item()
            
        critic_loss_val = critic_loss.item()
        self.critic_losses.append(critic_loss_val)
        if actor_loss is not None:
            self.actor_losses.append(actor_loss)
            
        return actor_loss, critic_loss_val

    def _soft_update(self, source, target):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
    def save(self, filepath):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'total_steps': self.total_steps
        }, filepath)
        
    def load(self, filepath):
        checkpoint = torch.load(filepath, map_location=device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.total_steps = checkpoint['total_steps']