"""
TD3 (Twin Delayed Deep Deterministic Policy Gradient) Agent for Hedging
Implementation based on Fujimoto et al. (2018) "Addressing Function Approximation Error in Actor-Critic Methods"
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random
from config import CONFIG, get_model_config, get_exploration_config


# Set device
device = torch.device("cuda" if torch.cuda.is_available() and CONFIG.get("use_gpu", False) else "cpu")


class ReplayBuffer:
    """Experience Replay Buffer for TD3"""
    
    def __init__(self, capacity=None):
        if capacity is None:
            capacity = CONFIG["replay_buffer_size"]
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size=None):
        """Sample a batch of experiences"""
        if batch_size is None:
            batch_size = CONFIG["batch_size"]
        
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        
        states = torch.FloatTensor(np.array([e[0] for e in batch])).to(device)
        actions = torch.FloatTensor(np.array([e[1] for e in batch])).to(device)
        rewards = torch.FloatTensor(np.array([e[2] for e in batch])).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(np.array([e[3] for e in batch])).to(device)
        dones = torch.FloatTensor(np.array([e[4] for e in batch])).unsqueeze(1).to(device)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)
    
    def clear(self):
        """Clear the replay buffer"""
        self.buffer.clear()


class Actor(nn.Module):
    """Actor Network - Outputs continuous action (hedge ratio)"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=None, max_action=None):
        super(Actor, self).__init__()
        
        if hidden_dim is None:
            hidden_dim = CONFIG["hidden_dim"]
        if max_action is None:
            max_action = CONFIG["max_action"]
            
        self.max_action = max_action
        alpha = CONFIG.get("leaky_relu_alpha", 0.05)
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc4 = nn.Linear(hidden_dim // 2, action_dim)
        
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim // 2)
        
        self.leaky_relu = nn.LeakyReLU(alpha)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
        # Final layer with smaller weights for stable initial policy
        nn.init.uniform_(self.fc4.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.fc4.bias, -3e-3, 3e-3)
    
    def forward(self, state):
        x = self.leaky_relu(self.ln1(self.fc1(state)))
        x = self.leaky_relu(self.ln2(self.fc2(x)))
        x = self.leaky_relu(self.ln3(self.fc3(x)))
        x = torch.tanh(self.fc4(x)) * self.max_action
        return x


class Critic(nn.Module):
    """Twin Critic Networks - Q1 and Q2 for TD3"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=None):
        super(Critic, self).__init__()
        
        if hidden_dim is None:
            hidden_dim = CONFIG["hidden_dim"]
        
        alpha = CONFIG.get("leaky_relu_alpha", 0.05)
        
        # Q1 Network
        self.q1_fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.q1_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q1_fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.q1_fc4 = nn.Linear(hidden_dim // 2, 1)
        
        self.q1_ln1 = nn.LayerNorm(hidden_dim)
        self.q1_ln2 = nn.LayerNorm(hidden_dim)
        self.q1_ln3 = nn.LayerNorm(hidden_dim // 2)
        
        # Q2 Network
        self.q2_fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.q2_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q2_fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.q2_fc4 = nn.Linear(hidden_dim // 2, 1)
        
        self.q2_ln1 = nn.LayerNorm(hidden_dim)
        self.q2_ln2 = nn.LayerNorm(hidden_dim)
        self.q2_ln3 = nn.LayerNorm(hidden_dim // 2)
        
        self.leaky_relu = nn.LeakyReLU(alpha)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, state, action):
        """Returns Q1 and Q2 values"""
        sa = torch.cat([state, action], dim=1)
        
        # Q1
        q1 = self.leaky_relu(self.q1_ln1(self.q1_fc1(sa)))
        q1 = self.leaky_relu(self.q1_ln2(self.q1_fc2(q1)))
        q1 = self.leaky_relu(self.q1_ln3(self.q1_fc3(q1)))
        q1 = self.q1_fc4(q1)
        
        # Q2
        q2 = self.leaky_relu(self.q2_ln1(self.q2_fc1(sa)))
        q2 = self.leaky_relu(self.q2_ln2(self.q2_fc2(q2)))
        q2 = self.leaky_relu(self.q2_ln3(self.q2_fc3(q2)))
        q2 = self.q2_fc4(q2)
        
        return q1, q2
    
    def Q1(self, state, action):
        """Returns only Q1 value"""
        sa = torch.cat([state, action], dim=1)
        
        q1 = self.leaky_relu(self.q1_ln1(self.q1_fc1(sa)))
        q1 = self.leaky_relu(self.q1_ln2(self.q1_fc2(q1)))
        q1 = self.leaky_relu(self.q1_ln3(self.q1_fc3(q1)))
        q1 = self.q1_fc4(q1)
        
        return q1


class OUNoise:
    """Ornstein-Uhlenbeck Process for exploration noise"""
    
    def __init__(self, action_dim, theta=None, sigma=None, mu=0.0):
        if theta is None:
            theta = CONFIG["ou_theta"]
        if sigma is None:
            sigma = CONFIG["ou_sigma"]
            
        self.action_dim = action_dim
        self.theta = theta
        self.sigma = sigma
        self.mu = mu
        self.state = np.ones(action_dim) * mu
        
    def reset(self):
        """Reset noise state"""
        self.state = np.ones(self.action_dim) * self.mu
    
    def sample(self):
        """Generate noise sample"""
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.action_dim)
        self.state += dx
        return self.state


class TD3Agent:
    """TD3 Agent for Options Hedging"""
    
    def __init__(self, state_dim, action_dim, config=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Load configuration
        model_config = get_model_config()
        exploration_config = get_exploration_config()
        
        if config is not None:
            model_config.update(config)
        
        self.max_action = model_config["max_action"]
        self.tau = model_config["tau"]
        self.gamma = model_config["gamma"]
        self.policy_noise = model_config["policy_noise"]
        self.noise_clip = model_config["noise_clip"]
        self.policy_freq = model_config["policy_freq"]
        self.batch_size = model_config["batch_size"]
        
        # Exploration parameters
        self.initial_noise = exploration_config["initial_noise"]
        self.final_noise = exploration_config["final_noise"]
        self.noise_decay_steps = exploration_config["noise_decay_steps"]
        self.min_noise = exploration_config["min_noise"]
        self.current_noise = self.initial_noise
        
        # Initialize networks
        hidden_dim = model_config["hidden_dim"]
        
        self.actor = Actor(state_dim, action_dim, hidden_dim, self.max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, hidden_dim, self.max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        self.critic = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=model_config["actor_lr"])
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=model_config["critic_lr"])
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(model_config["replay_buffer_size"])
        
        # OU Noise for exploration
        self.ou_noise = OUNoise(action_dim)
        
        # Training counter
        self.total_steps = 0
        self.update_counter = 0
        
        # Training metrics
        self.actor_losses = []
        self.critic_losses = []
        
    def select_action(self, state, add_noise=True):
        """Select action given state"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy().flatten()
        self.actor.train()
        
        if add_noise:
            # Combine Gaussian noise with OU noise
            gaussian_noise = np.random.normal(0, self.current_noise, size=self.action_dim)
            ou_noise = self.ou_noise.sample() * self.current_noise
            noise = 0.7 * gaussian_noise + 0.3 * ou_noise  # Weighted combination
            action = action + noise
        
        # Clip action to valid range
        action = np.clip(action, -self.max_action, self.max_action)
        
        return action
    
    def get_q_values(self, state, action):
        """Get Q-values for logging"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        action_tensor = torch.FloatTensor(action).unsqueeze(0).to(device)
        
        with torch.no_grad():
            q1, q2 = self.critic(state_tensor, action_tensor)
        
        return {
            'q1': q1.cpu().item(),
            'q2': q2.cpu().item(),
            'q_mean': (q1.cpu().item() + q2.cpu().item()) / 2,
            'action_raw': action[0] if len(action) > 0 else action,
            'state': state
        }
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer"""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def update_noise(self):
        """Decay exploration noise"""
        if self.total_steps < self.noise_decay_steps:
            decay_ratio = self.total_steps / self.noise_decay_steps
            self.current_noise = self.initial_noise - (self.initial_noise - self.final_noise) * decay_ratio
        else:
            self.current_noise = max(self.final_noise, self.min_noise)
    
    def train_step(self):
        """Perform one training step"""
        if len(self.replay_buffer) < self.batch_size:
            return None, None
        
        self.total_steps += 1
        self.update_noise()
        
        # Sample from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Compute target Q-value
        with torch.no_grad():
            # Target policy smoothing
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_actions = (self.actor_target(next_states) + noise).clamp(-self.max_action, self.max_action)
            
            # Clipped double Q-learning
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + (1 - dones) * self.gamma * target_q
        
        # Update critic
        current_q1, current_q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()
        
        actor_loss = None
        
        # Delayed policy updates
        self.update_counter += 1
        if self.update_counter % self.policy_freq == 0:
            # Update actor
            actor_loss = -self.critic.Q1(states, self.actor(states)).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            self.actor_optimizer.step()
            
            # Soft update target networks
            self._soft_update(self.critic, self.critic_target)
            self._soft_update(self.actor, self.actor_target)
            
            actor_loss = actor_loss.item()
        
        critic_loss_val = critic_loss.item()
        
        # Store losses
        self.critic_losses.append(critic_loss_val)
        if actor_loss is not None:
            self.actor_losses.append(actor_loss)
        
        return actor_loss, critic_loss_val
    
    def _soft_update(self, source, target):
        """Soft update of target network parameters"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def save(self, filepath):
        """Save model checkpoint"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'actor_target_state_dict': self.actor_target.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'total_steps': self.total_steps,
            'update_counter': self.update_counter,
            'current_noise': self.current_noise,
            'config': {
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'max_action': self.max_action,
                'tau': self.tau,
                'gamma': self.gamma,
            }
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=device)
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.total_steps = checkpoint['total_steps']
        self.update_counter = checkpoint['update_counter']
        self.current_noise = checkpoint.get('current_noise', self.final_noise)
        
        print(f"Model loaded from {filepath}")
        return checkpoint.get('config', {})
    
    def reset_noise(self):
        """Reset OU noise for new episode"""
        self.ou_noise.reset()
    
    def get_training_stats(self):
        """Get training statistics"""
        stats = {
            'total_steps': self.total_steps,
            'update_counter': self.update_counter,
            'current_noise': self.current_noise,
            'buffer_size': len(self.replay_buffer),
            'avg_critic_loss': np.mean(self.critic_losses[-100:]) if self.critic_losses else 0,
            'avg_actor_loss': np.mean(self.actor_losses[-100:]) if self.actor_losses else 0,
        }
        return stats
