import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        h = torch.relu(self.fc1(x))
        out = self.fc2(h)
        return torch.softmax(out, dim=-1), h

def train_hybrid(episodes=500):
    env = gym.make('CartPole-v1')
    net = PolicyNetwork(4, 16, 2)
    optimizer_fc1 = optim.Adam([net.fc1.weight, net.fc1.bias], lr=0.01)
    
    alpha_stdp = 0.005 # Learning rate for the hybrid STDP layer
    gamma = 0.99
    tau = 0.8 # Decay rate for STDP trace
    
    rewards_history = []
    
    for ep in range(episodes):
        state, _ = env.reset()
        log_probs = []
        rewards = []
        
        # We'll collect the traces manually
        episode_traces = []
        episode_h = []
        episode_actions = []
        
        trace = torch.zeros_like(net.fc2.weight) # shape (2, 16)
        
        done = False
        while not done:
            state_t = torch.FloatTensor(state).unsqueeze(0)
            probs, h = net(state_t)
            
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            
            # Post-synaptic activation (1 for the chosen action, 0 for others)
            post_synaptic = torch.zeros(2)
            post_synaptic[action.item()] = 1.0
            
            # Update trace: e(t) = tau * e(t-1) + post * pre^T
            # h is (1, 16), post_synaptic is (2)
            # We want outer product: post_synaptic.unsqueeze(1) @ h
            current_coactivation = post_synaptic.unsqueeze(1) @ h
            trace = tau * trace + current_coactivation
            
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            
            log_probs.append(dist.log_prob(action))
            rewards.append(reward)
            episode_traces.append(trace.clone())
            
            state = next_state
            
        # Calculate returns
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        
        # Update using 3-factor rule for fc2, and standard REINFORCE for fc1 (or freeze fc1)
        # Let's train fc1 with standard REINFORCE and fc2 with STDP
        
        # 1. Update fc1 with REINFORCE
        policy_loss = []
        for log_prob, G in zip(log_probs, returns):
            policy_loss.append(-log_prob * G)
        policy_loss = torch.stack(policy_loss).sum()
        
        optimizer_fc1.zero_grad()
        policy_loss.backward()
        optimizer_fc1.step()
        
        # 2. Update fc2 with 3-Factor STDP rule
        with torch.no_grad():
            for t, G in enumerate(returns):
                # Delta W = alpha * G_t * trace_t
                net.fc2.weight += alpha_stdp * G.item() * episode_traces[t]
                
        rewards_history.append(sum(rewards))
        if ep % 50 == 0:
            print(f"Episode {ep}, Reward: {sum(rewards)}")
            
    return rewards_history

if __name__ == "__main__":
    train_hybrid()
