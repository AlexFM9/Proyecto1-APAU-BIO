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

class BaseAgent:
    def __init__(self, input_dim=4, hidden_dim=32, output_dim=2, lr=0.01, gamma=0.99):
        self.gamma = gamma
        self.net = PolicyNetwork(input_dim, hidden_dim, output_dim)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        
    def select_action(self, state):
        state_t = torch.FloatTensor(state).unsqueeze(0)
        probs, _ = self.net(state_t)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)
        
    def calculate_returns(self, rewards):
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        return returns

class ReinforceAgent(BaseAgent):
    def update(self, log_probs, rewards):
        returns = self.calculate_returns(rewards)
        policy_loss = []
        for log_prob, G in zip(log_probs, returns):
            policy_loss.append(-log_prob * G)
        
        self.optimizer.zero_grad()
        policy_loss = torch.stack(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()

class HybridReinforceAgent(BaseAgent):
    def __init__(self, input_dim=4, hidden_dim=32, output_dim=2, lr=0.01, gamma=0.99, tau_stdp=0.8, alpha_stdp=0.005):
        super().__init__(input_dim, hidden_dim, output_dim, lr, gamma)
        # We only use Adam for the first layer. The second layer is trained via STDP.
        self.optimizer_fc1 = optim.Adam([self.net.fc1.weight, self.net.fc1.bias], lr=lr)
        self.tau_stdp = tau_stdp
        self.alpha_stdp = alpha_stdp
        
    def select_action_with_trace(self, state, current_trace):
        state_t = torch.FloatTensor(state).unsqueeze(0)
        probs, h = self.net(state_t)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        
        # Calculate Hebbian co-activation trace
        post_synaptic = torch.zeros(self.net.fc2.out_features)
        post_synaptic[action.item()] = 1.0
        
        # current_coactivation = post * pre^T
        coactivation = post_synaptic.unsqueeze(1) @ h
        
        # update trace: e(t) = tau * e(t-1) + coactivation
        new_trace = self.tau_stdp * current_trace + coactivation
        
        return action.item(), dist.log_prob(action), new_trace
        
    def update(self, log_probs, rewards, traces):
        returns = self.calculate_returns(rewards)
        
        # 1. Update first layer with analytical gradient (REINFORCE)
        policy_loss = []
        for log_prob, G in zip(log_probs, returns):
            policy_loss.append(-log_prob * G)
            
        self.optimizer_fc1.zero_grad()
        policy_loss = torch.stack(policy_loss).sum()
        policy_loss.backward()
        self.optimizer_fc1.step()
        
        # 2. Update second layer with 3-factor rule (STDP + REINFORCE reward)
        with torch.no_grad():
            for t, G in enumerate(returns):
                # Delta W = alpha_stdp * G_t * trace_t
                self.net.fc2.weight += self.alpha_stdp * G.item() * traces[t]
