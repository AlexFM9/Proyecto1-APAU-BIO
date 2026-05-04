import gymnasium as gym
import torch
import numpy as np
import pandas as pd
from agent import ReinforceAgent, HybridReinforceAgent
import os

def train_agent(agent, env_name='CartPole-v1', episodes=500, is_hybrid=False):
    env = gym.make(env_name)
    rewards_history = []
    
    for ep in range(episodes):
        state, _ = env.reset()
        log_probs = []
        rewards = []
        
        if is_hybrid:
            traces = []
            current_trace = torch.zeros_like(agent.net.fc2.weight)
        
        done = False
        while not done:
            if is_hybrid:
                action, log_prob, current_trace = agent.select_action_with_trace(state, current_trace)
                traces.append(current_trace.clone())
            else:
                action, log_prob = agent.select_action(state)
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            log_probs.append(log_prob)
            rewards.append(reward)
            state = next_state
            
        if is_hybrid:
            agent.update(log_probs, rewards, traces)
        else:
            agent.update(log_probs, rewards)
            
        rewards_history.append(sum(rewards))
        if (ep+1) % 100 == 0:
            print(f"Episode {ep+1}/{episodes}, Reward: {np.mean(rewards_history[-100:])}")
            
    return rewards_history

def main():
    if not os.path.exists('results'):
        os.makedirs('results')

    print("--- Training Baseline REINFORCE ---")
    baseline_agent = ReinforceAgent(lr=0.01)
    baseline_rewards = train_agent(baseline_agent, episodes=400)
    
    print("\n--- Training Hybrid REINFORCE + STDP ---")
    hybrid_agent = HybridReinforceAgent(lr=0.01, tau_stdp=0.9, alpha_stdp=0.01)
    hybrid_rewards = train_agent(hybrid_agent, episodes=400, is_hybrid=True)
    
    # Save results for plotting
    df = pd.DataFrame({
        'Episode': range(1, 401),
        'Baseline': baseline_rewards,
        'Hybrid': hybrid_rewards
    })
    df.to_csv('results/training_rewards.csv', index=False)
    print("\nResults saved to results/training_rewards.csv")

if __name__ == "__main__":
    main()
