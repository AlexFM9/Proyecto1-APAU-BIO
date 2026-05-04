import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_results():
    if not os.path.exists('results/training_rewards.csv'):
        print("Data not found. Run train.py first.")
        return
        
    df = pd.DataFrame(pd.read_csv('results/training_rewards.csv'))
    
    # Calculate moving averages for smoother curves
    window = 50
    df['Baseline_MA'] = df['Baseline'].rolling(window=window).mean()
    df['Hybrid_MA'] = df['Hybrid'].rolling(window=window).mean()
    
    plt.figure(figsize=(10, 6))
    
    # Plot raw data with low alpha
    plt.plot(df['Episode'], df['Baseline'], alpha=0.2, color='blue')
    plt.plot(df['Episode'], df['Hybrid'], alpha=0.2, color='orange')
    
    # Plot moving averages
    plt.plot(df['Episode'], df['Baseline_MA'], label=f'Baseline REINFORCE (MA {window})', color='blue', linewidth=2)
    plt.plot(df['Episode'], df['Hybrid_MA'], label=f'Hybrid STDP+REINFORCE (MA {window})', color='orange', linewidth=2)
    
    plt.axhline(y=500, color='r', linestyle='--', label='Max Reward (Solved)')
    
    plt.xlabel('Episodes')
    plt.ylabel('Accumulated Reward')
    plt.title('Ablation Study: REINFORCE vs Hybrid (REINFORCE + STDP)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig('results/learning_curves.png')
    print("Plot saved to results/learning_curves.png")

if __name__ == "__main__":
    plot_results()
