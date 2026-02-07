#!/usr/bin/env python3
"""
Analyze training log to identify loss trends and issues.
"""

import re
import numpy as np
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

def parse_training_log(log_file):
    """Parse training log and extract metrics."""
    episodes = []
    losses = []
    policy_losses = []
    value_losses = []
    entropies = []
    rewards = []
    pnls = []
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    in_data_section = False
    for line in lines:
        if 'EPISODE-BY-EPISODE P&L TRENDS' in line:
            in_data_section = True
            continue
        if 'SUMMARY STATISTICS' in line:
            break
        if not in_data_section:
            continue
        
        # Parse episode line: Episode    P&L             Reward          Loss            Policy       Value        Entropy      Steps
        # Example: 0          $-6475.00       -67.04          0.970589        -0.003112    1.000204     5.300714     64
        match = re.match(r'^\s*(\d+)\s+\$?(-?\d+\.?\d*)\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)\s+\d+', line)
        if match:
            episode = int(match.group(1))
            pnl = float(match.group(2))
            reward = float(match.group(3))
            loss = float(match.group(4))
            policy_loss = float(match.group(5))
            value_loss = float(match.group(6))
            entropy = float(match.group(7))
            
            episodes.append(episode)
            pnls.append(pnl)
            rewards.append(reward)
            losses.append(loss)
            policy_losses.append(policy_loss)
            value_losses.append(value_loss)
            entropies.append(entropy)
    
    return {
        'episodes': np.array(episodes),
        'losses': np.array(losses),
        'policy_losses': np.array(policy_losses),
        'value_losses': np.array(value_losses),
        'entropies': np.array(entropies),
        'rewards': np.array(rewards),
        'pnls': np.array(pnls)
    }

def analyze_trends(data, window_size=100):
    """Analyze trends in the data."""
    n = len(data['episodes'])
    
    print("=" * 80)
    print("LOSS TREND ANALYSIS")
    print("=" * 80)
    
    # Overall statistics
    print(f"\n📊 OVERALL STATISTICS ({n} episodes):")
    print(f"   Loss:        {np.mean(data['losses']):.6f} ± {np.std(data['losses']):.6f}")
    print(f"   Policy Loss: {np.mean(data['policy_losses']):.6f} ± {np.std(data['policy_losses']):.6f}")
    print(f"   Value Loss:  {np.mean(data['value_losses']):.6f} ± {np.std(data['value_losses']):.6f}")
    print(f"   Entropy:     {np.mean(data['entropies']):.6f} ± {np.std(data['entropies']):.6f}")
    
    # First vs Last comparison
    first_n = min(100, n // 10)
    last_n = min(100, n // 10)
    
    first_loss = np.mean(data['losses'][:first_n])
    last_loss = np.mean(data['losses'][-last_n:])
    first_policy = np.mean(data['policy_losses'][:first_n])
    last_policy = np.mean(data['policy_losses'][-last_n:])
    first_value = np.mean(data['value_losses'][:first_n])
    last_value = np.mean(data['value_losses'][-last_n:])
    first_entropy = np.mean(data['entropies'][:first_n])
    last_entropy = np.mean(data['entropies'][-last_n:])
    
    print(f"\n📈 FIRST {first_n} vs LAST {last_n} EPISODES:")
    print(f"   Loss:        {first_loss:.6f} → {last_loss:.6f} ({last_loss - first_loss:+.6f})")
    print(f"   Policy Loss: {first_policy:.6f} → {last_policy:.6f} ({last_policy - first_policy:+.6f})")
    print(f"   Value Loss:  {first_value:.6f} → {last_value:.6f} ({last_value - first_value:+.6f})")
    print(f"   Entropy:     {first_entropy:.6f} → {last_entropy:.6f} ({last_entropy - first_entropy:+.6f})")
    
    # Trend analysis
    print(f"\n🔍 TREND ANALYSIS:")
    
    # Loss trend
    loss_trend = "✅ DECREASING" if last_loss < first_loss else "❌ INCREASING/STABLE"
    print(f"   Loss: {loss_trend}")
    if abs(last_loss - first_loss) < 0.01:
        print(f"      ⚠️  Loss is essentially flat (change < 0.01)")
    
    # Value loss trend
    value_trend = "✅ DECREASING" if last_value < first_value else "❌ INCREASING/STABLE"
    print(f"   Value Loss: {value_trend}")
    if abs(last_value - first_value) < 0.01:
        print(f"      ⚠️  Value loss is essentially flat (change < 0.01)")
    
    # Policy loss trend
    policy_trend = "✅ DECREASING" if abs(last_policy) < abs(first_policy) else "❌ INCREASING/STABLE"
    print(f"   Policy Loss: {policy_trend}")
    
    # Entropy trend
    entropy_trend = "✅ DECREASING" if last_entropy < first_entropy else "❌ INCREASING"
    print(f"   Entropy: {entropy_trend}")
    
    # Key issues
    print(f"\n⚠️  KEY ISSUES IDENTIFIED:")
    issues = []
    
    if last_loss >= first_loss:
        issues.append("❌ Total loss is NOT decreasing")
    
    if last_value >= first_value or abs(last_value - first_value) < 0.01:
        issues.append("❌ Value loss is NOT decreasing (stuck around ~1.0)")
    
    if np.mean(data['value_losses']) > 0.95:
        issues.append("⚠️  Value loss is very high (>0.95), dominating total loss")
    
    if abs(np.mean(data['policy_losses'])) < 0.01:
        issues.append("⚠️  Policy loss is tiny (<0.01), value loss dominates")
    
    if np.mean(data['entropies'][-last_n:]) < 0.1:
        issues.append("⚠️  Entropy very low (<0.1), agent may have collapsed to single action")
    
    if not issues:
        print("   ✅ No major issues detected!")
    else:
        for issue in issues:
            print(f"   {issue}")
    
    # Moving averages
    if n >= window_size:
        moving_loss = np.convolve(data['losses'], np.ones(window_size)/window_size, mode='valid')
        moving_value = np.convolve(data['value_losses'], np.ones(window_size)/window_size, mode='valid')
        moving_entropy = np.convolve(data['entropies'], np.ones(window_size)/window_size, mode='valid')
        
        print(f"\n📉 MOVING AVERAGE TRENDS ({window_size}-episode window):")
        print(f"   Loss:        {moving_loss[0]:.6f} → {moving_loss[-1]:.6f} ({moving_loss[-1] - moving_loss[0]:+.6f})")
        print(f"   Value Loss:  {moving_value[0]:.6f} → {moving_value[-1]:.6f} ({moving_value[-1] - moving_value[0]:+.6f})")
        print(f"   Entropy:     {moving_entropy[0]:.6f} → {moving_entropy[-1]:.6f} ({moving_entropy[-1] - moving_entropy[0]:+.6f})")
    
    return {
        'first_loss': first_loss,
        'last_loss': last_loss,
        'first_value': first_value,
        'last_value': last_value,
        'first_entropy': first_entropy,
        'last_entropy': last_entropy
    }

def plot_trends(data, output_file=None):
    """Plot loss trends."""
    if not HAS_MATPLOTLIB:
        print("⚠️  matplotlib not available, skipping plot generation")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    episodes = data['episodes']
    
    # Loss
    axes[0, 0].plot(episodes, data['losses'], alpha=0.3, label='Raw', color='blue')
    if len(episodes) >= 100:
        window = 100
        moving = np.convolve(data['losses'], np.ones(window)/window, mode='valid')
        axes[0, 0].plot(episodes[window-1:], moving, label=f'{window}-ep MA', color='red', linewidth=2)
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Loss')
    axes[0, 0].set_title('Total Loss Trend')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Value Loss
    axes[0, 1].plot(episodes, data['value_losses'], alpha=0.3, label='Raw', color='green')
    if len(episodes) >= 100:
        moving = np.convolve(data['value_losses'], np.ones(window)/window, mode='valid')
        axes[0, 1].plot(episodes[window-1:], moving, label=f'{window}-ep MA', color='red', linewidth=2)
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Value Loss')
    axes[0, 1].set_title('Value Loss Trend')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Policy Loss
    axes[1, 0].plot(episodes, data['policy_losses'], alpha=0.3, label='Raw', color='orange')
    if len(episodes) >= 100:
        moving = np.convolve(data['policy_losses'], np.ones(window)/window, mode='valid')
        axes[1, 0].plot(episodes[window-1:], moving, label=f'{window}-ep MA', color='red', linewidth=2)
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Policy Loss')
    axes[1, 0].set_title('Policy Loss Trend')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Entropy
    axes[1, 1].plot(episodes, data['entropies'], alpha=0.3, label='Raw', color='purple')
    if len(episodes) >= 100:
        moving = np.convolve(data['entropies'], np.ones(window)/window, mode='valid')
        axes[1, 1].plot(episodes[window-1:], moving, label=f'{window}-ep MA', color='red', linewidth=2)
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Entropy')
    axes[1, 1].set_title('Entropy Trend')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150)
        print(f"\n📊 Plot saved to: {output_file}")
    else:
        plt.show()

def main():
    import sys
    
    if len(sys.argv) > 1:
        log_file = sys.argv[1]
    else:
        # Find most recent log file
        log_dir = Path(__file__).parent / 'reinforcement_results'
        log_files = list(log_dir.glob('training_log_*.txt'))
        if not log_files:
            print("❌ No training log files found!")
            return
        log_file = max(log_files, key=lambda p: p.stat().st_mtime)
        print(f"📋 Using most recent log: {log_file.name}")
    
    print(f"\n📖 Parsing: {log_file}")
    data = parse_training_log(log_file)
    
    if len(data['episodes']) == 0:
        print("❌ No data found in log file!")
        return
    
    trends = analyze_trends(data)
    
    # Generate plot
    plot_file = Path(log_file).parent / f"{Path(log_file).stem}_loss_analysis.png"
    plot_trends(data, output_file=str(plot_file))
    
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS:")
    print("=" * 80)
    
    if trends['last_value'] >= trends['first_value']:
        print("\n🔧 VALUE LOSS NOT DECREASING:")
        print("   1. Value loss is stuck around ~1.0, dominating total loss")
        print("   2. This suggests the value function is struggling to learn")
        print("   3. Possible fixes:")
        print("      - Increase value loss coefficient (currently 1.0)")
        print("      - Increase learning rate for value function")
        print("      - Check if value function architecture is sufficient")
        print("      - Consider value function normalization/scaling")
    
    if abs(trends['last_loss'] - trends['first_loss']) < 0.01:
        print("\n🔧 LOSS IS FLAT:")
        print("   1. Total loss is not decreasing significantly")
        print("   2. This could indicate:")
        print("      - Learning rate too low")
        print("      - Value function not learning (see above)")
        print("      - Policy gradient too small (policy loss is tiny)")
        print("   3. Possible fixes:")
        print("      - Increase learning rate")
        print("      - Reduce value loss coefficient to focus more on policy")
        print("      - Check if rewards are properly scaled")

if __name__ == '__main__':
    main()
