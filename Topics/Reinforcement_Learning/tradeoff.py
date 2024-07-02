import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Number of arms (bandits)
n_arms = 10
# Number of rounds
n_rounds = 10000
# Different epsilon values to evaluate
epsilons = [0.02, 0.03, 0.04, 0.05, 0.06]

# Generate true reward probabilities for each arm
true_probs = np.random.rand(n_arms)


def run_epsilon_greedy(epsilon, true_probs, n_rounds):
    n_arms = len(true_probs)
    Q_values = np.zeros(n_arms)
    action_counts = np.zeros(n_arms)
    rewards = np.zeros(n_rounds)

    for t in range(n_rounds):
        if np.random.rand() < epsilon:
            action = np.random.randint(n_arms)
        else:
            action = np.argmax(Q_values)

        reward = np.random.rand() < true_probs[action]
        action_counts[action] += 1
        Q_values[action] += (reward - Q_values[action]) / action_counts[action]
        rewards[t] = reward

    cumulative_rewards = np.cumsum(rewards)
    average_rewards = cumulative_rewards / (np.arange(n_rounds) + 1)

    return average_rewards


# Plotting the results
plt.figure(figsize=(14, 8))

for epsilon in epsilons:
    average_rewards = run_epsilon_greedy(epsilon, true_probs, n_rounds)
    final_avg_reward = average_rewards[-1]
    plt.plot(
        average_rewards,
        label=f"Epsilon = {epsilon}, Final Avg Reward = {final_avg_reward:.4f}",
    )

plt.xlabel("Rounds")
plt.ylabel("Average Reward")
plt.title("Exploration vs. Exploitation Trade-off")
plt.legend()
plt.grid(True)
plt.show()
