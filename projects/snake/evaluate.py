import time

import numpy as np
from projects.snake.agents.base_agent import BaseAgent
from projects.snake.agents.random_agent import RandomAgent
from projects.snake.environment import SnakeEnv

EvalResults = dict[str, float]


def evaluate(
    agent: BaseAgent,
    env: SnakeEnv,
    num_episodes: int,
    render: bool,
) -> EvalResults:
    """
    Evaluate an agent on the environment.

    Args:
        agent: Instantiated agent to evaluate
        env: Environment instance to evaluate on
        num_episodes: Number of episodes to evaluate
        render: Whether to render episodes

    Returns:
        Dictionary containing evaluation metrics
    """
    # Evaluation metrics
    episode_rewards = []
    episode_lengths = []
    episode_scores = []

    print(f"\nRunning {num_episodes} evaluation episodes...")
    print("=" * 50)

    # Evaluation loop
    for episode in range(1, num_episodes + 1):
        env.reset()
        total_reward = 0
        steps = 0
        done = False
        info = {"score": 0}

        if render:
            print(f"\n--- Episode {episode} ---")
            env.render()

        while not done:
            # Select action (greedy, no exploration)
            action = agent.get_action(training=False)
            _, reward, done, info = env.step(action)

            total_reward += reward
            steps += 1

            if render:
                time.sleep(0.1)  # Slow down for viewing
                env.render()

        # Store metrics
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        episode_scores.append(info["score"])

        if episode % 10 == 0 and not render:
            print(
                f"Episode {episode}: Score = {info['score']}, "
                f"Reward = {total_reward:.2f}, Steps = {steps}"
            )

    # Print statistics
    print("\n" + "=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    print(f"Episodes: {num_episodes}")
    print(
        f"Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}"
    )
    print(
        f"Average Length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}"
    )
    print(
        f"Average Score: {np.mean(episode_scores):.2f} ± {np.std(episode_scores):.2f}"
    )
    print(f"Max Score: {np.max(episode_scores):.0f}")
    print(f"Min Score: {np.min(episode_scores):.0f}")
    print("=" * 50)

    results: EvalResults = {
        "avg_reward": float(np.mean(episode_rewards)),
        "avg_length": float(np.mean(episode_lengths)),
        "avg_score": float(np.mean(episode_scores)),
        "max_score": float(np.max(episode_scores)),
        "min_score": float(np.min(episode_scores)),
    }
    return results


if __name__ == "__main__":
    # Initialize environment
    env = SnakeEnv(
        grid_size=10,
        max_steps=500,
        seed=42,
    )

    # Random
    agent = RandomAgent(action_space=env.action_space)

    # Run evaluation
    evaluate(agent=agent, env=env, num_episodes=100, render=False)

    response = input("\nWould you like to watch the agent play? (y/n): ")
    if response.lower() == "y":
        num_episodes = int(input("How many episodes to watch? (default 5): ") or "5")
        evaluate(agent=agent, env=env, num_episodes=num_episodes, render=True)
