from agents.random_agent import RandomAgent
from environment import SnakeEnv


def main() -> None:
    # Initialize environment and agent
    env = SnakeEnv(
        grid_size=10,
        max_steps=500,
        seed=42,
    )

    agent = RandomAgent(action_space=env.action_space)

    # Train the agent
    agent.train(
        num_episodes=20_000,
        save_interval=1_000,
        model_dir="models",
    )


if __name__ == "__main__":
    main()
