import numpy as np

from projects.snake.agents.base_agent import BaseAgent
from projects.snake.environment import Action


class RandomAgent(BaseAgent):
    """Agent that selects actions randomly."""

    def __init__(self, action_space: int) -> None:
        """
        Initialize random agent.

        Args:
            action_space: Number of possible actions
        """
        self.action_space = action_space

    def train(self, num_episodes: int, save_interval: int, model_dir: str) -> None:
        """
        Train the agent for a given number of episodes.

        Args:
            num_episodes: Number of episodes to train for
            save_interval: Interval at which to save the model
            model_dir: Directory to save the model to

        Returns:
            None
        """
        pass

    def get_action(self, training: bool = True) -> Action:
        """
        Select a random action.

        Args:
            training: Whether the agent is in training mode (unused)

        Returns:
            Randomly selected action
        """
        return Action(np.random.randint(0, self.action_space))
