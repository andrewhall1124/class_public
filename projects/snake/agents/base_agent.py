from abc import ABC, abstractmethod

from projects.snake.environment import Action


class BaseAgent(ABC):
    """Abstract base class for reinforcement learning agents."""

    @abstractmethod
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

    @abstractmethod
    def get_action(self, training: bool = True) -> Action:
        """
        Select an action given the current state.

        Args:
            training: Whether the agent is in training mode

        Returns:
            Selected action
        """
        pass
