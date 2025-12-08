import os
import pickle
from collections import defaultdict, deque
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from projects.snake.agents.base_agent import BaseAgent
from projects.snake.environment import SnakeEnv, Action

QTable: TypeAlias = defaultdict[tuple[int, ...], NDArray[np.float64]]


class SARSAAgent(BaseAgent):
    """SARSA agent with epsilon-greedy exploration."""

    def __init__(
        self,
        env: "SnakeEnv",
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        seed: int | None = None,
    ) -> None:
        """
        Initialize SARSA agent.

        Args:
            env: Environment instance to interact with
            learning_rate: Learning rate (alpha)
            discount_factor: Discount factor (gamma)
            epsilon: Initial exploration rate
            epsilon_decay: Decay rate for epsilon
            epsilon_min: Minimum epsilon value
            seed: Random seed for reproducibility
        """
        self.env: "SnakeEnv" = env
        self.action_space: int = env.action_space
        self.learning_rate: float = learning_rate
        self.discount_factor: float = discount_factor
        self.epsilon: float = epsilon
        self.epsilon_decay: float = epsilon_decay
        self.epsilon_min: float = epsilon_min

        # Q-table: dictionary mapping state tuples to action values
        # Using defaultdict to initialize unseen states to zeros
        self.q_table: QTable = defaultdict(lambda: np.zeros(self.action_space))

        self.rng: np.random.RandomState = np.random.RandomState(seed)

    def get_action(self, training: bool = True) -> Action:
        """
        Select action using epsilon-greedy policy.

        Args:
            training: If True, use epsilon-greedy; if False, use greedy

        Returns:
            action: Selected action
        """
        # Get feature vector from environment
        features = self.env.get_features()
        state_tuple = tuple(features)

        # Epsilon-greedy exploration
        if training and self.rng.random() < self.epsilon:
            return Action(self.rng.randint(0, self.action_space))
        else:
            # Exploit: choose best action
            q_values = self.q_table[state_tuple]
            return Action(np.argmax(q_values))

    def update(
        self,
        prev_features: NDArray[np.int8],
        action: int,
        reward: float,
        next_action: int,
        done: bool,
    ) -> None:
        """
        Update Q-table using SARSA update rule.

        Q(s,a) <- Q(s,a) + alpha * [r + gamma * Q(s',a') - Q(s,a)]

        Args:
            prev_features: Previous state features
            action: Action taken
            reward: Reward received
            next_action: Next action selected by policy
            done: Whether episode is finished
        """
        state_tuple = tuple(prev_features)

        # Current Q-value
        current_q = self.q_table[state_tuple][action]

        # Calculate target Q-value
        if done:
            target_q = reward
        else:
            # Get current features (after taking action)
            next_features = self.env.get_features()
            next_state_tuple = tuple(next_features)
            next_q = self.q_table[next_state_tuple][next_action]
            target_q = reward + self.discount_factor * next_q

        # Update Q-value
        self.q_table[state_tuple][action] = current_q + self.learning_rate * (
            target_q - current_q
        )
        #####################################################################
        # Implement update method for SARSA Agent
        # Note: Be sure to remember the difference in update between Q-learning and SARSA
        # 1. Get current Q value by using the state tuple and action index
        # 2. If the snake has not reached the food, retrieve the next state using self.env.get_features()
        # 3. Calculate Q-learning target Q value.
        # 4. Update Q table of the agent.
        # 5. But if the snake has reached the food, update the q table with the reward as the target Q value
        # TODO: Write your code here:
        pass
        #####################################################################

    def decay_epsilon(self) -> None:
        """Decay epsilon after each episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, filepath: str) -> None:
        """Save Q-table to file."""
        # Convert defaultdict to regular dict for pickling
        q_table_dict = dict(self.q_table)
        save_data = {
            "q_table": q_table_dict,
            "epsilon": self.epsilon,
            "action_space": self.action_space,
            "learning_rate": self.learning_rate,
            "discount_factor": self.discount_factor,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
        }
        with open(filepath, "wb") as f:
            pickle.dump(save_data, f)

    def load(self, filepath: str) -> None:
        """Load Q-table from file."""
        with open(filepath, "rb") as f:
            save_data = pickle.load(f)

        # Restore Q-table as defaultdict
        self.q_table = defaultdict(lambda: np.zeros(self.action_space))
        self.q_table.update(save_data["q_table"])

        self.epsilon = save_data["epsilon"]
        self.action_space = save_data["action_space"]
        self.learning_rate = save_data["learning_rate"]
        self.discount_factor = save_data["discount_factor"]
        self.epsilon_min = save_data["epsilon_min"]
        self.epsilon_decay = save_data["epsilon_decay"]

        print(f"Q-table loaded from {filepath}")
        print(f"Number of states in Q-table: {len(self.q_table)}")

    def train(
        self,
        num_episodes: int,
        save_interval: int = 100,
        model_dir: str = "models",
    ) -> None:
        """
        Train the agent on the given environment.

        Args:
            num_episodes: Number of episodes to train
            save_interval: Interval for saving Q-table checkpoints
            model_dir: Directory to save model checkpoints

        Returns:
            dict: Training metrics including episode_rewards and episode_scores
        """
        # Create directories
        os.makedirs(model_dir, exist_ok=True)

        # Training loop
        for episode in tqdm(range(1, num_episodes + 1), desc="Training"):
            self.env.reset()
            done = False

            # Get initial action using SARSA policy
            action = self.get_action(training=True)

            while not done:
                #####################################################################
                # Implement SARSA training function.
                # 1. Get current state
                # 2. Get action
                # 3. Step snake through grid with action, retrieving reward and done values
                # 4. Use SARSA update method to update Q table
                # 5. Increment counters
                # #TODO: Write code here:
                pass
                #####################################################################

            # Decay epsilon after episode
            self.decay_epsilon()

            # Save Q-table periodically
            if episode % save_interval == 0:
                save_path = os.path.join(model_dir, f"sarsa_episode_{episode}.pkl")
                self.save(save_path)

        # Final save
        final_path = os.path.join(model_dir, "sarsa_final.pkl")
        self.save(final_path)
