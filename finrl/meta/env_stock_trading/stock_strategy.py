# stock_strategy.py

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Any

class BaseStrategy(ABC):
    """Abstract base class for all trading strategies"""
    
    def __init__(self):
        self.position = 0
        
    @abstractmethod
    def generate_signals(self, state, data) -> np.ndarray:
        pass

class MACDStrategy(BaseStrategy):
    def generate_signals(self, state, data) -> np.ndarray:
        stock_dim = (len(state) - 1) // 3
        signals = np.zeros(stock_dim)
        
        for i in range(stock_dim):
            macd = data['macd'].iloc[-1]
            if macd > 0:
                signals[i] = 1
            elif macd < 0:
                signals[i] = -1
        return signals

class BollingerBandsStrategy(BaseStrategy):
    def generate_signals(self, state, data) -> np.ndarray:
        stock_dim = (len(state) - 1) // 3
        signals = np.zeros(stock_dim)
        
        for i in range(stock_dim):
            current_price = state[i + 1]
            upper_band = data['boll_ub'].iloc[-1]
            lower_band = data['boll_lb'].iloc[-1]
            
            if current_price < lower_band:
                signals[i] = 1
            elif current_price > upper_band:
                signals[i] = -1
        return signals

class RSICCIStrategy(BaseStrategy):
    def generate_signals(self, state, data) -> np.ndarray:
        stock_dim = (len(state) - 1) // 3
        signals = np.zeros(stock_dim)
        
        for i in range(stock_dim):
            rsi = data['rsi_30'].iloc[-1]
            cci = data['cci_30'].iloc[-1]
            
            if rsi < 30 and cci < -100:
                signals[i] = 1
            elif rsi > 70 and cci > 100:
                signals[i] = -1
        return signals

class CompositeStrategy(BaseStrategy):
    def __init__(self, strategies: List[BaseStrategy], weights: List[float]):
        super().__init__()
        if len(strategies) != len(weights) or not np.isclose(sum(weights), 1.0):
            raise ValueError("Invalid strategy weights")
        self.strategies = strategies
        self.weights = weights
    
    def generate_signals(self, state, data) -> np.ndarray:
        combined_signals = np.zeros((len(self.strategies), (len(state) - 1) // 3))
        for i, strategy in enumerate(self.strategies):
            combined_signals[i] = strategy.generate_signals(state, data)
        return np.clip(np.average(combined_signals, axis=0, weights=self.weights), -1, 1)

class StrategyEvaluator:
    """Class for strategy evaluation and demonstration generation"""
    
    def __init__(self, env, strategy=None):
        self.env = env
        self.strategy = strategy or self.default_strategy()
    
    @staticmethod
    def default_strategy():
        """Create default composite strategy"""
        return CompositeStrategy(
            strategies=[
                MACDStrategy(),
                BollingerBandsStrategy(),
                RSICCIStrategy()
            ],
            weights=[0.4, 0.3, 0.3]
        )
    
    def generate_expert_demonstrations(self, num_episodes: int = 100) -> List[Dict]:
        """Generate expert demonstrations using the current strategy"""
        demonstrations = []
        
        for episode in range(num_episodes):
            state = self.env.reset()[0]
            done = False
            
            while not done:
                actions = self.strategy.generate_signals(state, self.env.data)
                next_state, reward, done, _, info = self.env.step(actions)
                
                demonstrations.append({
                    'state': state.copy(),
                    'action': actions.copy(),
                    'reward': reward,
                    'next_state': next_state.copy(),
                    'done': done
                })
                
                state = next_state
            
            if (episode + 1) % 10 == 0:
                print(f"Generated {episode + 1} episodes of demonstrations")
        
        return demonstrations
    
    def evaluate_strategy(self, num_episodes: int = 10) -> Dict[str, Any]:
        """Evaluate the current strategy's performance"""
        results = []
        
        for episode in range(num_episodes):
            state = self.env.reset()[0]
            done = False
            episode_reward = 0
            
            while not done:
                actions = self.strategy.generate_signals(state, self.env.data)
                next_state, reward, done, _, info = self.env.step(actions)
                episode_reward += reward
                state = next_state
            
            results.append({
                'episode': episode,
                'reward': episode_reward,
                'final_value': self.env.asset_memory[-1],
                'trades': self.env.trades
            })
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        
        evaluation_metrics = {
            'mean_reward': results_df['reward'].mean(),
            'std_reward': results_df['reward'].std(),
            'mean_final_value': results_df['final_value'].mean(),
            'std_final_value': results_df['final_value'].std(),
            'avg_trades_per_episode': results_df['trades'].mean(),
            'results_df': results_df
        }
        
        return evaluation_metrics

# Helper function to plot strategy results
def plot_strategy_results(evaluation_metrics: Dict[str, Any]):
    """Plot the results of strategy evaluation"""
    results_df = evaluation_metrics['results_df']
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot rewards
    ax1.plot(results_df['episode'], results_df['reward'])
    ax1.set_title('Rewards per Episode')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.grid(True)
    
    # Plot final values
    ax2.plot(results_df['episode'], results_df['final_value'])
    ax2.set_title('Final Portfolio Value per Episode')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Portfolio Value')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
