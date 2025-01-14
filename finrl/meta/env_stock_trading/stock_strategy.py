# stock_strategy.py
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import List, Dict, Any

class BaseStrategy(ABC):
    """Abstract base class for all trading strategies"""
    
    def __init__(self):
        self.position = 0
        
    @abstractmethod
    def generate_signals(self, state: np.ndarray, data: pd.Series) -> np.ndarray:
        """Generate trading signals from current state and data"""
        pass
        
    def get_signal_features(self, state: np.ndarray, data: pd.Series) -> np.ndarray:
        """Get strategy signals as features for the model"""
        return self.generate_signals(state, data)

class MACDStrategy(BaseStrategy):
    def generate_signals(self, state: np.ndarray, data: pd.Series) -> np.ndarray:
        stock_dim = (len(state) - 1) // 3
        signals = np.zeros(stock_dim)
        
        # Access MACD values for each stock
        macd_values = data['macd'].values if isinstance(data['macd'], pd.Series) else np.array([data['macd']])
        
        # Generate signals based on MACD
        signals[macd_values > 0] = 1
        signals[macd_values < 0] = -1
        return signals

class BollingerBandsStrategy(BaseStrategy):
    def generate_signals(self, state: np.ndarray, data: pd.Series) -> np.ndarray:
        stock_dim = (len(state) - 1) // 3
        signals = np.zeros(stock_dim)
        
        # Get current prices and bands
        current_prices = state[1:stock_dim + 1]
        upper_bands = data['boll_ub'].values if isinstance(data['boll_ub'], pd.Series) else np.array([data['boll_ub']])
        lower_bands = data['boll_lb'].values if isinstance(data['boll_lb'], pd.Series) else np.array([data['boll_lb']])
        
        # Generate signals based on Bollinger Bands
        signals[current_prices < lower_bands] = 1
        signals[current_prices > upper_bands] = -1
        return signals

class RSICCIStrategy(BaseStrategy):
    def generate_signals(self, state: np.ndarray, data: pd.Series) -> np.ndarray:
        stock_dim = (len(state) - 1) // 3
        signals = np.zeros(stock_dim)
        
        # Get RSI and CCI values
        rsi_values = data['rsi_30'].values if isinstance(data['rsi_30'], pd.Series) else np.array([data['rsi_30']])
        cci_values = data['cci_30'].values if isinstance(data['cci_30'], pd.Series) else np.array([data['cci_30']])
        
        # Generate signals based on RSI and CCI
        signals[(rsi_values < 30) & (cci_values < -100)] = 1
        signals[(rsi_values > 70) & (cci_values > 100)] = -1
        return signals

class CompositeStrategy(BaseStrategy):
    def __init__(self, strategies: List[BaseStrategy], weights: List[float]):
        super().__init__()
        if not np.isclose(sum(weights), 1.0):
            raise ValueError("Strategy weights must sum to 1.0")
        self.strategies = strategies
        self.weights = np.array(weights)
    
    def generate_signals(self, state: np.ndarray, data: pd.Series) -> np.ndarray:
        stock_dim = (len(state) - 1) // 3
        combined_signals = np.zeros((len(self.strategies), stock_dim))
        
        for i, strategy in enumerate(self.strategies):
            combined_signals[i] = strategy.generate_signals(state, data)
            
        return np.clip(np.average(combined_signals, axis=0, weights=self.weights), -1, 1)
    
    def get_signal_features(self, state: np.ndarray, data: pd.Series) -> Dict[str, np.ndarray]:
        """Generate separate signals from each strategy"""
        signals = {}
        for strategy in self.strategies:
            if isinstance(strategy, MACDStrategy):
                signals['macd_signal'] = strategy.generate_signals(state, data)
            elif isinstance(strategy, BollingerBandsStrategy):
                signals['bollinger_signal'] = strategy.generate_signals(state, data)
            elif isinstance(strategy, RSICCIStrategy):
                signals['rscci_signal'] = strategy.generate_signals(state, data)
        
        signals['combined_signal'] = self.generate_signals(state, data)
        return signals

class HybridSignalGenerator:
    def __init__(self, strategy: CompositeStrategy, strategy_weight: float = 0.3):
        if not 0 <= strategy_weight <= 1:
            raise ValueError("Strategy weight must be between 0 and 1")
        self.strategy = strategy
        self.strategy_weight = strategy_weight
        self.model_weight = 1 - strategy_weight
    
    def combine_signals(self, model_action: np.ndarray, state: np.ndarray, data: pd.Series) -> np.ndarray:
        """Combine model actions with strategy signals"""
        strategy_signals = self.strategy.generate_signals(state, data)
        hybrid_action = (model_action * self.model_weight + 
                        strategy_signals * self.strategy_weight)
        return np.clip(hybrid_action, -1, 1)
    
    def get_signal_features(self, state: np.ndarray, data: pd.Series) -> Dict[str, np.ndarray]:
        """Get all strategy signals as features for the model"""
        return self.strategy.get_signal_features(state, data)
