import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
from stable_baselines3.common.vec_env import DummyVecEnv
from typing import List, Dict, Tuple, Any, Optional
from collections import deque
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StockTradingEnv(gym.Env):
    """A stock trading environment with improved error handling and consistency"""
    
    metadata = {'render.modes': ['human']}
    
    def __init__(
        self,
        df: pd.DataFrame,
        stock_dim: int,
        hmax: int,
        initial_amount: int,
        num_stock_shares: list[int],
        buy_cost_pct: list[float],
        sell_cost_pct: list[float],
        reward_scaling: float,
        tech_indicator_list: list[str],
        turbulence_threshold=None,
        risk_indicator_col="turbulence",
        day=0,
        initial=True,
        previous_state=[],
        max_memory_length=10000
    ):
        super().__init__()
        
        # Validate input data
        self._validate_dataframe(df)
        self._validate_parameters(
            stock_dim, hmax, initial_amount, 
            num_stock_shares, buy_cost_pct, sell_cost_pct
        )
        
        # Basic parameters with type enforcement
        self.day = int(day)
        self.df = df.copy()  # Make a copy to prevent external modifications
        self.stock_dim = stock_dim
        self.hmax = hmax
        self.initial_amount = float(initial_amount)
        self.num_stock_shares = np.array(num_stock_shares, dtype=np.float32)
        self.buy_cost_pct = np.array(buy_cost_pct, dtype=np.float32)
        self.sell_cost_pct = np.array(sell_cost_pct, dtype=np.float32)
        self.reward_scaling = float(reward_scaling)
        self.tech_indicator_list = list(tech_indicator_list)
        self.max_memory_length = max_memory_length
        
        # Risk parameters
        self.turbulence_threshold = turbulence_threshold
        self.risk_indicator_col = risk_indicator_col
        self.initial = initial
        self.previous_state = previous_state
        
        # Risk thresholds
        self.risk_thresholds = {
            'vix_normal': 20,
            'vix_elevated': 30,
            'vix_extreme': 40,
            'turbulence_normal': 1.0,
            'turbulence_elevated': 2.0,
            'turbulence_extreme': 3.0
        }
        
        # Position sizing based on risk
        self.position_sizing = {
            'normal': 1.0,
            'elevated': 0.5,
            'extreme': 0.25,
            'crisis': 0
        }
        
        # Initialize spaces and state
        self._initialize_spaces()
        self.terminal = False
        self.state = self._initiate_state()
        
        # Initialize memory with deques for bounded memory usage
        self._initialize_memory()
        
        # Initialize trading statistics
        self.trades = 0
        self.cost = 0
        
        # Set random seed
        self._seed()

    def _validate_dataframe(self, df: pd.DataFrame) -> None:
        """Validate the input DataFrame has required columns"""
        required_columns = {'date', 'tic', 'close'}
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise ValueError(f"DataFrame missing required columns: {missing_columns}")

    def _validate_parameters(self, stock_dim, hmax, initial_amount, 
                           num_stock_shares, buy_cost_pct, sell_cost_pct) -> None:
        """Validate input parameters"""
        if not all(isinstance(x, (int, float)) for x in [stock_dim, hmax, initial_amount]):
            raise ValueError("stock_dim, hmax, and initial_amount must be numeric")
        
        if len(num_stock_shares) != stock_dim:
            raise ValueError("Length of num_stock_shares must match stock_dim")
        
        if len(buy_cost_pct) != stock_dim or len(sell_cost_pct) != stock_dim:
            raise ValueError("Length of cost percentages must match stock_dim")

    def _calculate_state_space(self) -> int:
        """Calculate the total dimension of the state space"""
        # Base state components: [cash] + [prices] + [shares]
        state_space = 1 + self.stock_dim + self.stock_dim
        
        # Add technical indicators
        tech_indicators = [tech for tech in self.tech_indicator_list 
                         if tech not in ['vix', 'turbulence']]
        
        # For multiple stocks
        if len(self.df.tic.unique()) > 1:
            state_space += len(tech_indicators) * self.stock_dim
        else:
            state_space += len(tech_indicators)
        
        # Add risk indicators (market-wide)
        if 'vix' in self.tech_indicator_list:
            state_space += 1
        if 'turbulence' in self.tech_indicator_list:
            state_space += 1
            
        return state_space

    def _initialize_spaces(self) -> None:
        """Initialize action and observation spaces"""
        self.state_space = self._calculate_state_space()
        self.action_space = spaces.Box(
            low=-1, high=1, 
            shape=(self.stock_dim,), 
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.state_space,), 
            dtype=np.float32
        )

    def _initialize_memory(self) -> None:
        """Initialize memory structures with bounded size"""
        self.asset_memory = deque(maxlen=self.max_memory_length)
        self.rewards_memory = deque(maxlen=self.max_memory_length)
        self.actions_memory = deque(maxlen=self.max_memory_length)
        self.date_memory = deque(maxlen=self.max_memory_length)
        self.trading_memory = {
            'buys': deque(maxlen=self.max_memory_length),
            'sells': deque(maxlen=self.max_memory_length),
            'holds': deque(maxlen=self.max_memory_length),
            'costs': deque(maxlen=self.max_memory_length)
        }

    def _calculate_vix_risk(self, vix_value: float) -> str:
        """Calculate risk level based on VIX value"""
        if vix_value >= self.risk_thresholds['vix_extreme']:
            return 'extreme'
        elif vix_value >= self.risk_thresholds['vix_elevated']:
            return 'elevated'
        return 'normal'

    def _calculate_turb_risk(self, turb_value: float) -> str:
        """Calculate risk level based on turbulence value"""
        if turb_value >= self.risk_thresholds['turbulence_extreme']:
            return 'extreme'
        elif turb_value >= self.risk_thresholds['turbulence_elevated']:
            return 'elevated'
        return 'normal'

    def _get_position_size(self, risk_level: str) -> float:
        """Get position size multiplier based on risk level"""
        return self.position_sizing.get(risk_level, self.position_sizing['extreme'])

    def _assess_market_risk(self) -> dict:
        """Assess market risk with proper error handling"""
        try:
            vix_value = self.data['vix'].values[0] if 'vix' in self.data.columns else None
            turb_value = self.data[self.risk_indicator_col].values[0] if self.risk_indicator_col in self.data.columns else None
            
            # Default to elevated risk if indicators are missing
            vix_risk = 'elevated' if vix_value is None else self._calculate_vix_risk(vix_value)
            turb_risk = 'elevated' if turb_value is None else self._calculate_turb_risk(turb_value)
            
            # Determine position sizing
            position_size_mult = min(
                self._get_position_size(vix_risk),
                self._get_position_size(turb_risk)
            )
            
            return {
                'vix_risk': vix_risk,
                'turbulence_risk': turb_risk,
                'position_size_mult': position_size_mult,
                'vix_value': vix_value,
                'turbulence_value': turb_value
            }
        except Exception as e:
            logger.error(f"Error in risk assessment: {e}")
            return {
                'vix_risk': 'extreme',
                'turbulence_risk': 'extreme',
                'position_size_mult': self.position_sizing['extreme'],
                'vix_value': None,
                'turbulence_value': None
            }

    def _initiate_state(self) -> np.ndarray:
        """Initialize the state space"""
        try:
            if len(self.df.tic.unique()) > 1:
                # Multiple stocks
                state = [self.initial_amount] + \
                        self.data.close.values.tolist() + \
                        self.num_stock_shares.tolist()
                
                # Add technical indicators
                for tech in self.tech_indicator_list:
                    if tech not in ['vix', 'turbulence']:
                        state += self.data[tech].values.tolist()
                
                # Add risk indicators
                if 'vix' in self.tech_indicator_list:
                    state.append(self.data['vix'].values[0])
                if 'turbulence' in self.tech_indicator_list:
                    state.append(self.data['turbulence'].values[0])
            
            else:
                # Single stock
                state = [self.initial_amount] + \
                        [self.data.close] + \
                        self.num_stock_shares.tolist()
                
                # Add technical indicators
                for tech in self.tech_indicator_list:
                    if tech not in ['vix', 'turbulence']:
                        state.append(self.data[tech])
                
                # Add risk indicators
                if 'vix' in self.tech_indicator_list:
                    state.append(self.data['vix'])
                if 'turbulence' in self.tech_indicator_list:
                    state.append(self.data['turbulence'])
            
            return np.array(state, dtype=np.float32)
        
        except Exception as e:
            logger.error(f"Error initiating state: {e}")
            raise ValueError("Failed to initialize state")

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """Execute one step in the environment"""
        try:
            self.terminal = self.day >= len(self.df.index.unique()) - 1
            
            if self.terminal:
                return self.state, 0, True, {'date': self._get_date()}
            
            # Get current risk assessment
            risk_assessment = self._assess_market_risk()
            position_size_mult = risk_assessment['position_size_mult']
            
            # Calculate beginning total asset value
            begin_total_asset = self.state[0] + \
                sum(np.array(self.num_stock_shares) * np.array(self.state[1:1+self.stock_dim]))
            
            # Process actions
            actions = np.clip(actions, -1, 1)  # Ensure actions are in valid range
            
            # Handle sells first
            self._handle_sells(actions, position_size_mult)
            
            # Handle buys second
            self._handle_buys(actions, position_size_mult)
            
            # Move to next day
            self.day += 1
            self.data = self.df.loc[self.day, :]
            
            # Update state
            self.state = self._update_state()
            
            # Calculate reward
            end_total_asset = self.state[0] + \
                sum(np.array(self.num_stock_shares) * np.array(self.state[1:1+self.stock_dim]))
            reward = (end_total_asset - begin_total_asset) * self.reward_scaling
            
            # Update memory
            self._update_memory(end_total_asset, reward, actions)
            
            return self.state, reward, False, {
                'date': self._get_date(),
                'cost': self.cost,
                'trades': self.trades,
                'risk_assessment': risk_assessment
            }
            
        except Exception as e:
            logger.error(f"Error in step: {e}")
            return self.state, 0, True, {'error': str(e)}

    def _handle_sells(self, actions: np.ndarray, position_size_mult: float) -> None:
        """Handle sell actions"""
        sell_index = np.where(actions < 0)[0]
        
        for index in sell_index:
            if self.state[index + self.stock_dim + 1] > 0:  # If we have shares
                sell_num_shares = min(
                    abs(actions[index]) * self.state[index + self.stock_dim + 1],
                    self.state[index + self.stock_dim + 1]
                )
                
                sell_amount = self.state[index + 1] * sell_num_shares * (1 - self.sell_cost_pct[index])
                
                self.state[0] += sell_amount
                self.state[index + self.stock_dim + 1] -= sell_num_shares
                self.cost += self.state[index + 1] * sell_num_shares * self.sell_cost_pct[index]
                self.trades += 1
                
                self.trading_memory['sells'].append({
                    'day': self.day,
                    'stock': index,
                    'shares': sell_num_shares,
                    'price': self.state[index + 1],
                    'amount': sell_amount
                })

    def _handle_buys(self, actions: np.ndarray, position_size_mult: float) -> None:
        """Handle buy actions"""
        buy_index = np.where(actions > 0)[0]
        
        for index in buy_index:
            if self.state[0] > 0:  # If we have cash
                max_buy = self.state[0] // (self.state[index + 1] * (1 + self.buy_cost_pct[index]))
                
                buy_num_shares = min(
                    max_buy,
                    self.hmax - self.state[index + self.stock_dim + 1],
                    actions[index] * self.hmax * position_size_mult
                )
                
                buy_amount = self.state[index + 1] * buy_num_shares * (1 + self.buy_cost_pct[index])
                
                self.state[0] -= buy_amount
                self.state[index + self.stock_dim + 1] += buy_num_shares
                self.cost += self.state[index + 1] * buy_num_shares * self.buy_cost_pct[index]
                self.trades += 1
                
                self.trading_memory['buys'].append({
                    'day': self.day,
                    'stock': index,
                    'shares': buy_num_shares,
                    'price': self.state[index + 1],
                    'amount': buy_amount
                })

    def _update_state(self) -> np.ndarray:
        """Update the state space with new day's data"""
        try:
            if len(self.df.tic.unique()) > 1:
                # Multiple stocks
                state = [self.state[0]] + \
                        self.data.close.values.tolist() + \
                        list(self.state[self.stock_dim + 1:self.stock_dim * 2 + 1])
                
                # Add technical indicators
                for tech in self.tech_indicator_list:
                    if tech not in ['vix', 'turbulence']:
                        state += self.data[tech].values.tolist()
                
                # Add risk indicators
                if 'vix' in self.tech_indicator_list:
                    state.append(self.data['vix'].values[0])
                if 'turbulence' in self.tech_indicator_list:
                    state.append(self.data['turbulence'].values[0])
            
            else:
                # Single stock
                state = [self.state[0]] + \
                        [self.data.close] + \
                        list(self.state[self.stock_dim + 1:self.stock_dim * 2 + 1])
                
                # Add technical indicators
                for tech in self.tech_indicator_list:
                    if tech not in ['vix', 'turbulence']:
                        state.append(self.data[tech])
                
                # Add risk indicators
                if 'vix' in self.tech_indicator_list:
                    state.append(self.data['vix'])
                if 'turbulence' in self.tech_indicator_list:
                    state.append(self.data['turbulence'])
            
            return np.array(state, dtype=np.float32)
        
        except Exception as e:
            logger.error(f"Error updating state: {e}")
            raise ValueError("Failed to update state")

    def _update_memory(self, end_total_asset: float, reward: float, actions: np.ndarray) -> None:
        """Update memory structures with new data"""
        self.asset_memory.append(end_total_asset)
        self.rewards_memory.append(reward)
        self.actions_memory.append(actions)
        self.date_memory.append(self._get_date())

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, dict]:
        """Reset the environment to initial state"""
        try:
            # Reset random seed if provided
            if seed is not None:
                self._seed(seed)
            
            self.day = 0
            self.data = self.df.loc[self.day, :]
            self.terminal = False
            
            # Reset state
            if self.initial:
                self.state = self._initiate_state()
            else:
                previous_total_asset = self.previous_state[0] + \
                    sum(np.array(self.num_stock_shares) * 
                        np.array(self.previous_state[1:1+self.stock_dim]))
                self.state = [previous_total_asset] + self.previous_state[1:]
                self.state = np.array(self.state, dtype=np.float32)
            
            # Reset memory structures
            self._initialize_memory()
            
            # Reset trading statistics
            self.trades = 0
            self.cost = 0
            
            return self.state, {'date': self._get_date()}
        
        except Exception as e:
            logger.error(f"Error in reset: {e}")
            raise ValueError("Failed to reset environment")

    def render(self, mode='human') -> Optional[np.ndarray]:
        """Render the environment"""
        if mode != 'human':
            raise NotImplementedError(f"{mode} mode is not supported")
        
        try:
            self._render_summary()
            self._render_risk_assessment()
            self._render_holdings()
            self._render_trading_stats()
            
            return self.state
        
        except Exception as e:
            logger.error(f"Error in render: {e}")
            return None

    def _render_summary(self) -> None:
        """Render summary information"""
        print(f"\n{'='*50}")
        print(f"Date: {self._get_date()}")
        print(f"Portfolio Value: ${self.asset_memory[-1]:,.2f}")
        print(f"Cash: ${self.state[0]:,.2f}")

    def _render_risk_assessment(self) -> None:
        """Render risk assessment information"""
        risk_assessment = self._assess_market_risk()
        print(f"\nRisk Assessment:")
        print(f"VIX Risk Level: {risk_assessment['vix_risk']}")
        print(f"Turbulence Risk Level: {risk_assessment['turbulence_risk']}")
        print(f"Position Size Multiplier: {risk_assessment['position_size_mult']:.2f}")

    def _render_holdings(self) -> None:
        """Render current holdings information"""
        print(f"\nHoldings:")
        for i in range(self.stock_dim):
            current_price = self.state[i + 1]
            holdings = self.state[i + self.stock_dim + 1]
            position_value = current_price * holdings
            print(f"Stock {i}: {holdings:.0f} shares @ ${current_price:.2f} = ${position_value:,.2f}")

    def _render_trading_stats(self) -> None:
        """Render trading statistics"""
        print(f"\nTrading Statistics:")
        print(f"Total Trades: {self.trades}")
        print(f"Total Trading Cost: ${self.cost:,.2f}")
        print(f"{'='*50}")

    def get_sb_env(self) -> Tuple[DummyVecEnv, np.ndarray]:
        """Get stable-baselines environment wrapper"""
        try:
            e = DummyVecEnv([lambda: self])
            obs = e.reset()
            return e, obs
        except Exception as e:
            logger.error(f"Error creating stable-baselines environment: {e}")
            raise ValueError("Failed to create stable-baselines environment")

    def _seed(self, seed: Optional[int] = None) -> List[int]:
        """Set random seed"""
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _get_date(self) -> str:
        """Get current date"""
        try:
            if len(self.df.tic.unique()) > 1:
                date = self.data.date.unique()[0]
            else:
                date = self.data.date
            return str(date)
        except Exception as e:
            logger.error(f"Error getting date: {e}")
            return "Unknown Date"

    def get_portfolio_stats(self) -> dict:
        """Get portfolio statistics"""
        try:
            current_value = self.asset_memory[-1]
            initial_value = self.asset_memory[0]
            total_return = (current_value - initial_value) / initial_value
            
            returns = np.diff(self.asset_memory) / self.asset_memory[:-1]
            sharpe_ratio = np.mean(returns) / np.std(returns) if len(returns) > 1 else 0
            
            return {
                'current_value': current_value,
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'total_trades': self.trades,
                'total_costs': self.cost,
                'returns': returns.tolist()
            }
        except Exception as e:
            logger.error(f"Error calculating portfolio stats: {e}")
            return {}

    def save_memory_to_df(self, path: str) -> None:
        """Save trading memory to CSV file"""
        try:
            memory_df = pd.DataFrame({
                'date': list(self.date_memory),
                'portfolio_value': list(self.asset_memory),
                'rewards': list(self.rewards_memory)
            })
            memory_df.to_csv(path, index=False)
            logger.info(f"Successfully saved memory to {path}")
        except Exception as e:
            logger.error(f"Error saving memory to CSV: {e}")

    def close(self) -> None:
        """Clean up environment resources"""
        pass
