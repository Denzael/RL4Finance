import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
from stable_baselines3.common.vec_env import DummyVecEnv
from typing import List, Dict

class StockTradingEnv(gym.Env):
    """A stock trading environment that includes VIX and turbulence for risk management"""
    
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
    ):
        super().__init__()
        
        # Basic parameters
        self.day = day
        self.df = df
        self.stock_dim = stock_dim
        self.hmax = hmax
        self.initial_amount = initial_amount
        self.num_stock_shares = num_stock_shares
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.reward_scaling = reward_scaling
        self.tech_indicator_list = tech_indicator_list
        
        # Risk parameters
        self.turbulence_threshold = turbulence_threshold
        self.risk_indicator_col = risk_indicator_col
        
        # Calculate actual state space
        self.state_space = self._calculate_state_space()
        self.action_space_size = self.stock_dim
        
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
        
        # Technical indicator weights
        self.indicator_weights = {
            'trend': {
                'sma': 0.3,
                'ema': 0.3,
                'macd': 0.4
            },
            'momentum': {
                'rsi': 0.4,
                'cci': 0.3,
                'dx': 0.3
            },
            'volatility': {
                'bbands': 1.0
            }
        }
        
        # Category weights
        self.category_weights = {
            'trend': 0.4,
            'momentum': 0.3,
            'volatility': 0.3
        }
        
        # Technical thresholds
        self.tech_thresholds = {
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'cci_oversold': -100,
            'cci_overbought': 100,
            'bbands_squeeze': 0.1
        }
        
        # Spaces
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_space_size,))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_space,))
        
        # Initialize state
        self.data = self.df.loc[self.day, :]
        self.terminal = False
        self.state = self._initiate_state()
        self.initial = initial
        self.previous_state = previous_state
        
        # Initialize tracking
        self.asset_memory = [self.initial_amount + np.sum(
            np.array(self.num_stock_shares) * np.array(self.state[1:1 + self.stock_dim])
        )]
        self.rewards_memory = []
        self.actions_memory = []
        self.date_memory = [self._get_date()]
        
        self._initialize_trading_memory()
        self._seed()

    def _calculate_state_space(self):
        """Calculate the total dimension of the state space"""
        # Base state components: [cash] + [prices] + [shares]
        state_space = 1 + self.stock_dim + self.stock_dim
        
        # For multiple stocks
        if len(self.df.tic.unique()) > 1:
            # Add space for technical indicators (excluding vix and turbulence)
            tech_indicators = [tech for tech in self.tech_indicator_list 
                             if tech not in ['vix', 'turbulence']]
            state_space += len(tech_indicators) * self.stock_dim
        else:
            # For single stock, each technical indicator only needs one value
            tech_indicators = [tech for tech in self.tech_indicator_list 
                             if tech not in ['vix', 'turbulence']]
            state_space += len(tech_indicators)
        
        # Add space for risk indicators (these are market-wide, so only add 1 each)
        if 'vix' in self.tech_indicator_list:
            state_space += 1
        if 'turbulence' in self.tech_indicator_list:
            state_space += 1
            
        return state_space

    def _initiate_state(self):
        """Initialize the state space"""
        if len(self.df.tic.unique()) > 1:
            # Basic state components
            state = [self.initial_amount] + \
                    self.data.close.values.tolist() + \
                    self.num_stock_shares
            
            # Add technical indicators
            for tech in self.tech_indicator_list:
                if tech not in ['vix', 'turbulence']:  # Handle regular technical indicators
                    state += self.data[tech].values.tolist()
            
            # Add risk indicators at the end of state
            if 'vix' in self.tech_indicator_list:
                state.append(self.data['vix'].values[0])
            if 'turbulence' in self.tech_indicator_list:
                state.append(self.data['turbulence'].values[0])
        
        else:
            # Single stock case
            state = [self.initial_amount] + \
                    [self.data.close] + \
                    self.num_stock_shares
            
            # Add technical indicators
            for tech in self.tech_indicator_list:
                if tech not in ['vix', 'turbulence']:  # Handle regular technical indicators
                    state.append(self.data[tech])
            
            # Add risk indicators at the end of state
            if 'vix' in self.tech_indicator_list:
                state.append(self.data['vix'])
            if 'turbulence' in self.tech_indicator_list:
                state.append(self.data['turbulence'])
        
        return state
    
    def _update_state(self):
        """Update the state space"""
        if len(self.df.tic.unique()) > 1:
            # Basic state components
            state = [self.state[0]] + \
                    self.data.close.values.tolist() + \
                    list(self.state[self.stock_dim + 1:self.stock_dim * 2 + 1])
            
            # Add technical indicators
            for tech in self.tech_indicator_list:
                if tech not in ['vix', 'turbulence']:  # Handle regular technical indicators
                    state += self.data[tech].values.tolist()
            
            # Add risk indicators at the end of state
            if 'vix' in self.tech_indicator_list:
                state.append(self.data['vix'].values[0])
            if 'turbulence' in self.tech_indicator_list:
                state.append(self.data['turbulence'].values[0])
        
        else:
            # Single stock case
            state = [self.state[0]] + \
                    [self.data.close] + \
                    list(self.state[self.stock_dim + 1:self.stock_dim * 2 + 1])
            
            # Add technical indicators
            for tech in self.tech_indicator_list:
                if tech not in ['vix', 'turbulence']:  # Handle regular technical indicators
                    state.append(self.data[tech])
            
            # Add risk indicators at the end of state
            if 'vix' in self.tech_indicator_list:
                state.append(self.data['vix'])
            if 'turbulence' in self.tech_indicator_list:
                state.append(self.data['turbulence'])
        
        return state

    def _initialize_trading_memory(self):
        """Initialize trading memory for tracking costs and trades"""
        self.trades = 0
        self.cost = 0
        self.trading_memory = {
            'buys': [],
            'sells': [],
            'holds': [],
            'costs': []
        }

    def _assess_market_risk(self):
        """
        Assess market risk based on VIX and turbulence indicators
        Returns risk levels and position sizing multiplier
        """
        # Get current VIX and turbulence values if available
        vix_value = self.data['vix'].values[0] if 'vix' in self.data.columns else 0
        turb_value = self.data[self.risk_indicator_col].values[0] if self.risk_indicator_col in self.data.columns else 0
        
        # Assess VIX risk level
        if vix_value >= self.risk_thresholds['vix_extreme']:
            vix_risk = 'extreme'
        elif vix_value >= self.risk_thresholds['vix_elevated']:
            vix_risk = 'elevated'
        else:
            vix_risk = 'normal'
        
        # Assess turbulence risk level
        if turb_value >= self.risk_thresholds['turbulence_extreme']:
            turb_risk = 'extreme'
        elif turb_value >= self.risk_thresholds['turbulence_elevated']:
            turb_risk = 'elevated'
        else:
            turb_risk = 'normal'
        
        # Determine position sizing based on worst risk level
        if 'extreme' in [vix_risk, turb_risk]:
            position_size_mult = self.position_sizing['extreme']
        elif 'elevated' in [vix_risk, turb_risk]:
            position_size_mult = self.position_sizing['elevated']
        else:
            position_size_mult = self.position_sizing['normal']
        
        return {
            'vix_risk': vix_risk,
            'turbulence_risk': turb_risk,
            'position_size_mult': position_size_mult
        }

    def step(self, actions):
        """
        Execute one step in the environment
        Args:
            actions: array of actions for each stock [-1, 1] where:
                    -1 = sell maximum allowed
                     1 = buy maximum allowed
                     0 = hold
        Returns:
            next_state: new state after action
            reward: reward for the action
            done: whether episode is finished
            info: additional information
        """
        self.terminal = self.day >= len(self.df.index.unique()) - 1
        
        if self.terminal:
            return self.state, 0, True, {'date': self._get_date()}
        
        # Get current risk assessment
        risk_assessment = self._assess_market_risk()
        position_size_mult = risk_assessment['position_size_mult']
        
        # Calculate available amounts
        begin_total_asset = self.state[0] + \
            sum(np.array(self.num_stock_shares) * np.array(self.state[1:1+self.stock_dim]))
        
        # Process each action
        argsort_actions = np.argsort(actions)
        sell_index = argsort_actions[:np.where(actions < 0)[0].shape[0]]
        buy_index = argsort_actions[::-1][:np.where(actions > 0)[0].shape[0]]
        
        for index in sell_index:
            # Implement sell logic
            actions_norm = actions[index]
            if self.state[index + self.stock_dim + 1] > 0:
                # Sell based on action and risk assessment
                sell_num_shares = min(
                    abs(actions_norm) * self.state[index + self.stock_dim + 1],
                    self.state[index + self.stock_dim + 1]
                )
                sell_amount = self.state[index + 1] * sell_num_shares * (1 - self.sell_cost_pct[index])
                
                # Update state
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
        
        for index in buy_index:
            # Implement buy logic
            actions_norm = actions[index]
            if self.state[0] > 0:
                # Calculate maximum shares that can be bought
                max_buy = self.state[0] // (self.state[index + 1] * (1 + self.buy_cost_pct[index]))
                buy_num_shares = min(
                    max_buy,
                    self.hmax - self.state[index + self.stock_dim + 1]
                )
                buy_num_shares = min(
                    buy_num_shares,
                    actions_norm * self.hmax * position_size_mult
                )
                
                buy_amount = self.state[index + 1] * buy_num_shares * (1 + self.buy_cost_pct[index])
                
                # Update state
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
        self.asset_memory   




 def step(self, actions):
        # ... (previous step implementation) ...
        
        # Update memory
        self.asset_memory.append(end_total_asset)
        self.rewards_memory.append(reward)
        self.actions_memory.append(actions)
        self.date_memory.append(self._get_date())
        
        return self.state, reward, False, {
            'date': self._get_date(),
            'cost': self.cost,
            'trades': self.trades,
            'risk_assessment': risk_assessment
        }

    def reset(self):
        """
        Reset the environment to initial state
        Returns: initial state
        """
        self.day = 0
        self.data = self.df.loc[self.day, :]
        self.terminal = False
        
        # Reset state
        if self.initial:
            self.state = self._initiate_state()
        else:
            previous_total_asset = self.previous_state[0] + \
                sum(np.array(self.num_stock_shares) * np.array(self.previous_state[1:1+self.stock_dim]))
            self.state = [previous_total_asset] + self.previous_state[1:]
        
        # Reset memory
        self.asset_memory = [self.initial_amount + 
            np.sum(np.array(self.num_stock_shares) * np.array(self.state[1:1+self.stock_dim]))]
        self.rewards_memory = []
        self.actions_memory = []
        self.date_memory = [self._get_date()]
        
        self._initialize_trading_memory()
        
        return self.state

    def render(self, mode='human'):
        """Render the environment"""
        if mode != 'human':
            raise NotImplementedError(f"{mode} mode is not supported")
        
        print(f"\nDate: {self._get_date()}")
        print(f"Portfolio Value: ${self.asset_memory[-1]:,.2f}")
        print(f"Cash: ${self.state[0]:,.2f}")
        
        risk_assessment = self._assess_market_risk()
        print(f"\nRisk Assessment:")
        print(f"VIX Risk Level: {risk_assessment['vix_risk']}")
        print(f"Turbulence Risk Level: {risk_assessment['turbulence_risk']}")
        print(f"Position Size Multiplier: {risk_assessment['position_size_mult']:.2f}")
        
        print(f"\nHoldings:")
        for i in range(self.stock_dim):
            current_price = self.state[i + 1]
            holdings = self.state[i + self.stock_dim + 1]
            position_value = current_price * holdings
            print(f"Stock {i}: {holdings:.0f} shares @ ${current_price:.2f} = ${position_value:,.2f}")
        
        print(f"\nTrading Statistics:")
        print(f"Total Trades: {self.trades}")
        print(f"Total Trading Cost: ${self.cost:,.2f}")
        print("-" * 50)
        
        return self.state

    def get_sb_env(self):
        """Get stable-baselines environment wrapper"""
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs

    def _seed(self, seed=None):
        """Set random seed"""
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _get_date(self):
        """Get current date"""
        if len(self.df.tic.unique()) > 1:
            date = self.data.date.unique()[0]
        else:
            date = self.data.date
        return date
