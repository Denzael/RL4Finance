# trading_env.py
import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from gymnasium.utils import seeding
from typing import Dict, List, Optional, Tuple, Any
from stable_baselines3.common.vec_env import DummyVecEnv

class StockTradingEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        stock_dim: int,
        hmax: int,
        initial_amount: float,
        num_stock_shares: List[int],
        buy_cost_pct: List[float],
        sell_cost_pct: List[float],
        reward_scaling: float,
        state_space: int,
        action_space: int,
        tech_indicator_list: List[str],
        turbulence_threshold: Optional[float] = None,
        risk_indicator_col: str = "turbulence",
        make_plots: bool = False,
        print_verbosity: int = 10,
        day: int = 0,
        initial: bool = True,
        previous_state: Optional[List] = None,
        model_name: str = "",
        mode: str = "",
        iteration: str = "",
        hybrid_strategy = None,
    ):
        self.df = df
        self.stock_dim = stock_dim
        self.hmax = hmax
        self.initial_amount = initial_amount
        self.num_stock_shares = num_stock_shares
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.reward_scaling = reward_scaling
        self.state_space = state_space
        self.action_space = action_space
        self.tech_indicator_list = tech_indicator_list
        self.turbulence_threshold = turbulence_threshold
        self.risk_indicator_col = risk_indicator_col
        self.make_plots = make_plots
        self.print_verbosity = print_verbosity
        self.day = day
        self.initial = initial
        self.previous_state = previous_state or []
        self.model_name = model_name
        self.mode = mode
        self.iteration = iteration
        self.hybrid_strategy = hybrid_strategy

        # Spaces
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(self.action_space,), dtype=np.float32
        )
        
        # State space includes strategy signals
        signal_dim = 4  # macd, bollinger, rscci, combined
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.state_space + signal_dim,),
            dtype=np.float32
        )

        # Initialize state
        self.data = self.df.loc[self.day, :]
        self.state = self._initiate_state()
        
        # Initialize other variables
        self.terminal = False
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.episode = 0
        self.asset_memory = [self._calculate_total_asset()]
        self.rewards_memory = []
        self.actions_memory = []
        self.state_memory = []
        self.date_memory = [self._get_date()]

    def _calculate_total_asset(self) -> float:
        """Calculate total asset value"""
        return self.state[0] + sum(
            np.array(self.state[1:self.stock_dim + 1]) *
            np.array(self.state[self.stock_dim + 1:self.stock_dim * 2 + 1])
        )

    def _initiate_state(self) -> np.ndarray:
        """Initialize the state"""
        if self.initial:
            # For initial state
            state = (
                [self.initial_amount] +  # Cash
                self._get_stock_prices() +  # Stock prices
                self.num_stock_shares +  # Stock shares
                self._get_tech_indicators()  # Technical indicators
            )
        else:
            # Using previous state
            state = (
                [self.previous_state[0]] +  # Cash
                self._get_stock_prices() +  # Stock prices
                list(self.previous_state[self.stock_dim + 1:self.stock_dim * 2 + 1]) +  # Shares
                self._get_tech_indicators()  # Technical indicators
            )

        # Add strategy signals if hybrid strategy is enabled
        if self.hybrid_strategy is not None:
            strategy_signals = self.hybrid_strategy.get_signal_features(state, self.data)
            state.extend([v[0] for v in strategy_signals.values()])

        return np.array(state, dtype=np.float32)

    def _get_stock_prices(self) -> List[float]:
        """Get current stock prices"""
        if len(self.df.tic.unique()) > 1:
            return self.data.close.values.tolist()
        return [self.data.close]

    def _get_tech_indicators(self) -> List[float]:
        """Get technical indicators"""
        if len(self.df.tic.unique()) > 1:
            return sum(
                (self.data[tech].values.tolist() for tech in self.tech_indicator_list),
                []
            )
        return sum(([self.data[tech]] for tech in self.tech_indicator_list), [])

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one time step within the environment"""
        self.terminal = self.day >= len(self.df.index.unique()) - 1

        if self.terminal:
            return self._return_terminal_step()

        # Process the actions
        actions = self._process_actions(actions)

        # Calculate immediate reward
        begin_total_asset = self._calculate_total_asset()
        
        # Execute trades
        self._execute_trades(actions)

        # Move to next time step
        self.day += 1
        self.data = self.df.loc[self.day, :]
        
        # Update state
        self.state = self._update_state()

        # Calculate reward
        end_total_asset = self._calculate_total_asset()
        self.asset_memory.append(end_total_asset)
        self.date_memory.append(self._get_date())
        
        self.reward = self.reward_scaling * (end_total_asset - begin_total_asset)
        self.rewards_memory.append(self.reward)

        self.state_memory.append(self.state)

        return self.state, self.reward, self.terminal, False, {}

    def _process_actions(self, actions: np.ndarray) -> np.ndarray:
        """Process and validate actions"""
        if self.turbulence_threshold is not None:
            if self.turbulence >= self.turbulence_threshold:
                actions = np.array([-self.hmax] * self.stock_dim)
        
        if self.hybrid_strategy is not None:
            actions = self.hybrid_strategy.combine_signals(actions, self.state, self.data)
        
        return actions * self.hmax

    def _execute_trades(self, actions: np.ndarray):
        """Execute trading actions"""
        argsort_actions = np.argsort(actions)
        sell_index = argsort_actions[: np.where(actions < 0)[0].shape[0]]
        buy_index = argsort_actions[::-1][: np.where(actions > 0)[0].shape[0]]

        for index in sell_index:
            self._sell_stock(index, actions[index])

        for index in buy_index:
            self._buy_stock(index, actions[index])

        self.actions_memory.append(actions)

    def _update_state(self) -> np.ndarray:
        """Update the state"""
        state = (
            [self.state[0]] +  # Cash
            self._get_stock_prices() +  # Stock prices
            list(self.state[self.stock_dim + 1:self.stock_dim * 2 + 1]) +  # Shares
            self._get_tech_indicators()  # Technical indicators
        )

        if self.hybrid_strategy is not None:
            strategy_signals = self.hybrid_strategy.get_signal_features(state, self.data)
            state.extend([v[0] for v in strategy_signals.values()])

        return np.array(state, dtype=np.float32)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """Reset the environment"""
        self.day = 0
        self.data = self.df.loc[self.day, :]
        self.state = self._initiate_state()
        
        self.asset_memory = [self._calculate_total_asset()]
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.terminal = False
        self.rewards_memory = []
        self.actions_memory = []
        self.date_memory = [self._get_date()]
        
        self.episode += 1

        return self.state, {}

    def _get_date(self) -> str:
        """Get current date
        
        Returns:
            str: The current date for the dataset
        """
        return str(self.data.date)

    def _buy_stock(self, index: int, action: float) -> float:
        """Execute buy order"""
        # Check if stock is tradable
        if self.state[index + 2 * self.stock_dim + 1]:
            return 0
            
        # Calculate available amount considering transaction costs
        available_amount = self.state[0] // (
            self.state[index + 1] * (1 + self.buy_cost_pct[index])
        )
        
        # Calculate number of shares to buy
        buy_num_shares = min(available_amount, abs(action))
        buy_amount = self.state[index + 1] * buy_num_shares * (1 + self.buy_cost_pct[index])
        
        # Update state
        self.state[0] -= buy_amount  # Reduce cash
        self.state[index + self.stock_dim + 1] += buy_num_shares  # Increase shares
        self.cost += self.state[index + 1] * buy_num_shares * self.buy_cost_pct[index]
        self.trades += 1
        
        return buy_num_shares

    def _sell_stock(self, index: int, action: float) -> float:
        """Execute sell order"""
        # Check if stock is tradable
        if self.state[index + 2 * self.stock_dim + 1]:
            return 0
            
        # Calculate number of shares to sell
        sell_num_shares = min(abs(action), self.state[index + self.stock_dim + 1])
        sell_amount = (
            self.state[index + 1] * 
            sell_num_shares * 
            (1 - self.sell_cost_pct[index])
        )
        
        # Update state
        self.state[0] += sell_amount  # Increase cash
        self.state[index + self.stock_dim + 1] -= sell_num_shares  # Reduce shares
        self.cost += (
            self.state[index + 1] * 
            sell_num_shares * 
            self.sell_cost_pct[index]
        )
        self.trades += 1
        
        return sell_num_shares

    def _return_terminal_step(self) -> tuple:
        """Process and return terminal state"""
        if self.make_plots:
            self._make_plot()
            
        # Calculate final portfolio value
        end_total_asset = self.state[0] + sum(
            np.array(self.state[1:self.stock_dim + 1]) *
            np.array(self.state[self.stock_dim + 1:self.stock_dim * 2 + 1])
        )
        
        # Calculate statistics
        df_total_value = pd.DataFrame(self.asset_memory, columns=['account_value'])
        df_total_value['date'] = self.date_memory
        df_total_value['daily_return'] = df_total_value['account_value'].pct_change(1)
        
        # Calculate Sharpe ratio if possible
        if df_total_value['daily_return'].std() != 0:
            sharpe = (
                (252 ** 0.5) * 
                df_total_value['daily_return'].mean() / 
                df_total_value['daily_return'].std()
            )
        
        # Print episode summary if needed
        if self.episode % self.print_verbosity == 0:
            self._print_episode_summary(end_total_asset, sharpe)
            
        # Save results if model name and mode are specified
        if self.model_name and self.mode:
            self._save_results(df_total_value)
            
        return self.state, self.reward, self.terminal, False, {}

    def _make_plot(self):
        """Create and save portfolio value plot"""
        plt.figure(figsize=(10, 5))
        plt.plot(self.asset_memory, 'r')
        plt.xlabel('Time Steps')
        plt.ylabel('Portfolio Value')
        plt.title(f'Portfolio Value Over Time - Episode {self.episode}')
        plt.grid(True)
        plt.savefig(f'results/portfolio_value_episode_{self.episode}.png')
        plt.close()

    def _print_episode_summary(self, end_total_asset: float, sharpe: float):
        """Print episode performance summary"""
        print(f"\nEpisode: {self.episode}")
        print(f"Initial Portfolio Value: ${self.asset_memory[0]:,.2f}")
        print(f"Final Portfolio Value: ${end_total_asset:,.2f}")
        print(f"Total Return: ${(end_total_asset - self.asset_memory[0]):,.2f}")
        print(f"Total Cost: ${self.cost:,.2f}")
        print(f"Total Trades: {self.trades}")
        if sharpe:
            print(f"Sharpe Ratio: {sharpe:.3f}")
        print("=" * 50)

    def _save_results(self, df_total_value: pd.DataFrame):
        """Save trading results to files"""
        # Save actions
        df_actions = pd.DataFrame(self.actions_memory)
        if len(self.df.tic.unique()) > 1:
            df_actions.columns = self.df.tic.unique()
        df_actions['date'] = self.date_memory[:-1]
        df_actions.to_csv(
            f'results/actions_{self.mode}_{self.model_name}_{self.iteration}.csv',
            index=False
        )
        
        # Save portfolio values
        df_total_value.to_csv(
            f'results/portfolio_value_{self.mode}_{self.model_name}_{self.iteration}.csv',
            index=False
        )
        
        # Save rewards
        pd.DataFrame({
            'date': self.date_memory[:-1],
            'reward': self.rewards_memory
        }).to_csv(
            f'results/rewards_{self.mode}_{self.model_name}_{self.iteration}.csv',
            index=False
        )

    def render(self, mode='human'):
        """Render the environment"""
        return self.state

    def get_sb_env(self):
        """Get stable-baselines3 environment"""
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs

