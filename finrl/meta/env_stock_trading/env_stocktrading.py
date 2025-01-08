from __future__ import annotations

from typing import List

import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gymnasium import spaces
from gymnasium.utils import seeding
from stable_baselines3.common.vec_env import DummyVecEnv

matplotlib.use("Agg")

class StockTradingEnv(gym.Env):
    """A stock trading environment for OpenAI gym with technical indicators and risk management"""

    metadata = {"render.modes": ["human"]}

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
        state_space: int,
        action_space: int,
        tech_indicator_list: list[str],
        turbulence_threshold=None,
        risk_indicator_col="turbulence",
        vix_col="vix",  # Added VIX column name
        make_plots: bool = False,
        print_verbosity=10,
        day=0,
        initial=True,
        previous_state=[],
        model_name="",
        mode="",
        iteration="",
    ):
        self.day = day
        self.df = df
        self.stock_dim = stock_dim
        self.hmax = hmax
        self.num_stock_shares = num_stock_shares
        self.initial_amount = initial_amount
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.reward_scaling = reward_scaling
        self.state_space = state_space
        self.action_space = action_space
        self.tech_indicator_list = tech_indicator_list
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_space,))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_space,)
        )

        # Trading strategy parameters
        self.strategy_weights = {
            'trend_following': 0.3,
            'mean_reversion': 0.3,
            'risk_management': 0.4
        }

        # Risk management parameters
        self.turbulence_threshold = turbulence_threshold
        self.risk_indicator_col = risk_indicator_col
        self.vix_col = vix_col
        self.risk_multiplier = 1.0
        
        # Environment parameters
        self.terminal = False
        self.make_plots = make_plots
        self.print_verbosity = print_verbosity
        self.initial = initial
        self.previous_state = previous_state
        self.model_name = model_name
        self.mode = mode
        self.iteration = iteration

        # Initialize state
        self.data = self.df.loc[self.day, :]
        self.state = self._initiate_state()

        # Initialize tracking variables
        self.reward = 0
        self.turbulence = 0
        self.vix = 0
        self.cost = 0
        self.trades = 0
        self.episode = 0
        self.asset_memory = [
            self.initial_amount + np.sum(
                np.array(self.num_stock_shares) * np.array(self.state[1 : 1 + self.stock_dim])
            )
        ]
        self.rewards_memory = []
        self.actions_memory = []
        self.state_memory = []
        self.date_memory = [self._get_date()]

        # Initialize random seed
        self._seed()

    def reset(self, seed=None, options=None):
        """
        Reset the environment for a new episode.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options for reset (not used currently)
            
        Returns:
            observation: Initial state of the environment
            info: Additional information (empty dict)
        """
        # Reset seed if provided
        if seed is not None:
            self._seed(seed)
        
        # Reset episode variables
        self.day = 0
        self.data = self.df.loc[self.day, :]
        self.terminal = False
        self.trades = 0
        self.episode += 1
        self.reward = 0
        self.cost = 0
        
        # Reset state memory
        self.asset_memory = [self.initial_amount + np.sum(
            np.array(self.num_stock_shares) * 
            np.array(self.state[1:1 + self.stock_dim])
        )]
        self.rewards_memory = []
        self.actions_memory = []
        self.date_memory = [self._get_date()]
        
        # Get initial state
        self.state = self._initiate_state()
        
        # Reset turbulence and VIX
        if len(self.df.tic.unique()) > 1:
            self.turbulence = self.data[self.risk_indicator_col].values[0]
            if self.vix_col in self.data:
                self.vix = self.data[self.vix_col].values[0]
        else:
            self.turbulence = self.data[self.risk_indicator_col]
            if self.vix_col in self.data:
                self.vix = self.data[self.vix_col]
        
        return self.state, {}
    
    def _initiate_state(self):
        if self.initial:
            # For Initial State
            if len(self.df.tic.unique()) > 1:
                # for multiple stock
                state = (
                    [self.initial_amount]  # cash
                    + self.data.close.values.tolist()  # stock close price
                    + self.num_stock_shares  # stock share
                    + sum(
                        (
                            self.data[tech].values.tolist()
                            for tech in self.tech_indicator_list
                        ),
                        [],
                    )  # technical indicators
                    + [self.data[self.risk_indicator_col].iloc[0]]  # turbulence
                )
                if self.vix_col in self.data:
                    state += [self.data[self.vix_col].iloc[0]]  # VIX
            else:
                # for single stock
                state = (
                    [self.initial_amount]
                    + [self.data.close]
                    + [0] * self.stock_dim
                    + sum(([self.data[tech]] for tech in self.tech_indicator_list), [])
                    + [self.data[self.risk_indicator_col]]
                )
                if self.vix_col in self.data:
                    state += [self.data[self.vix_col]]
        else:
            # Using Previous State
            if len(self.df.tic.unique()) > 1:
                state = (
                    [self.previous_state[0]]
                    + self.data.close.values.tolist()
                    + self.previous_state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)]
                    + sum(
                        (
                            self.data[tech].values.tolist()
                            for tech in self.tech_indicator_list
                        ),
                        [],
                    )
                    + [self.data[self.risk_indicator_col].iloc[0]]
                )
                if self.vix_col in self.data:
                    state += [self.data[self.vix_col].iloc[0]]
            else:
                state = (
                    [self.previous_state[0]]
                    + [self.data.close]
                    + self.previous_state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)]
                    + sum(([self.data[tech]] for tech in self.tech_indicator_list), [])
                    + [self.data[self.risk_indicator_col]]
                )
                if self.vix_col in self.data:
                    state += [self.data[self.vix_col]]
        return state
    def _update_state(self):
        if len(self.df.tic.unique()) > 1:
            # for multiple stock
            state = (
                [self.state[0]]  # cash
                + self.data.close.values.tolist()  # stock close price
                + list(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])  # stock share
                + sum(
                    (
                        self.data[tech].values.tolist()
                        for tech in self.tech_indicator_list
                    ),
                    [],
                )  # technical indicators
                + [self.data[self.risk_indicator_col].iloc[0]]  # turbulence
            )
            if self.vix_col in self.data:
                state += [self.data[self.vix_col].iloc[0]]  # VIX
        else:
            # for single stock
            state = (
                [self.state[0]]  # cash
                + [self.data.close]  # stock close price
                + list(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])  # stock share
                + sum(([self.data[tech]] for tech in self.tech_indicator_list), [])  # technical indicators
                + [self.data[self.risk_indicator_col]]  # turbulence
            )
            if self.vix_col in self.data:
                state += [self.data[self.vix_col]]  # VIX

        return state

    def _sell_stock(self, index, action):
        def _do_sell_normal():
            if (
                self.state[index + 2 * self.stock_dim + 1] != True
            ):  # check if the stock is able to sell
                if self.state[index + self.stock_dim + 1] > 0:
                    # Sell only if current asset is > 0
                    sell_num_shares = min(
                        abs(action), self.state[index + self.stock_dim + 1]
                    )
                    sell_amount = (
                        self.state[index + 1]
                        * sell_num_shares
                        * (1 - self.sell_cost_pct[index])
                    )
                    # update balance
                    self.state[0] += sell_amount

                    self.state[index + self.stock_dim + 1] -= sell_num_shares
                    self.cost += (
                        self.state[index + 1]
                        * sell_num_shares
                        * self.sell_cost_pct[index]
                    )
                    self.trades += 1
                else:
                    sell_num_shares = 0
            else:
                sell_num_shares = 0

            return sell_num_shares

        # perform sell action based on the sign of the action
        if self.turbulence_threshold is not None:
            if self.turbulence >= self.turbulence_threshold:
                if self.state[index + 1] > 0:
                    # Sell only if the price is > 0 (no missing data in this particular date)
                    # if turbulence goes over threshold, just clear out all positions
                    if self.state[index + self.stock_dim + 1] > 0:
                        # Sell only if current asset is > 0
                        sell_num_shares = self.state[index + self.stock_dim + 1]
                        sell_amount = (
                            self.state[index + 1]
                            * sell_num_shares
                            * (1 - self.sell_cost_pct[index])
                        )
                        # update balance
                        self.state[0] += sell_amount
                        self.state[index + self.stock_dim + 1] = 0
                        self.cost += (
                            self.state[index + 1]
                            * sell_num_shares
                            * self.sell_cost_pct[index]
                        )
                        self.trades += 1
                    else:
                        sell_num_shares = 0
                else:
                    sell_num_shares = 0
            else:
                sell_num_shares = _do_sell_normal()
        else:
            sell_num_shares = _do_sell_normal()

        return sell_num_shares

    def _buy_stock(self, index, action):
        def _do_buy():
            if (
                self.state[index + 2 * self.stock_dim + 1] != True
            ):  # check if the stock is able to buy
                available_amount = self.state[0] // (
                    self.state[index + 1] * (1 + self.buy_cost_pct[index])
                )

                # update balance
                buy_num_shares = min(available_amount, action)
                buy_amount = (
                    self.state[index + 1]
                    * buy_num_shares
                    * (1 + self.buy_cost_pct[index])
                )
                self.state[0] -= buy_amount

                self.state[index + self.stock_dim + 1] += buy_num_shares

                self.cost += (
                    self.state[index + 1] * buy_num_shares * self.buy_cost_pct[index]
                )
                self.trades += 1
            else:
                buy_num_shares = 0

            return buy_num_shares

        # perform buy action based on the sign of the action
        if self.turbulence_threshold is None:
            buy_num_shares = _do_buy()
        else:
            if self.turbulence < self.turbulence_threshold:
                buy_num_shares = _do_buy()
            else:
                buy_num_shares = 0
                pass

        return buy_num_shares

    def _handle_terminal(self):
        if self.make_plots:
            self._make_plot()
            
        end_total_asset = self.state[0] + sum(
            np.array(self.state[1 : (self.stock_dim + 1)])
            * np.array(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
        )
        
        # Calculate statistics
        df_total_value = pd.DataFrame(self.asset_memory)
        df_total_value.columns = ["account_value"]
        df_total_value["date"] = self.date_memory
        df_total_value["daily_return"] = df_total_value["account_value"].pct_change(1)
        
        if df_total_value["daily_return"].std() != 0:
            sharpe = (
                (252 ** 0.5)
                * df_total_value["daily_return"].mean()
                / df_total_value["daily_return"].std()
            )
        
        df_rewards = pd.DataFrame(self.rewards_memory)
        df_rewards.columns = ["account_rewards"]
        df_rewards["date"] = self.date_memory[:-1]
        
        # Print metrics if verbose
        if self.episode % self.print_verbosity == 0:
            print(f"day: {self.day}, episode: {self.episode}")
            print(f"begin_total_asset: {self.asset_memory[0]:0.2f}")
            print(f"end_total_asset: {end_total_asset:0.2f}")
            print(f"total_reward: {sum(self.rewards_memory):0.2f}")
            print(f"total_cost: {self.cost:0.2f}")
            print(f"total_trades: {self.trades}")
            if df_total_value["daily_return"].std() != 0:
                print(f"Sharpe: {sharpe:0.3f}")
            print("=================================")
            
        # Save metrics
        if (self.model_name != "") and (self.mode != ""):
            df_actions = self.save_action_memory()
            df_actions.to_csv(
                f"results/actions_{self.mode}_{self.model_name}_{self.iteration}.csv"
            )
            df_total_value.to_csv(
                f"results/account_value_{self.mode}_{self.model_name}_{self.iteration}.csv",
                index=False,
            )
            df_rewards.to_csv(
                f"results/account_rewards_{self.mode}_{self.model_name}_{self.iteration}.csv",
                index=False,
            )
            plt.plot(self.asset_memory, "r")
            plt.savefig(
                f"results/account_value_{self.mode}_{self.model_name}_{self.iteration}.png"
            )
            plt.close()

        return self.state, self.reward, self.terminal, False, {}

    def _calculate_total_asset(self):
        """Calculate total asset value"""
        return self.state[0] + sum(
            np.array(self.state[1:(self.stock_dim + 1)]) *
            np.array(self.state[(self.stock_dim + 1):(self.stock_dim * 2 + 1)])
        )

    def save_asset_memory(self):
        """Save account value memory"""
        date_list = self.date_memory
        asset_list = self.asset_memory
        df_account_value = pd.DataFrame({
            "date": date_list,
            "account_value": asset_list
        })
        return df_account_value

    def save_action_memory(self):
        """Save trading action memory"""
        if len(self.df.tic.unique()) > 1:
            # Date and close price must match actions length
            date_list = self.date_memory[:-1]
            df_date = pd.DataFrame(date_list)
            df_date.columns = ["date"]

            action_list = self.actions_memory
            df_actions = pd.DataFrame(action_list)
            df_actions.columns = self.data.tic.values
            df_actions.index = df_date.date
        else:
            date_list = self.date_memory[:-1]
            action_list = self.actions_memory
            df_actions = pd.DataFrame({
                "date": date_list,
                "actions": action_list
            })
        return df_actions

    def _seed(self, seed=None):
        """Set random seed"""
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_sb_env(self):
        """Get stable-baselines environment"""
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs

    def _get_date(self):
        """Get current date"""
        if len(self.df.tic.unique()) > 1:
            date = self.data.date.unique()[0]
        else:
            date = self.data.date
        return date

    def render(self, mode="human", close=False):
        """Render the environment"""
        return self.state
