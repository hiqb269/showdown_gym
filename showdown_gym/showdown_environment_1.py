import os
import time
from typing import Any, Dict

import numpy as np
from poke_env import (
    AccountConfiguration,
    MaxBasePowerPlayer,
    RandomPlayer,
    SimpleHeuristicsPlayer,
)
from poke_env.battle import AbstractBattle
from poke_env.environment.single_agent_wrapper import SingleAgentWrapper
from poke_env.environment.singles_env import ObsType
from poke_env.player.player import Player

from showdown_gym.base_environment import BaseShowdownEnv


class ShowdownEnvironment(BaseShowdownEnv):

    def __init__(
        self,
        battle_format: str = "gen9randombattle",
        account_name_one: str = "train_one",
        account_name_two: str = "train_two",
        team: str | None = None,
    ):
        super().__init__(
            battle_format=battle_format,
            account_name_one=account_name_one,
            account_name_two=account_name_two,
            team=team,
        )

        self.rl_agent = account_name_one

    def _get_action_size(self) -> int | None:
        """
        None just uses the default number of actions as laid out in process_action - 26 actions.

        This defines the size of the action space for the agent - e.g. the output of the RL agent.

        This should return the number of actions you wish to use if not using the default action scheme.
        """
        # Reduced action space based on expert agent priorities:
        # 4 moves + 6 switches = 10 core actions (removing complex mechanics initially)
        return 10 

    def process_action(self, action: np.int64) -> np.int64:
        """
        Returns the np.int64 relative to the given action.

        The action mapping is as follows:
        action = -2: default
        action = -1: forfeit
        0 <= action <= 5: switch
        6 <= action <= 9: move
        10 <= action <= 13: move and mega evolve
        14 <= action <= 17: move and z-move
        18 <= action <= 21: move and dynamax
        22 <= action <= 25: move and terastallize
        
        Expert agent focuses on core actions: switching and damaging moves.
        Complex mechanics (mega, z-move, dynamax, tera) removed for cleaner learning.
        
        Edge case handling:
        - Switch to fainted Pokemon -> fallback to first available switch or move
        - Move index out of bounds -> fallback to first available move or switch

        :param action: The action to take.
        :type action: int64

        :return: The battle order ID for the given action in context of the current battle.
        :rtype: np.Int64
        """
        # Get current battle state for validation
        battle = self.battle1
        if not battle:
            return np.int64(6)  # Default to first move if no battle context
        
        # Map simplified 10-action space to actual battle actions with validation
        if action <= 5:  # Switch actions (0-5)
            # Validate switch target exists and is not fainted
            team_list = list(battle.team.values())
            if action < len(team_list):
                target_pokemon = team_list[action]
                if target_pokemon and not target_pokemon.fainted and target_pokemon != battle.active_pokemon:
                    return np.int64(action)
            
            # Fallback: find first valid switch or default to move
            for i, pokemon in enumerate(team_list):
                if pokemon and not pokemon.fainted and pokemon != battle.active_pokemon:
                    return np.int64(i)
            
            # No valid switches available, fallback to first move
            return np.int64(6)
            
        elif action <= 9:  # Move actions (6-9 -> map to moves 0-3)
            # Validate move index exists
            move_index = action - 6  # Convert to 0-3 range
            if battle.active_pokemon and battle.active_pokemon.moves:
                available_moves = list(battle.active_pokemon.moves.values())
                if move_index < len(available_moves):
                    return np.int64(action)
                    
                # Move index out of bounds, fallback to first available move
                if available_moves:
                    return np.int64(6)  # First move
            
            # No moves available, try to switch
            team_list = list(battle.team.values())
            for i, pokemon in enumerate(team_list):
                if pokemon and not pokemon.fainted and pokemon != battle.active_pokemon:
                    return np.int64(i)
            
            # Last resort: forfeit (should rarely happen)
            return np.int64(-1)
        else:
            # Action out of range, fallback to first move
            return np.int64(6)

    def get_additional_info(self) -> Dict[str, Dict[str, Any]]:
        info = super().get_additional_info()

        # Add any additional information you want to include in the info dictionary that is saved in logs
        # For example, you can add the win status

        if self.battle1 is not None:
            agent = self.possible_agents[0]
            info[agent]["win"] = self.battle1.won
            
            # Add strategic metrics from expert agent analysis
            active_pokemon = self.battle1.active_pokemon
            if active_pokemon:
                info[agent]["active_hp"] = active_pokemon.current_hp_fraction
                info[agent]["team_alive"] = len([p for p in self.battle1.team.values() if not p.fainted])
                info[agent]["opp_alive"] = len([p for p in self.battle1.opponent_team.values() if not p.fainted])
                info[agent]["turn"] = self.battle1.turn
                
                # Track damage potential (expert focuses on damage estimation)
                if self.battle1.available_moves:
                    max_damage = 0
                    for move in self.battle1.available_moves:
                        if move.base_power > 0:
                            damage_est = self._estimate_damage_simple(move, active_pokemon, self.battle1.opponent_active_pokemon)
                            max_damage = max(max_damage, damage_est)
                    info[agent]["max_damage_potential"] = max_damage
                
                # Track if in danger (expert switches when in danger)
                info[agent]["in_danger"] = active_pokemon.current_hp_fraction < 0.5

        return info

    def calc_reward(self, battle: AbstractBattle) -> float:
        """
        Calculates the reward based on priorities
        1. Guaranteed knockouts (highest priority)
        2. Avoiding being knocked out (switching when in danger)
        3. Healing when low on health
        4. Dealing maximum damage
        5. Type effectiveness and strategic positioning

        Args:
            battle (AbstractBattle): The current battle instance containing information
                about the player's team and the opponent's team from the player's perspective.
            prior_battle (AbstractBattle): The prior battle instance to compare against.
        Returns:
            float: The calculated reward based on expert-informed strategic priorities.
        """

        prior_battle = self._get_prior_battle(battle)
        reward = 0.0

        # Basic health tracking
        health_team = [mon.current_hp_fraction for mon in battle.team.values()]
        health_opponent = [
            mon.current_hp_fraction for mon in battle.opponent_team.values()
        ]

        if len(health_opponent) < len(health_team):
            health_opponent.extend([1.0] * (len(health_team) - len(health_opponent)))

        if prior_battle is not None:
            prior_health_opponent = [
                mon.current_hp_fraction for mon in prior_battle.opponent_team.values()
            ]
            if len(prior_health_opponent) < len(health_team):
                prior_health_opponent.extend([1.0] * (len(health_team) - len(prior_health_opponent)))

            prior_health_team = [mon.current_hp_fraction for mon in prior_battle.team.values()]

            prior_opp_alive = len([p for p in prior_battle.opponent_team.values() if not p.fainted])
            current_opp_alive = len([p for p in battle.opponent_team.values() if not p.fainted])

            prior_my_alive = len([p for p in prior_battle.team.values() if not p.fainted])
            current_my_alive = len([p for p in battle.team.values() if not p.fainted])

            # PRIORITY 1: Knockout rewards (guaranteed KO is top priority)
            if (prior_opp_alive - current_opp_alive) > 0:
                ko_reward = (prior_opp_alive - current_opp_alive) * 2.0  # Moderate reward for KOs
                reward += ko_reward

            # PRIORITY 2: Survival penalty (avoid being KO'd)
            survival_penalty = (prior_my_alive - current_my_alive) * -3.0  # Significant but not overwhelming penalty
            reward += survival_penalty

            # PRIORITY 3: Damage dealing (core strategy)
            opp_damage = np.sum(np.array(prior_health_opponent) - np.array(health_opponent))
            my_damage = np.sum(np.array(prior_health_team) - np.array(health_team))
            damage_reward = opp_damage * 1.0 - my_damage * 0.5  # Prefer dealing damage over taking it
            reward += damage_reward

            # PRIORITY 4: Smart switching rewards (expert switches when in danger)
            active_pokemon = battle.active_pokemon
            prior_active = prior_battle.active_pokemon
            
            # Reward switching when previous Pokemon was in danger
            if (prior_active and active_pokemon and 
                prior_active.species != active_pokemon.species and
                prior_active.current_hp_fraction < 0.5):
                reward += 0.5  # Small strategic switch bonus
            
            # PRIORITY 5: Healing move rewards (expert uses healing when low HP)
            if (active_pokemon and prior_active and 
                active_pokemon.species == prior_active.species and
                active_pokemon.current_hp_fraction > prior_active.current_hp_fraction):
                # Pokemon gained health - likely used healing move
                heal_amount = active_pokemon.current_hp_fraction - prior_active.current_hp_fraction
                reward += heal_amount * 0.01  # Small healing reward
            

        # PRIORITY 6: Turn efficiency (not sure if needed)
        if battle.turn > 30:  # Penalize overly long battles
            reward -= 0.001 * (battle.turn - 30)  # Very small turn penalty

        # Final battle outcome (expert's ultimate goal)
        if battle.finished:
            if battle.won:
                reward += 5.0  # Moderate win bonus
            else:
                reward -= 3.0  # Moderate loss penalty

        return reward

    def _observation_size(self) -> int:
        """
        Returns the size of the observation size to create the observation space for all possible agents in the environment.

        Expert-informed observation size based on decision factors:
        - Health states (team + opponent): 12 features
        - Damage potential estimations: 8 features  
        - Type effectiveness: 4 features
        - Battle state: 4 features
        - Immediate danger assessment: 2 features
        Total: 30 features for strategic decision making

        Returns:
            int: The size of the observation space.
        """
        return 30

    def embed_battle(self, battle: AbstractBattle) -> np.ndarray:
        """
        Embeds battle state based on expert agent decision-making patterns.
        
        Expert agent considers:
        1. Health states (immediate danger assessment)
        2. Damage potential (best move calculations)
        3. Type effectiveness (damage multipliers)
        4. Strategic positioning (KO opportunities)

        Args:
            battle (AbstractBattle): The current battle instance containing information about
                the player's team and the opponent's team.
        Returns:
            np.float32: A 30-dimensional numpy array for strategic decision making.
        """

        # 1. Basic health information (12 features) - foundation for all expert decisions
        health_team = [mon.current_hp_fraction for mon in battle.team.values()]
        health_opponent = [
            mon.current_hp_fraction for mon in battle.opponent_team.values()
        ]

        # Ensure consistent sizing
        if len(health_opponent) < len(health_team):
            health_opponent.extend([1.0] * (len(health_team) - len(health_opponent)))

        # 2. Damage potential assessment (8 features) - expert's core decision factor
        damage_features = self._get_damage_features(battle)
        
        # 3. Type effectiveness (4 features) - expert uses type multipliers
        type_features = self._get_type_features(battle)
        
        # 4. Battle state information (4 features) - strategic context
        battle_state = self._get_battle_state_features(battle)
        
        # 5. Immediate danger assessment (2 features) - expert's switching trigger
        danger_features = self._get_danger_features(battle)

        # Assemble final vector (30 features total)
        final_vector = np.concatenate([
            health_team,        # 6 features
            health_opponent,    # 6 features  
            damage_features,    # 8 features
            type_features,      # 4 features
            battle_state,       # 4 features
            danger_features,    # 2 features
        ])

        return final_vector.astype(np.float32)

    def _get_damage_features(self, battle: AbstractBattle) -> np.ndarray:
        """Extract damage-related features that mirror expert agent's damage calculations."""
        features = np.zeros(8)
        
        active = battle.active_pokemon
        opponent = battle.opponent_active_pokemon
        
        if active and opponent:
            # Best move damage potential (expert's primary calculation)
            if battle.available_moves:
                max_damage = 0.0
                for move in battle.available_moves:
                    if move.base_power > 0:
                        damage = self._estimate_damage_simple(move, active, opponent)
                        max_damage = max(max_damage, damage)
                features[0] = min(max_damage / 200.0, 2.0)  # Normalize but allow > 1 for strong moves
            
            # KO potential (expert prioritizes guaranteed KOs)
            features[1] = 1.0 if features[0] * 200.0 >= opponent.current_hp else 0.0
            
            # Opponent's best damage against us (expert switches when in danger)
            if opponent.moves:
                opp_max_damage = 0.0
                for move in opponent.moves.values():
                    if move.base_power > 0:
                        damage = self._estimate_damage_simple(move, opponent, active)
                        opp_max_damage = max(opp_max_damage, damage)
                features[2] = min(opp_max_damage / 200.0, 2.0)
            
            # Opponent KO potential against us (critical for expert switching)
            features[3] = 1.0 if features[2] * 200.0 >= active.current_hp else 0.0
            
            # Damage advantage ratio
            if features[2] > 0:
                features[4] = min(features[0] / features[2], 3.0)  # Our damage / their damage
            
            # Health-based damage efficiency
            features[5] = active.current_hp_fraction
            features[6] = opponent.current_hp_fraction
            
            # STAB potential (expert considers STAB in damage calculations)
            stab_moves = 0
            if battle.available_moves:
                for move in battle.available_moves:
                    if move.type in active.types:
                        stab_moves += 1
                features[7] = stab_moves / 4.0  # Normalize by max moves
        
        return features

    def _get_type_features(self, battle: AbstractBattle) -> np.ndarray:
        """Extract type effectiveness features used by expert agent."""
        features = np.zeros(4)
        
        active = battle.active_pokemon
        opponent = battle.opponent_active_pokemon
        
        if active and opponent and battle.available_moves:
            # Type advantage of our best move
            best_effectiveness = 0.0
            for move in battle.available_moves:
                if move.base_power > 0:
                    effectiveness = opponent.damage_multiplier(move)
                    best_effectiveness = max(best_effectiveness, effectiveness)
            features[0] = min(best_effectiveness, 4.0) / 4.0  # Normalize
            
            # Defensive type matchup
            if opponent.moves:
                worst_effectiveness = 0.0
                for move in opponent.moves.values():
                    if move.base_power > 0:
                        effectiveness = active.damage_multiplier(move)
                        worst_effectiveness = max(worst_effectiveness, effectiveness)
                features[1] = min(worst_effectiveness, 4.0) / 4.0
            
            # Type advantage ratio
            if features[1] > 0:
                features[2] = min(features[0] / features[1], 4.0) / 4.0
            
            # Resistance factor (expert considers defensive typing)
            if features[1] < 1.0:
                features[3] = (1.0 - features[1])  # Reward resistances
        
        return features

    def _get_battle_state_features(self, battle: AbstractBattle) -> np.ndarray:
        """Extract battle state features for strategic context."""
        features = np.zeros(4)
        
        # Team size advantage (expert considers team composition)
        my_alive = len([p for p in battle.team.values() if not p.fainted])
        opp_alive = len([p for p in battle.opponent_team.values() if not p.fainted])
        features[0] = my_alive / 6.0
        features[1] = opp_alive / 6.0
        
        # Team advantage
        if opp_alive > 0:
            features[2] = my_alive / opp_alive
        
        # Turn pressure (expert prefers efficient battles)
        features[3] = min(battle.turn / 50.0, 1.0)
        
        return features

    def _get_danger_features(self, battle: AbstractBattle) -> np.ndarray:
        """Extract immediate danger assessment - key for expert switching decisions."""
        features = np.zeros(2)
        
        active = battle.active_pokemon
        if active:
            # Low health danger 
            features[0] = 1.0 if active.current_hp_fraction < 0.5 else 0.0
            
            # Immediate KO danger
            opponent = battle.opponent_active_pokemon
            if opponent and opponent.moves:
                for move in opponent.moves.values():
                    if move.base_power > 0:
                        damage = self._estimate_damage_simple(move, opponent, active)
                        if damage >= active.current_hp:
                            features[1] = 1.0
                            break
        
        return features

    def _estimate_damage_simple(self, move, attacker, defender) -> float:
        """Simplified damage estimation based on expert agent's method."""
        if not move or move.base_power == 0 or not attacker or not defender:
            return 0.0
        base_damage = move.base_power
        
        # Type effectiveness
        if defender:
            multiplier = defender.damage_multiplier(move)
        else:
            multiplier = 1.0
        
        # STAB bonus
        if move.type in attacker.types:
            multiplier *= 1.5
        
        return base_damage * multiplier
    


########################################
# DO NOT EDIT THE CODE BELOW THIS LINE #
########################################


class SingleShowdownWrapper(SingleAgentWrapper):
    """
    A wrapper class for the PokeEnvironment that simplifies the setup of single-agent
    reinforcement learning tasks in a Pokémon battle environment.

    This class initializes the environment with a specified battle format, opponent type,
    and evaluation mode. It also handles the creation of opponent players and account names
    for the environment.

    Do NOT edit this class!

    Attributes:
        battle_format (str): The format of the Pokémon battle (e.g., "gen9randombattle").
        opponent_type (str): The type of opponent player to use ("simple", "max", "random").
        evaluation (bool): Whether the environment is in evaluation mode.
    Raises:
        ValueError: If an unknown opponent type is provided.
    """

    def __init__(
        self,
        team_type: str = "random",
        opponent_type: str = "random",
        evaluation: bool = False,
    ):
        opponent: Player
        unique_id = time.strftime("%H%M%S")

        opponent_account = "ot" if not evaluation else "oe"
        opponent_account = f"{opponent_account}_{unique_id}"

        opponent_configuration = AccountConfiguration(opponent_account, None)
        if opponent_type == "simple":
            opponent = SimpleHeuristicsPlayer(
                account_configuration=opponent_configuration
            )
        elif opponent_type == "max":
            opponent = MaxBasePowerPlayer(account_configuration=opponent_configuration)
        elif opponent_type == "random":
            opponent = RandomPlayer(account_configuration=opponent_configuration)
        else:
            raise ValueError(f"Unknown opponent type: {opponent_type}")

        account_name_one: str = "t1" if not evaluation else "e1"
        account_name_two: str = "t2" if not evaluation else "e2"

        account_name_one = f"{account_name_one}_{unique_id}"
        account_name_two = f"{account_name_two}_{unique_id}"

        team = self._load_team(team_type)

        battle_format = "gen9randombattle" if team is None else "gen9ubers"

        primary_env = ShowdownEnvironment(
            battle_format=battle_format,
            account_name_one=account_name_one,
            account_name_two=account_name_two,
            team=team,
        )

        super().__init__(env=primary_env, opponent=opponent)

    def _load_team(self, team_type: str) -> str | None:
        bot_teams_folders = os.path.join(os.path.dirname(__file__), "teams")

        bot_teams = {}

        for team_file in os.listdir(bot_teams_folders):
            if team_file.endswith(".txt"):
                with open(
                    os.path.join(bot_teams_folders, team_file), "r", encoding="utf-8"
                ) as file:
                    bot_teams[team_file[:-4]] = file.read()

        if team_type in bot_teams:
            return bot_teams[team_type]

        return None
