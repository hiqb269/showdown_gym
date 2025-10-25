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
from poke_env.battle import AbstractBattle,Move, Pokemon, Weather
from poke_env.environment.single_agent_wrapper import SingleAgentWrapper
from poke_env.environment.singles_env import ObsType
from poke_env.player.player import Player

from showdown_gym.base_environment import BaseShowdownEnv

#Version 4 - Simple reward function - rich observations (train with Rainbow and SACD)
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

    def _get_action_size(self) -> int:
        """
        Returns the size of the simplified action space: 6 switches + 4 moves = 10 actions.
        """
        return 10

    def process_action(self, action: np.int64) -> np.int64:
        """
        Maps a simplified action (0-9) to a valid switch or move action for poke-env.
        0-5: switch to slot (if valid)
        6-9: use move index (if valid)
        Returns a valid poke-env action code, or falls back to first valid move/switch.
        """
        battle = getattr(self, 'battle1', None)
        if battle is None:
            return np.int64(-2)  # default/no-op if no battle
        
        mask = self._get_action_mask(battle)
        valid_indices = np.where(mask == 1)[0]

        if action < 0 or action >= len(mask) or mask[action] == 0:
            if len(valid_indices) > 0:
                # Deterministic correction (pick first valid) or random if you prefer
                action = valid_indices[0]
            else:
                return np.int64(-2)

        # Switch actions: 0-5 or rather 0-4 (excluding active)
        if 0 <= action <= 4:
            available_switches = list(getattr(battle, "available_switches", []) or [])
            available_switches = [mon for mon in available_switches if mon is not battle.active_pokemon]
            if available_switches:
                switch_idx = min(action, len(available_switches) - 1)
                target = available_switches[switch_idx]
                try:
                    team_list = list(battle.team.values())
                    slot_index = team_list.index(target)
                    return np.int64(slot_index)
                except ValueError:
                    pass
            # No valid switches, fallback to first move
            action = 6
        # Move actions: 6-9
        if 6 <= action <= 9:
            move_idx = action - 6
            available_moves = list(getattr(battle, "available_moves", []) or [])
            if available_moves:
                move_idx = min(move_idx, len(available_moves) - 1)
                return np.int64(6 + move_idx)
        # If no valid action, forfeit
        return np.int64(-1)

    def _get_action_mask(self, battle: AbstractBattle) -> np.ndarray:
        """
        Returns a binary mask (size 10): 1 for valid actions, 0 for invalid.
        0–5 = switches, 6–9 = moves.
        """
        mask = np.zeros(10, dtype=np.float32)

        # Switches (0–5)
        available_switches = getattr(battle, "available_switches", []) or []
        # Defensive: exclude the currently active Pokémon
        available_switches = [
            mon for mon in available_switches if mon is not battle.active_pokemon
        ]
        for i in range(min(5, len(available_switches))):
            mask[i] = 1.0

        # Moves (6–9)
        available_moves = getattr(battle, "available_moves", []) or []
        for i in range(min(4, len(available_moves))):
            mask[6 + i] = 1.0

        return mask

    def get_additional_info(self) -> Dict[str, Dict[str, Any]]:
        info = super().get_additional_info()

        # Add any additional information you want to include in the info dictionary that is saved in logs
        # For example, you can add the win status

        if self.battle1 is not None:
            agent = self.possible_agents[0]
            info[agent]["win"] = self.battle1.won
        
            info[agent]["team_faint_count"] = sum(1 for mon in self.battle1.team.values() if mon.fainted)
            info[agent]["opponent_faint_count"] = sum(1 for mon in self.battle1.opponent_team.values() if mon.fainted)
            info[agent]["opponent_revealed_count"] = len(self.battle1.opponent_team)
            
            if self.battle1.active_pokemon:
                info[agent]["active_pokemon"] = self.battle1.active_pokemon.species
                info[agent]["active_pokemon_hp"] = self.battle1.active_pokemon.current_hp_fraction

            if self.battle1.opponent_active_pokemon:
                info[agent]["opponent_active_pokemon"] = self.battle1.opponent_active_pokemon.species
                info[agent]["opponent_active_hp"] = self.battle1.opponent_active_pokemon.current_hp_fraction

        return info

    def calc_reward(self, battle: AbstractBattle) -> float:
        """
        Calculates the reward based on the changes in state of the battle.

        You need to implement this method to define how the reward is calculated

        Args:
            battle (AbstractBattle): The current battle instance containing information
                about the player's team and the opponent's team from the player's perspective.
            prior_battle (AbstractBattle): The prior battle instance to compare against.
        Returns:
            float: The calculated reward based on the change in state of the battle.
        """

        prior_battle = self._get_prior_battle(battle)

        reward = 0.0

        health_team = [mon.current_hp_fraction for mon in battle.team.values()]
        health_opponent = [
            mon.current_hp_fraction for mon in battle.opponent_team.values()
        ]

        # If the opponent has less than 6 Pokémon, fill the missing values with 1.0 (fraction of health)
        if len(health_opponent) < len(health_team):
            health_opponent.extend([1.0] * (len(health_team) - len(health_opponent)))

        prior_health_opponent = []
        prior_health_team = []
        if prior_battle is not None:
            prior_health_opponent = [
                mon.current_hp_fraction for mon in prior_battle.opponent_team.values()
            ]
            prior_health_team = [mon.current_hp_fraction for mon in prior_battle.team.values()]
        #PLAY AROUND WITH MULTIPLIERS OF THE REWARD COMPONENTS HERE TO TUNE THE AGENT
        # Ensure health_opponent has 6 components, filling missing values with 1.0 (fraction of health)
        if len(prior_health_opponent) < len(health_team):
            prior_health_opponent.extend(
                [1.0] * (len(health_team) - len(prior_health_opponent))
            )

        diff_health_opponent = np.array(prior_health_opponent) - np.array(
            health_opponent
        )
        diff_health_team = np.array(prior_health_team) - np.array(health_team)

        reward += 3.0 * (np.sum(diff_health_opponent) - np.sum(diff_health_team))

        # KO bonuses for newly-fainted mons (detect where prev was >0 and curr==0) 
        
        faint_opp = np.sum((np.array(prior_health_opponent) > 0.001) & (np.array(health_opponent) <= 0.001))
        faint_own = np.sum((np.array(prior_health_team) > 0.001) & (np.array(health_team) <= 0.001))
        reward += 4.0 * float(faint_opp)     # reward for opponent faint
        reward -= 4.0 * float(faint_own)     # penalty for own faint
        # Team balance reward
        # Calculate prior_team_balance 
        prior_own_alive=[]
        prior_opp_alive=[]
        prior_team_balance = 0.0
        if prior_battle is not None:
            prior_own_alive = sum(1 for p in prior_battle.team.values() if not p.fainted)
            prior_opp_alive = sum(1 for p in prior_battle.opponent_team.values() if not p.fainted)
            max_team_prior = max(len(prior_battle.team), len(prior_battle.opponent_team), 1)
            prior_team_balance = (prior_own_alive - prior_opp_alive) / max_team_prior

        own_alive = sum(1 for p in battle.team.values() if not p.fainted)
        opp_alive = sum(1 for p in battle.opponent_team.values() if not p.fainted)
        max_team = max(len(battle.team), len(battle.opponent_team), 1)
        team_balance = (own_alive - opp_alive) / max_team

        reward += 2.0 * (team_balance - prior_team_balance)

        if battle.finished:
            if battle.won:
                reward += 15.0  
            else:
                reward -= 15.0 

        return reward

    def _observation_size(self) -> int:
        """
        Returns the size of the observation size to create the observation space for all possible agents in the environment.

        You need to set obvervation size to the number of features you want to include in the observation.
        Annoyingly, you need to set this manually based on the features you want to include in the observation from emded_battle.

        Returns:
            int: The size of the observation space.
        """

        # Updated calculation:
        return 213

    def embed_battle(self, battle: AbstractBattle) -> np.ndarray:
        """
        Embeds the current state of a Pokémon battle into a numerical vector representation.
        This method generates a feature vector that represents the current state of the battle,
        this is used by the agent to make decisions.

        You need to implement this method to define how the battle state is represented.

        Args:
            battle (AbstractBattle): The current battle instance containing information about
                the player's team and the opponent's team.
        Returns:
            np.float32: A 1D numpy array containing the state you want the agent to observe.
        """

        # Basic health info (12 components)
        health_team = [mon.current_hp_fraction for mon in battle.team.values()]
        health_opponent = [
            mon.current_hp_fraction for mon in battle.opponent_team.values()
        ]

        # Ensure health_opponent has 6 components, filling missing values with 1.0
        if len(health_opponent) < len(health_team):
            health_opponent.extend([1.0] * (len(health_team) - len(health_opponent)))

        # Active Pokemon info
        active_pokemon = battle.active_pokemon
        opp_active_pokemon = battle.opponent_active_pokemon
        
        # Speed estimation for priority decisions 
        my_speed = self._estimate_speed(active_pokemon) / 400.0 if active_pokemon else 0.0  # normalize
        opp_speed = self._estimate_speed(opp_active_pokemon) / 400.0 if opp_active_pokemon else 0.0
        speed_priority = 1.0 if my_speed > opp_speed else (0.0 if my_speed < opp_speed else 0.5) # one component for speed
        
        # Stat boosts -  (12 components: 6 stats x 2 pokemon)
        my_boosts = [
            active_pokemon.boosts.get(stat, 0) / 6.0 if active_pokemon else 0.0 
            for stat in ['atk', 'def', 'spa', 'spd', 'spe', 'accuracy']
        ]
        opp_boosts = [
            opp_active_pokemon.boosts.get(stat, 0) / 6.0 if opp_active_pokemon else 0.0 
            for stat in ['atk', 'def', 'spa', 'spd', 'spe', 'accuracy']
        ]
        
        
        # Battle hazards (8 entry hazards)
        battle_hazards = [
            1.0 if battle.side_conditions.get('stealthrock') else 0.0,
            battle.side_conditions.get('spikes', 0) / 3.0,  # 0-3 layers normalized
            1.0 if battle.side_conditions.get('toxicspikes') else 0.0,
            1.0 if battle.side_conditions.get('stickyweb') else 0.0,
    
            # Entry Hazards - Opponent Side (4 components)
            1.0 if battle.opponent_side_conditions.get('stealthrock') else 0.0,
            battle.opponent_side_conditions.get('spikes', 0) / 3.0,
            1.0 if battle.opponent_side_conditions.get('toxicspikes') else 0.0,
            1.0 if battle.opponent_side_conditions.get('stickyweb') else 0.0,
        ]

        weather_encoding = self._get_weather_one_hot(battle)  # 5 components

        # Battle state info (4 components)
        battle_state = [
            battle.turn / 100.0,  # normalize turn number
            len([p for p in battle.team.values() if not p.fainted]) / 6.0,  # team alive ratio
            (len([p for p in battle.opponent_team.values() if not p.fainted]) + (6 - len(battle.opponent_team))) / 6.0,  # opp alive ratio (assume unrevealed are alive)
            1.0 if battle.can_mega_evolve else 0.0,
        ]
        
        # Move effectiveness predictions (4 components) 
        move_effectiveness = self._get_move_effectiveness_for_active(battle)
        bp_vals = self._get_move_base_powers(battle)
        # Normalize base power (cap to 1.0) and weight effectiveness by normalized power
        bp_array = np.clip(np.array(bp_vals, dtype=np.float32) / 100.0, 0.0, 1.0) #Base power array (4 components)

        #Defense strategy - One-hot encode Pokemon types (18 + 18 components)
        my_types = self._get_type_one_hot(active_pokemon)     # 18 dims
        opp_types = self._get_type_one_hot(opp_active_pokemon) # 18 dims

        #Switch strategy - One-hot encode all team pokemon types in teams (not just active)
        my_bench_info = self._get_bench_info(battle)

        status_conditions = self._get_status_conditions(battle)
        # Final vector assembly
        final_vector = np.concatenate([
            health_team,              # 6 components
            health_opponent,          # 6 components  
            [speed_priority],         # 1 components
            my_boosts,                # 6 components
            opp_boosts,               # 6 components
            battle_hazards,           # 8 components
            weather_encoding,         # 5 components
            battle_state,             # 4 components
            move_effectiveness,       # 4 components
            bp_array,                 # 4 components
            my_types,                 # 18 components
            opp_types,                # 18 components
            my_bench_info,            # 115 components
            status_conditions         # 12 components
        ])
        #print(f"DEBUG: Health team size: {len(health_team)} expected 6\n Health opponent size: {len(health_opponent)} expected 6\n Speed priority size:{len([speed_priority])} expected 1\n My boosts size: {len(my_boosts)} expected 6\n Opp boosts size: {len(opp_boosts)} expected 6\n Battle hazards size: {len(battle_hazards)} expected 4\n Weather encoding size: {len(weather_encoding)} expected 5\n Battle state size: {len(battle_state)} expected 4\n Move effectiveness size: {len(move_effectiveness)} expected 4\n BP array size: {len(bp_array)} expected 4\n My types size: {len(my_types)} expected 18\n Opp types size: {len(opp_types)} expected 18\n My bench info size: {len(my_bench_info)} expected 115\n Status conditions size: {len(status_conditions)} expected 12\n Final vector size: {len(final_vector)} expected {len(final_vector)} expected 213")

        return final_vector.astype(np.float32)
    
    def _get_weather_one_hot(self, battle) -> list:
        """
        One-hot encode weather conditions.
        Returns: [no_weather, sun, rain, sandstorm, snow] (5 components)
        """
        weather = battle.weather
        # One-hot encoding for 5 weather states
        weather_encoding = [0.0, 0.0, 0.0, 0.0, 0.0]
    
        if weather is None:
            weather_encoding[0] = 1.0  # No weather
        elif weather == Weather.SUNNYDAY:
            weather_encoding[1] = 1.0  # Sun
        elif weather == Weather.RAINDANCE:
            weather_encoding[2] = 1.0  # Rain
        elif weather == Weather.SANDSTORM:
            weather_encoding[3] = 1.0  # Sandstorm
        elif weather == Weather.SNOW: 
            weather_encoding[4] = 1.0  # Snow/Hail
        else:
            weather_encoding[0] = 1.0  # Default to no weather if unknown
        return weather_encoding
    
    def _get_type_one_hot(self, pokemon: Pokemon) -> np.ndarray:
        """
        One-hot encode Pokemon types.
        18 types in Pokemon: Normal, Fire, Water, Electric, Grass, Ice, Fighting, 
        Poison, Ground, Flying, Psychic, Bug, Rock, Ghost, Dragon, Dark, Steel, Fairy
        
        Returns:
            np.ndarray: 18-component one-hot encoded array
        """
        type_names = [
            'normal', 'fire', 'water', 'electric', 'grass', 'ice',
            'fighting', 'poison', 'ground', 'flying', 'psychic',
            'bug', 'rock', 'ghost', 'dragon', 'dark', 'steel', 'fairy'
        ]
        
        type_encoding = np.zeros(18, dtype=np.float32)
    
        if pokemon and pokemon.types:
            for poke_type in pokemon.types:
                type_name = str(poke_type).lower()
                if type_name in type_names:
                    i = type_names.index(type_name)
                    type_encoding[i] = 1.0

        return type_encoding

    def _estimate_speed(self, pokemon):
        # Defensive: handle missing pokemon or missing base_stats
        if not pokemon:
            return 100.0

        base_stats = getattr(pokemon, "base_stats", None) or {}
        base_speed = base_stats.get('spe', 0)

        # Assume level 100 and common investment: max for base >= 100, else neutral
        if base_speed >= 100:
            # Max investment: 31 IV, 252 EV, beneficial nature (1.1x)
            raw_speed = (2 * base_speed + 31 + 63) + 5  # 63 from floor(252/4)
            estimated_speed = int(raw_speed * 1.1)
        else:
            # Neutral investment: 31 IV, 0 EV, neutral nature (1.0x)
            estimated_speed = (2 * base_speed + 31 + 0) + 5

        # Apply boosts/drops (from moves like Dragon Dance)
        boost = 0
        if hasattr(pokemon, 'boosts') and pokemon.boosts:
            try:
                boost = pokemon.boosts.get('spe', 0)
            except Exception:
                boost = 0

        multiplier = {
            -6: 0.25, -5: 2/7, -4: 2/6, -3: 0.4, -2: 0.5, -1: 2/3,
            0: 1.0, 1: 1.5, 2: 2.0, 3: 2.5, 4: 3.0, 5: 3.5, 6: 4.0
        }.get(boost, 1.0)
        estimated_speed *= multiplier

        return float(estimated_speed)
    
    def _get_move_effectiveness_for_active(self, battle):
        """Get effectiveness of moves for the active Pokémon.

        Returns:
            list[float]: effectiveness values for up to 4 moves, padded with 0.0
        """
        effectiveness_values = [0.0, 0.0, 0.0, 0.0]

        active = getattr(battle, "active_pokemon", None)
        defender = getattr(battle, "opponent_active_pokemon", None)
        effective_values = self._get_move_effectiveness(active, defender)

        return effective_values


    def _get_move_effectiveness(self, attacker: Pokemon, defender: Pokemon):
        """Get effectiveness of up to 4 available moves.

        Returns:
            list[float]: effectiveness values for up to 4 moves, padded with 0.0
        """
        effectiveness_values = [0.0, 0.0, 0.0, 0.0]
        
        moves = list(attacker.moves.values()) if attacker and hasattr(attacker, "moves") else []

        if defender is None or attacker is None:
            return effectiveness_values

        for i, move in enumerate(moves[:4]):  # limit to 4 moves
            try:
                eff = self._get_damage_multiplier(move, attacker, defender)
            except Exception:
                eff = 1.0
            effectiveness_values[i] = float(eff)
        return effectiveness_values
    
    
    def _get_bench_info(self, battle) -> np.ndarray:
        """
        Essential bench information for switching decisions.
        Returns 115 components:
        - bench_types: 90 (5 × 18 one-hot per Pokemon)
        - bench_can_switch: 5 (availability)
        - bench_move_effectiveness: 20 (5 × 4 best moves vs opponent)
        """
        team_pokemon = [p for p in battle.team.values() if p != battle.active_pokemon]
        
        # Bench types (90 dims)
        bench_types = []
        for i in range(5):
            if i < len(team_pokemon):
                pokemon_types = self._get_type_one_hot(team_pokemon[i])
            else:
                pokemon_types = np.zeros(18, dtype=np.float32)
            bench_types.append(pokemon_types)
    
        # Bench availability (5 dims)
        available_switches = battle.available_switches
        bench_availability = []
        for i in range(5):
            if i < len(team_pokemon):
                can_switch = 1.0 if team_pokemon[i] in available_switches else 0.0
            else:
                can_switch = 0.0
            bench_availability.append(can_switch)
    
        # Bench move effectiveness (20 dims)
        bench_moves = []
        for i in range(5):
            if i < len(team_pokemon) and battle.opponent_active_pokemon:
                pokemon = team_pokemon[i]
                move_effs = self._get_move_effectiveness(
                    pokemon, battle.opponent_active_pokemon
                )
            else:
                move_effs = [0.0, 0.0, 0.0, 0.0]
            bench_moves.extend(move_effs)
    
        # Concatenate all bench info
        return np.concatenate([
            np.concatenate(bench_types),           # 90
            np.array(bench_availability),           # 5
            np.array(bench_moves, dtype=np.float32) # 20
        ])
        # Total: 115 dimensions

    def _get_move_base_powers(self, battle):
        """Get base powers of up to 4 available moves.

        Returns:
            list[float]: base power values for up to 4 moves, padded with 0.0
        """
        base_power_values = [0.0, 0.0, 0.0, 0.0]

        # Defensive checks: battle may not have available_moves
        if not hasattr(battle, "available_moves") or not battle.available_moves:
            return base_power_values
        for i, move in enumerate(list(battle.available_moves)[:4]):  # limit to 4 moves
            try:
                bp = getattr(move, "base_power", 0) or 0
            except Exception:
                bp = 0
            base_power_values[i] = float(bp)

        return base_power_values
    
    def _get_status_conditions(self, battle) -> np.ndarray:
        """
        Get status condition information for active Pokemon on both sides.
        
        Returns 12 components:
        - Our active status (6): [none, burn, paralysis, poison, sleep, freeze]
        - Opponent active status (6): [none, burn, paralysis, poison, sleep, freeze]
        """
        our_status = self._encode_status_one_hot(battle.active_pokemon)
        opp_status = self._encode_status_one_hot(battle.opponent_active_pokemon)
        
        return np.concatenate([our_status, opp_status])
    

    def _encode_status_one_hot(self, pokemon: Pokemon) -> np.ndarray:
        """
        One-hot encode status condition.
        [none, burn, paralysis, poison/toxic, sleep, freeze]
        """
        status_encoding = np.zeros(6, dtype=np.float32)
        
        if not pokemon or not pokemon.status:
            status_encoding[0] = 1.0  # No status
        else:
            status_str = str(pokemon.status).lower()
            if 'brn' in status_str or 'burn' in status_str:
                status_encoding[1] = 1.0
            elif 'par' in status_str or 'paralysis' in status_str:
                status_encoding[2] = 1.0
            elif 'psn' in status_str or 'tox' in status_str or 'poison' in status_str:
                status_encoding[3] = 1.0
            elif 'slp' in status_str or 'sleep' in status_str:
                status_encoding[4] = 1.0
            elif 'frz' in status_str or 'freeze' in status_str:
                status_encoding[5] = 1.0
            else:
                status_encoding[0] = 1.0  # Unknown = treat as none
        
        return status_encoding
    
    def _get_damage_multiplier(self, move: Move, attacker: Pokemon, defender: Pokemon) -> float:
        """
        Estimates the damage a move would deal.
        """
        # Defensive: ensure move, attacker, defender exist and have expected attributes
        if not move:
            return 0.0

        base_power = getattr(move, 'base_power', None)
        if base_power in (None, 0):
            return 0.0

        multiplier = 1.0
        try:
            if defender is not None and not getattr(defender, 'fainted', False):
                if hasattr(defender, 'damage_multiplier'):
                    multiplier = defender.damage_multiplier(move)
        except Exception:
            multiplier = 1.0

        # STAB check: attacker may be None or missing types
        try:
            attacker_types = getattr(attacker, 'types', []) or []
            if getattr(move, 'type', None) in attacker_types:
                multiplier *= 1.5  # STAB
        except Exception:
            pass

        return float(multiplier)


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