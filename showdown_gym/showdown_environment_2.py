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
from poke_env.battle import AbstractBattle, Move, Pokemon
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
        # Reduced action space: 4 moves + 6 switches = 10 actions (removing complex mechanics)
        # return 10  # Uncomment this line to use reduced action space
        return None  # Return None if action size is default

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

        :param action: The action to take.
        :type action: int64

        :return: The battle order ID for the given action in context of the current battle.
        :rtype: np.Int64
        """
        return action

    def get_additional_info(self) -> Dict[str, Dict[str, Any]]:
        info = super().get_additional_info()

        # Add any additional information you want to include in the info dictionary that is saved in logs
        # For example, you can add the win status

        if self.battle1 is not None:
            agent = self.possible_agents[0]
            info[agent]["win"] = self.battle1.won
            
            # Add strategic metrics for analysis
            active_pokemon = self.battle1.active_pokemon
            if active_pokemon:
                info[agent]["active_hp"] = active_pokemon.current_hp_fraction
                info[agent]["team_alive"] = len([p for p in self.battle1.team.values() if not p.fainted])
                # If a Pokémon is not known (not revealed), we assume it's still alive (not fainted)
                info[agent]["opp_alive"] = len([
                    p for p in self.battle1.opponent_team.values() if not p.fainted
                ]) + (6 - len(self.battle1.opponent_team))
                info[agent]["turn"] = self.battle1.turn
                
                # Track setup success (boosts gained)
                total_boosts = sum(active_pokemon.boosts.values()) if active_pokemon.boosts else 0
                info[agent]["total_boosts"] = total_boosts
                
                # Track hazards
                # Track hazards (Stealth Rock, Spikes, Toxic Spikes, Sticky Web)
                info[agent]["hazards_set"] = {
                    "stealthrock": 1 if self.battle1.opponent_side_conditions.get('stealthrock') else 0,
                    "spikes": self.battle1.opponent_side_conditions.get('spikes', 0),
                    "toxicspikes": self.battle1.opponent_side_conditions.get('toxicspikes', 0),
                    "stickyweb": 1 if self.battle1.opponent_side_conditions.get('stickyweb') else 0,
                }

        return info

    def calc_reward(self, battle: AbstractBattle) -> float:
        """
        Calculates the reward based on the changes in state of the battle.

        - All rewards reduced to [-3.0, +3.0] range

        Args:
            battle (AbstractBattle): The current battle instance containing information
                about the player's team and the opponent's team from the player's perspective.
            prior_battle (AbstractBattle): The prior battle instance to compare against.
        Returns:
            float: The calculated reward based on the change in state of the battle.
        """

        prior_battle = self._get_prior_battle(battle)
        reward = 0.0

        # Basic health-based rewards (from original)
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

            # 1. Damage rewards
            opp_damage = np.sum(np.array(prior_health_opponent) - np.array(health_opponent))
            my_damage = np.sum(np.array(prior_health_team) - np.array(health_team))
            reward += opp_damage * 0.4 - my_damage * 0.3  

            # 2. KO rewards 
            prior_opp_fainted = len([p for p in prior_battle.opponent_team.values() if p.fainted])
            current_opp_fainted = len([p for p in battle.opponent_team.values() if p.fainted])
            ko_bonus = (current_opp_fainted - prior_opp_fainted) * 1.5  # Reward for each new KO
            reward += ko_bonus

            # 3. Survival penalty 
            prior_my_alive = len([p for p in prior_battle.team.values() if not p.fainted])
            current_my_alive = len([p for p in battle.team.values() if not p.fainted])
            survival_penalty = (prior_my_alive - current_my_alive) * -1.2  
            reward += survival_penalty

            # 4. Strategic rewards
            # Reward for setting hazards (Stealth Rock, Spikes, Toxic Spikes, Sticky Web)
            hazard_types = ['stealthrock', 'spikes', 'toxicspikes', 'stickyweb']
            for hazard in hazard_types:
                current = battle.opponent_side_conditions.get(hazard, 0)
                prior = prior_battle.opponent_side_conditions.get(hazard, 0)
                # For hazards that are boolean (stealthrock, stickyweb), treat as 1 if present
                if hazard in ['stealthrock', 'stickyweb']:
                    if current and not prior:
                        reward += 0.5
                else:
                    # For stackable hazards (spikes, toxicspikes), reward for each new layer
                    if current > prior:
                        reward += 0.3 * (current - prior)
            
            # Boost rewards
            active_pokemon = battle.active_pokemon
            prior_active = prior_battle.active_pokemon
            if active_pokemon and prior_active and active_pokemon.species == prior_active.species:
                boost_improvement = 0.0
                for stat in ['atk', 'def', 'spa', 'spd', 'spe']:
                    current_boost = active_pokemon.boosts.get(stat, 0)
                    prior_boost = prior_active.boosts.get(stat, 0)
                    boost_improvement += (current_boost - prior_boost)
                reward += boost_improvement * 0.3 


        # 5. Turn efficiency
        turn_penalty = -0.01 * battle.turn if battle.turn > 30 else 0.0
        reward += turn_penalty

        # 6. Final battle outcome rewards
        if battle.finished:
            if battle.won:
                reward += 2.0 
            else:
                reward -= 1.5 
        # Clamp reward to safe range for RL training
        reward = np.clip(reward, -3.0, 3.0)

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
        # health_team: 6 + health_opponent: 6 + speed: 2 + boosts: 12 + types: 4 + conditions: 6 + moves: 4 = 40
        return 40

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

        # Active Pokemon info (8 components)
        active_pokemon = battle.active_pokemon
        opp_active_pokemon = battle.opponent_active_pokemon
        
        # Speed estimation for priority decisions (2 components)
        my_speed = self._estimate_speed(active_pokemon) / 400.0 if active_pokemon else 0.0  # normalize
        opp_speed = self._estimate_speed(opp_active_pokemon) / 400.0 if opp_active_pokemon else 0.0
        
        # Stat boosts - critical for setup sweepers (12 components: 6 stats x 2 pokemon)
        my_boosts = [
            active_pokemon.boosts.get(stat, 0) / 6.0 if active_pokemon else 0.0 
            for stat in ['atk', 'def', 'spa', 'spd', 'spe', 'accuracy']
        ]
        opp_boosts = [
            opp_active_pokemon.boosts.get(stat, 0) / 6.0 if opp_active_pokemon else 0.0 
            for stat in ['atk', 'def', 'spa', 'spd', 'spe', 'accuracy']
        ]
        
        # Type effectiveness for current matchup (4 components)
        my_types, opp_types = self._get_type_vectors(active_pokemon, opp_active_pokemon)
        
        # Battle conditions (6 components)
        battle_conditions = [
            1.0 if battle.opponent_side_conditions.get('stealthrock') else 0.0,
            1.0 if battle.side_conditions.get('stealthrock') else 0.0,
            battle.turn / 100.0,  # normalize turn number
            len([p for p in battle.team.values() if not p.fainted]) / 6.0,  # team alive ratio
            len([p for p in battle.opponent_team.values() if not p.fainted]) + (6 - len(battle.opponent_team)) / 6.0,  # opp alive ratio (assume unrevealed are alive)
            1.0 if battle.can_mega_evolve else 0.0,
        ]
        
        # Move effectiveness predictions (4 components) 
        move_effectiveness = self._get_move_effectiveness(battle)

        # Final vector assembly
        final_vector = np.concatenate([
            health_team,              # 6 components
            health_opponent,          # 6 components  
            [my_speed, opp_speed],    # 2 components
            my_boosts,                # 6 components
            opp_boosts,               # 6 components
            my_types,                 # 2 components
            opp_types,                # 2 components
            battle_conditions,        # 6 components
            move_effectiveness,       # 4 components
        ])

        return final_vector.astype(np.float32)
    
    def _estimate_speed(self, pokemon):
        base_speed = pokemon.base_stats.get('spe', 0)
        # Assume level 100 and common investment: max for base >= 100, else neutral
        if base_speed >= 100:
            # Max investment: 31 IV, 252 EV, beneficial nature (1.1x)
            raw_speed = (2 * base_speed + 31 + 63) + 5  # 63 from floor(252/4)
            estimated_speed = int(raw_speed * 1.1)
        else:
            # Neutral investment: 31 IV, 0 EV, neutral nature (1.0x)
            estimated_speed = (2 * base_speed + 31 + 0) + 5

        # Apply boosts/drops (from moves like Dragon Dance)
        boost = pokemon.boosts.get('spe', 0) if hasattr(pokemon, 'boosts') else 0
        multiplier = {
            -6: 0.25, -5: 2/7, -4: 2/6, -3: 0.4, -2: 0.5, -1: 2/3,
            0: 1.0, 1: 1.5, 2: 2.0, 3: 2.5, 4: 3.0, 5: 3.5, 6: 4.0
        }.get(boost, 1.0)
        estimated_speed *= multiplier

        # Account for common items and abilities if possible
        if hasattr(pokemon, 'item'):
            if pokemon.item == 'choicescarf':
                estimated_speed *= 1.5
        if hasattr(pokemon, 'ability'): # TODO: check for speed abilities
            if pokemon.ability == 'speedboost':
                estimated_speed *= 1.5
        return float(estimated_speed)



    #def _estimate_speed(self, pokemon) -> float:
        
        # if not pokemon:
        #     return 100.0  # Return a default speed if pokemon is None
        #
        # # Base speed lookup for common Pokemon (simplified for Ubers/OU)
        # base_speeds = {
        #     'koraidon': 135, 'miraidon': 135, 'zaciancrowned': 148, 'zamazentacrowned': 128,
        #     'kyogre': 90, 'groudon': 90, 'necrozmaduskmane': 77, 'necrozmadawnwings': 77,
        #     'arceus': 120, 'arceusground': 120, 'arceuswater': 120, 'hooh': 90,
        #     'lugia': 110, 'rayquaza': 95, 'dialga': 90, 'palkia': 100, 'giratina': 90,
        #     'reshiram': 90, 'zekrom': 90, 'kyurem': 95, 'yveltal': 99, 'xerneas': 99,
        #     'lunala': 97, 'solgaleo': 97, 'eternatus': 130
        # }
        #
        # # Get the base speed for the species, default to 100 if not found
        # base_speed = base_speeds.get(pokemon.species, 100)
        #
        # # Get the current speed boost (stat stage), default to 0 if not present
        # speed_boost = pokemon.boosts.get('spe', 0) if hasattr(pokemon, 'boosts') and pokemon.boosts else 0
        #
        # # Calculate the boost multiplier (simplified: each stage = +0.5x, min 0.25x)
        # boost_multiplier = max(0.25, 1.0 + speed_boost * 0.5)
        #
        # # Return the estimated speed after applying boosts
        # return base_speed * boost_multiplier

    def _get_type_vectors(self, my_pokemon: Pokemon, opp_pokemon: Pokemon):
        """Get simplified type representation vectors."""
        # Simplified type encoding - just primary type effectiveness
        type_chart = {
            'normal': 0, 'fire': 1, 'water': 2, 'electric': 3, 'grass': 4, 'ice': 5,
            'fighting': 6, 'poison': 7, 'ground': 8, 'flying': 9, 'psychic': 10,
            'bug': 11, 'rock': 12, 'ghost': 13, 'dragon': 14, 'dark': 15, 'steel': 16, 'fairy': 17
        }
        
        my_types = [0.0, 0.0]  # Pokemon may have two types
        opp_types = [0.0, 0.0]
        
        if my_pokemon and my_pokemon.types:
            my_types[0] = type_chart.get(my_pokemon.types[0].name, 0) / 17.0
            if len(my_pokemon.types) > 1:
                my_types[1] = type_chart.get(my_pokemon.types[1].name, 0) / 17.0
                
        if opp_pokemon and opp_pokemon.types:
            opp_types[0] = type_chart.get(opp_pokemon.types[0].name, 0) / 17.0
            if len(opp_pokemon.types) > 1:
                opp_types[1] = type_chart.get(opp_pokemon.types[1].name, 0) / 17.0
                
        return my_types, opp_types

    def _get_move_effectiveness(self, battle):
        """Get effectiveness of available moves against opponent."""
        effectiveness_values = [0.0, 0.0, 0.0, 0.0]  # top 4 moves effectiveness
        
        if not battle.available_moves or not battle.opponent_active_pokemon:
            return effectiveness_values
            
        move_effs = []
        for move in battle.available_moves[:4]:  # limit to 4 moves
            eff = self._estimate_damage(move, battle.active_pokemon, battle.opponent_active_pokemon)
            move_effs.append(eff)
                
        # Pad to 4 values
        while len(move_effs) < 4:
            move_effs.append(0.0)
            
        return move_effs[:4]
    
    def _estimate_damage(self, move: Move, attacker: Pokemon, defender: Pokemon) -> float:
        """
        Estimates the damage a move would deal.
        """
        if move.base_power == 0:
            return 0
        
        if defender is None or defender.fainted:
            multiplier = 1.0
        else:
            multiplier = defender.damage_multiplier(move)
        
        if move.type in attacker.types:
            multiplier *= 1.5 # STAB
            
        #damage = move.base_power * multiplier
        return multiplier



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