#!/usr/bin/env python3
"""
Gradio Configuration Generator

This module creates configurations compatible with the existing runner.py and
simulate.py infrastructure, eliminating code duplication in the Gradio app.
"""

import tempfile
import yaml
from typing import Dict, Any, Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)


def _legal_actions_with_labels(env, pid: int) -> List[Tuple[int, str]]:
    """Return the current player's legal actions as (id, label) pairs."""
    try:
        actions = env.state.legal_actions(pid)
    except Exception:
        return []
    labelled = []
    for a in actions:
        label = None
        if hasattr(env, "get_action_display"):
            try:
                label = env.get_action_display(a, pid)
            except Exception:
                label = None
        elif hasattr(env.state, "action_to_string"):
            try:
                label = env.state.action_to_string(pid, a)
            except Exception:
                label = None
        labelled.append((a, label or str(a)))
    return labelled


def start_game_interactive(
    game_name: str,
    player1_type: str,
    player2_type: str,
    player1_model: Optional[str],
    player2_model: Optional[str],
    rounds: int,
    seed: int,
) -> Tuple[str, Dict[str, Any], List[Tuple[int, str]], List[Tuple[int, str]]]:
    """Initialize env + policies; return (log, state, legal_p0, legal_p1)."""
    from src.game_reasoning_arena.arena.utils.seeding import set_seed
    from src.game_reasoning_arena.backends import initialize_llm_registry
    from src.game_reasoning_arena.arena.games.registry import registry
    from src.game_reasoning_arena.arena.agents.policy_manager import (
        initialize_policies,
    )

    cfg = create_config_for_gradio_game(
        game_name=game_name,
        player1_type=player1_type,
        player2_type=player2_type,
        player1_model=player1_model,
        player2_model=player2_model,
        rounds=1,
        seed=seed,
    )

    set_seed(seed)
    try:
        initialize_llm_registry()
    except Exception:
        # ok if LLM backend not available for random/human vs random/human
        pass

    # Build agents + env using your existing infra
    policies = initialize_policies(cfg, game_name, seed)
    env = registry.make_env(game_name, cfg)
    obs, _ = env.reset(seed=seed)

    # Map policy order to player ids (same as your simulate.py)
    player_to_agent: Dict[int, Any] = {}
    for i, policy_name in enumerate(policies.keys()):
        player_to_agent[i] = policies[policy_name]

    log = []
    log.append("🎮 INTERACTIVE GAME")
    log.append("=" * 50)
    log.append(f"Game: {game_name.replace('_', ' ').title()}")
    log.append("")

    # Choose which agent_id's board to show:
    # - If P0 is human -> agent_id=0; elif P1 is human -> agent_id=1; else 0.
    show_id = 0 if player1_type == "human" else (1 if player2_type == "human" else 0)
    try:
        board = env.render_board(show_id)
        log.append("Initial board:")
        log.append(board)
    except NotImplementedError:
        log.append("Board rendering not implemented for this game.")
    except Exception as e:
        log.append(f"Board not available: {e}")

    state = {
        "env": env,
        "obs": obs,
        "terminated": False,
        "truncated": False,
        "rewards": {0: 0, 1: 0},
        "players": {
            0: {"type": player1_type},
            1: {"type": player2_type},
        },
        "agents": player_to_agent,
        "show_id": show_id,
    }

    # Auto-advance the game until it's a human player's turn
    def _is_human(pid: int) -> bool:
        return ((pid == 0 and player1_type == "human") or
                (pid == 1 and player2_type == "human"))

    def _any_human_needs_action() -> bool:
        """Check if any human player needs to make an action."""
        try:
            if env.state.is_simultaneous_node():
                return _is_human(0) or _is_human(1)
            else:
                cur = env.state.current_player()
                return _is_human(cur)
        except Exception:
            return False

    # Process AI moves until a human needs to act or game ends
    term = False
    trunc = False
    while not (term or trunc) and not _any_human_needs_action():
        # Build actions for current turn
        if env.state.is_simultaneous_node():
            actions = {}
            # P0
            if not _is_human(0):
                response = player_to_agent[0](obs[0])
                a0, _ = _extract_action_and_reasoning(response)
                actions[0] = a0
            # P1
            if not _is_human(1):
                response = player_to_agent[1](obs[1])
                a1, _ = _extract_action_and_reasoning(response)
                actions[1] = a1
            log.append(f"Auto-play: P0={actions.get(0, 'waiting')}, "
                       f"P1={actions.get(1, 'waiting')}")
        else:
            # Sequential game
            cur = env.state.current_player()
            if not _is_human(cur):
                response = player_to_agent[cur](obs[cur])
                a, reasoning = _extract_action_and_reasoning(response)
                actions = {cur: a}
                log.append(f"Player {cur} (auto) chooses {a}")
                if reasoning and reasoning != "None":
                    prev = reasoning[:100]
                    if len(reasoning) > 100:
                        prev += "..."
                    log.append(f" Reasoning: {prev}")
            else:
                # Human's turn - break out of loop
                break

        # Step env
        obs, step_rewards, term, trunc, _ = env.step(actions)
        for pid, r in step_rewards.items():
            state["rewards"][pid] += r

        # Update board display
        try:
            log.append("Board:")
            log.append(env.render_board(show_id))
        except NotImplementedError:
            log.append("Board rendering not implemented for this game.")
        except Exception as e:
            log.append(f"Board not available: {e}")

    # Update state with current observations
    state["obs"] = obs
    state["terminated"] = term
    state["truncated"] = trunc

    # Prepare human choices for current state
    legal_p0: List[Tuple[int, str]] = []
    legal_p1: List[Tuple[int, str]] = []

    if not (term or trunc):
        try:
            if env.state.is_simultaneous_node():
                if player1_type == "human":
                    legal_p0 = _legal_actions_with_labels(env, 0)
                if player2_type == "human":
                    legal_p1 = _legal_actions_with_labels(env, 1)
            else:
                cur = env.state.current_player()
                if cur == 0 and player1_type == "human":
                    legal_p0 = _legal_actions_with_labels(env, 0)
                if cur == 1 and player2_type == "human":
                    legal_p1 = _legal_actions_with_labels(env, 1)
        except Exception:
            pass

    return "\n".join(log), state, legal_p0, legal_p1


def submit_human_move(
    action_p0: Optional[int],
    action_p1: Optional[int],
    state: Dict[str, Any],
) -> Tuple[str, Dict[str, Any], List[Tuple[int, str]], List[Tuple[int, str]]]:
    """
    Process human move and continue advancing the game automatically until:
    - It's a human player's turn again, OR
    - The game ends

    Returns (log_append, state, next_legal_p0, next_legal_p1)
    """
    if not state:
        return "No game is running.", state, [], []

    env = state["env"]
    obs = state["obs"]
    term = state["terminated"]
    trunc = state["truncated"]
    rewards = state["rewards"]
    ptypes = state["players"]
    agents = state["agents"]
    show_id = state["show_id"]

    if term or trunc:
        return "Game already finished.", state, [], []

    def _is_human(pid: int) -> bool:
        return ptypes[pid]["type"] == "human"

    def _any_human_needs_action() -> bool:
        """Check if any human player needs to make an action."""
        try:
            if env.state.is_simultaneous_node():
                return _is_human(0) or _is_human(1)
            else:
                cur = env.state.current_player()
                return _is_human(cur)
        except Exception:
            return False

    log = []

    # Continue processing moves until a human needs to act or game ends
    while not (term or trunc):
        # Build actions for current turn
        if env.state.is_simultaneous_node():
            actions = {}
            # P0
            if _is_human(0):
                if action_p0 is None:
                    return ("Pick an action for Player 0.", state,
                            _legal_actions_with_labels(env, 0), [])
                actions[0] = action_p0
                action_p0 = None  # Only use human action once
            else:
                a0, _ = _extract_action_and_reasoning(agents[0](obs[0]))
                actions[0] = a0
            # P1
            if _is_human(1):
                if action_p1 is None:
                    return ("Pick an action for Player 1.", state,
                            [], _legal_actions_with_labels(env, 1))
                actions[1] = action_p1
                action_p1 = None  # Only use human action once
            else:
                a1, _ = _extract_action_and_reasoning(agents[1](obs[1]))
                actions[1] = a1
            log.append(f"Actions: P0={actions[0]}, P1={actions[1]}")
        else:
            # Sequential game
            cur = env.state.current_player()
            if _is_human(cur):
                chosen = action_p0 if cur == 0 else action_p1
                if chosen is None:
                    choices = _legal_actions_with_labels(env, cur)
                    return ("Pick an action first.", state,
                            choices if cur == 0 else [],
                            choices if cur == 1 else [])
                actions = {cur: chosen}
                log.append(f"Player {cur} (human) chooses {chosen}")
                # Clear the action so it's not reused
                if cur == 0:
                    action_p0 = None
                else:
                    action_p1 = None
            else:
                a, reasoning = _extract_action_and_reasoning(agents[cur](obs[cur]))
                actions = {cur: a}
                log.append(f"Player {cur} (agent) chooses {a}")
                if reasoning and reasoning != "None":
                    prev = reasoning[:100] + ("..." if len(reasoning) > 100 else "")
                    log.append(f" Reasoning: {prev}")

        # Step env
        obs, step_rewards, term, trunc, _ = env.step(actions)
        for pid, r in step_rewards.items():
            rewards[pid] += r

        # Board
        try:
            log.append("Board:")
            log.append(env.render_board(show_id))
        except NotImplementedError:
            log.append("Board rendering not implemented for this game.")
        except Exception as e:
            log.append(f"Board not available: {e}")

        # Check if game ended
        if term or trunc:
            break

        # Check if we should continue automatically (AI turn) or stop (human turn)
        if _any_human_needs_action():
            break  # Stop here, human needs to act

        # If we reach here, it's an AI's turn - continue the loop

    # Game ended or waiting for human input
    if term or trunc:
        if rewards[0] > rewards[1]:
            winner = "Player 0"
        elif rewards[1] > rewards[0]:
            winner = "Player 1"
        else:
            winner = "Draw"
        log.append(f"Winner: {winner}")
        log.append(f"Scores: P0={rewards[0]}, P1={rewards[1]}")
        state["terminated"] = term
        state["truncated"] = trunc
        state["obs"] = obs
        return "\n".join(log), state, [], []

    # Determine next human choices
    next_p0, next_p1 = [], []
    try:
        if env.state.is_simultaneous_node():
            if _is_human(0):
                next_p0 = _legal_actions_with_labels(env, 0)
            if _is_human(1):
                next_p1 = _legal_actions_with_labels(env, 1)
        else:
            cur = env.state.current_player()
            if _is_human(cur):
                choices = _legal_actions_with_labels(env, cur)
                if cur == 0:
                    next_p0 = choices
                else:
                    next_p1 = choices
    except Exception:
        pass

    state["obs"] = obs
    return "\n".join(log), state, next_p0, next_p1


def create_config_for_gradio_game(
    game_name: str,
    player1_type: str,
    player2_type: str,
    player1_model: str = None,
    player2_model: str = None,
    rounds: int = 1,
    seed: int = 42,
    use_ray: bool = False
) -> Dict[str, Any]:
    """
    Create a configuration dictionary compatible with the existing
    runner.py and simulate.py infrastructure.

    Args:
        game_name: Name of the game to play
        player1_type: Type of player 1 (human, random, llm)
        player2_type: Type of player 2 (human, random, llm)
        player1_model: LLM model for player 1 (if applicable)
        player2_model: LLM model for player 2 (if applicable)
        rounds: Number of episodes to play
        seed: Random seed for reproducibility
        use_ray: Whether to use Ray for parallel processing

    Returns:
        Configuration dictionary compatible with runner.py
    """

    # Base configuration structure (matches default_simulation_config)
    config = {
        "env_config": {
            "game_name": game_name,
            "max_game_rounds": None,
        },
        "num_episodes": rounds,
        "seed": seed,
        "use_ray": use_ray,
        "mode": f"{player1_type}_vs_{player2_type}",
        "agents": {},
        "llm_backend": {
            "max_tokens": 250,
            "temperature": 0.1,
            "default_model": "litellm_groq/gemma-7b-it",
        },
        "log_level": "INFO",
    }

    # Configure player agents
    config["agents"]["player_0"] = _create_agent_config(
        player1_type, player1_model)
    config["agents"]["player_1"] = _create_agent_config(
        player2_type, player2_model)

    # Debug: Print the agent configurations
    print("📋 CONFIG DEBUG: Agent configurations created:")
    print(f"   Player 0 config: {config['agents']['player_0']}")
    print(f"   Player 1 config: {config['agents']['player_1']}")

    # Update backend default model if LLM is used
    # Check player 1 first
    if (player1_type == "llm" and player1_model) or player1_type.startswith("llm_"):
        if player1_model:
            config["llm_backend"]["default_model"] = player1_model
        elif player1_type.startswith("llm_"):
            # Extract model from player type (e.g., "llm_gpt2" -> "gpt2")
            config["llm_backend"]["default_model"] = player1_type[4:]
    # Check player 2 if player 1 doesn't have LLM
    elif (player2_type == "llm" and player2_model) or player2_type.startswith("llm_"):
        if player2_model:
            config["llm_backend"]["default_model"] = player2_model
        elif player2_type.startswith("llm_"):
            # Extract model from player type (e.g., "llm_gpt2" -> "gpt2")
            config["llm_backend"]["default_model"] = player2_type[4:]

    return config


def _create_agent_config(player_type: str,
                         model: str = None) -> Dict[str, Any]:
    """
    Create agent configuration based on player type and model.

    Handles both Gradio-specific formats (e.g., "hf_gpt2", "random_bot")
    and standard formats (e.g., "llm", "random", "human").

    Args:
        player_type: Type of player (human, random, random_bot, hf_*, etc.)
        model: Model name for LLM agents

    Returns:
        Agent configuration dictionary
    """
    print("🔧 AGENT CONFIG DEBUG: Creating agent config for:")
    print(f"   player_type: {player_type}")
    print(f"   model: {model}")

    # Handle Gradio-specific formats
    if player_type == "random_bot":
        config = {"type": "random"}
    elif player_type == "human":
        config = {"type": "human"}
    elif player_type.startswith("hf_"):
        # Extract model from player type (e.g., "hf_gpt2" -> "gpt2")
        model_from_type = player_type[3:]  # Remove "hf_" prefix

        # Use the hf_prefixed model name for LLM registry lookup
        model_name = f"hf_{model_from_type}"

        config = {
            "type": "llm",  # Use standard LLM agent type
            "model": model_name  # This will be looked up in LLM_REGISTRY
        }
    elif player_type.startswith("llm_"):
        # For backwards compatibility with LiteLLM models
        model_from_type = player_type[4:]  # Remove "llm_" prefix

        # Map display model names to actual model names with prefixes
        model_name = model or model_from_type
        if not model_name.startswith(("litellm_", "vllm_")):
            # Add litellm_ prefix for LiteLLM models
            model_name = f"litellm_{model_name}"

        config = {
            "type": "llm",
            "model": model_name
        }
    elif player_type == "llm":
        model_name = model or "litellm_groq/gemma-7b-it"
        if not model_name.startswith(("litellm_", "vllm_")):
            model_name = f"litellm_{model_name}"
        config = {
            "type": "llm",
            "model": model_name
        }
    elif player_type == "random":
        config = {"type": "random"}
    else:
        # Default to random for unknown types
        config = {"type": "random"}

    print(f"   → Created config: {config}")
    return config


def create_temporary_config_file(config: Dict[str, Any]) -> str:
    """
    Create a temporary YAML config file that can be used with runner.py.

    Args:
        config: Configuration dictionary

    Returns:
        Path to the temporary config file
    """
    # Create temporary file
    temp_file = tempfile.NamedTemporaryFile(
        mode='w',
        suffix='.yaml',
        delete=False
    )

    try:
        yaml.dump(config, temp_file, default_flow_style=False)
        temp_file.flush()
        return temp_file.name
    finally:
        temp_file.close()


def run_game_with_existing_infrastructure(
    game_name: str,
    player1_type: str,
    player2_type: str,
    player1_model: str = None,
    player2_model: str = None,
    rounds: int = 1,
    seed: int = 42
) -> str:
    """
    Run a game using the existing runner.py and simulate.py infrastructure,
    but capture detailed game logs for Gradio display.

    This function reuses the existing simulation infrastructure while providing
    detailed game output for the Gradio interface.

    Args:
        game_name: Name of the game to play
        player1_type: Type of player 1
        player2_type: Type of player 2
        player1_model: LLM model for player 1 (if applicable)
        player2_model: LLM model for player 2 (if applicable)
        rounds: Number of episodes to play
        seed: Random seed

    Returns:
        Detailed game simulation results as a string
    """
    try:
        # Import the existing infrastructure
        from src.game_reasoning_arena.arena.utils.seeding import set_seed
        from src.game_reasoning_arena.backends import initialize_llm_registry
        from src.game_reasoning_arena.arena.games.registry import registry
        from src.game_reasoning_arena.arena.agents.policy_manager import (
            initialize_policies, policy_mapping_fn
        )

        # Create configuration
        config = create_config_for_gradio_game(
            game_name=game_name,
            player1_type=player1_type,
            player2_type=player2_type,
            player1_model=player1_model,
            player2_model=player2_model,
            rounds=rounds,
            seed=seed
        )

        # Set seed
        set_seed(seed)

        # Initialize LLM registry (required for simulate_game)
        initialize_llm_registry()

        # Use existing infrastructure but capture detailed logs
        return _run_game_with_detailed_logging(game_name, config, seed)

    except ImportError as e:
        logger.error(f"Failed to import simulation infrastructure: {e}")
        return f"Error: Simulation infrastructure not available. {e}"
    except Exception as e:
        logger.error(f"Game simulation failed: {e}")
        return f"Error during game simulation: {e}"


def _run_game_with_detailed_logging(
    game_name: str,
    config: Dict[str, Any],
    seed: int
) -> str:
    """
    Run game simulation with detailed logging for Gradio display.

    This reuses the existing infrastructure components but captures
    detailed game state information for user display.
    """
    from src.game_reasoning_arena.arena.games.registry import registry
    from src.game_reasoning_arena.arena.agents.policy_manager import (
        initialize_policies, policy_mapping_fn
    )

    # Initialize using existing infrastructure
    policies_dict = initialize_policies(config, game_name, seed)
    env = registry.make_env(game_name, config)

    # Create player mapping (reusing existing logic)
    player_to_agent = {}
    for i, policy_name in enumerate(policies_dict.keys()):
        player_to_agent[i] = policies_dict[policy_name]

    game_log = []

    # Add header
    game_log.append("🎮 GAME SIMULATION RESULTS")
    game_log.append("=" * 50)
    game_log.append(f"Game: {game_name.replace('_', ' ').title()}")
    game_log.append(f"Episodes: {config['num_episodes']}")
    game_log.append("")

    # Player information
    game_log.append("👥 PLAYERS:")
    player1 = config["agents"]["player_0"]
    player2 = config["agents"]["player_1"]
    game_log.append(f"  Player 0: {_format_player_info(player1)}")
    game_log.append(f"  Player 1: {_format_player_info(player2)}")
    game_log.append("")

    # Run episodes (reusing compute_actions logic from simulate.py)
    for episode in range(config["num_episodes"]):
        episode_seed = seed + episode
        game_log.append(f"🎯 Episode {episode + 1}")
        game_log.append("-" * 30)

        observation_dict, _ = env.reset(seed=episode_seed)
        terminated = truncated = False
        step_count = 0
        episode_rewards = {0: 0, 1: 0}

        while not (terminated or truncated):
            step_count += 1
            game_log.append(f"\n📋 Step {step_count}")

            # Show board state
            try:
                board = env.render_board(0)
                game_log.append("Current board:")
                game_log.append(board)
            except:
                game_log.append("Board state not available")

            # Use the existing compute_actions logic from simulate.py
            try:
                action_dict = _compute_actions_for_gradio(
                    env, player_to_agent, observation_dict, game_log
                )
            except Exception as e:
                game_log.append(f"❌ Error computing actions: {e}")
                truncated = True
                break

            # Step forward (reusing existing environment logic)
            if not truncated:
                observation_dict, rewards, terminated, truncated, _ = env.step(action_dict)
                for player_id, reward in rewards.items():
                    episode_rewards[player_id] += reward

        # Episode results
        game_log.append(f"\n🏁 Episode {episode + 1} Complete!")
        try:
            game_log.append("Final board:")
            game_log.append(env.render_board(0))
        except:
            game_log.append("Final board state not available")

        if episode_rewards[0] > episode_rewards[1]:
            winner = "Player 0"
        elif episode_rewards[1] > episode_rewards[0]:
            winner = "Player 1"
        else:
            winner = "Draw"

        game_log.append(f"🏆 Winner: {winner}")
        game_log.append(f"📊 Scores: Player 0={episode_rewards[0]}, Player 1={episode_rewards[1]}")
        game_log.append("")

    game_log.append("✅ Simulation completed successfully!")
    game_log.append("Check the database logs for detailed move analysis.")

    return "\n".join(game_log)


def _compute_actions_for_gradio(env, player_to_agent, observations, game_log):
    """
    Compute actions and log details for Gradio display.
    This reuses the compute_actions logic from simulate.py.
    """
    if env.state.is_simultaneous_node():
        # Simultaneous-move game
        actions = {}
        for player in player_to_agent:
            agent_response = player_to_agent[player](observations[player])
            action, reasoning = _extract_action_and_reasoning(agent_response)
            actions[player] = action

            # Always show both action number and action name (universal solution)
            try:
                action_name = env.state.action_to_string(player, action)
            except Exception:
                action_name = str(action)
            game_log.append(f"  Player {player} chooses action {action} ({action_name})")
            if reasoning and reasoning != "None":
                reasoning_preview = reasoning[:100] + ("..." if len(reasoning) > 100 else "")
                game_log.append(f"    Reasoning: {reasoning_preview}")
        return actions
    else:
        # Turn-based game
        current_player = env.state.current_player()
        game_log.append(f"Player {current_player}'s turn")

        agent_response = player_to_agent[current_player](observations[current_player])
        action, reasoning = _extract_action_and_reasoning(agent_response)

        game_log.append(f"  Player {current_player} chooses action {action}")
        if reasoning and reasoning != "None":
            reasoning_preview = reasoning[:100] + ("..." if len(reasoning) > 100 else "")
            game_log.append(f"    Reasoning: {reasoning_preview}")

        return {current_player: action}


def _extract_action_and_reasoning(agent_response):
    """Extract action and reasoning from agent response."""
    if isinstance(agent_response, dict) and "action" in agent_response:
        action = agent_response.get("action", -1)
        reasoning = agent_response.get("reasoning", "None")
        return action, reasoning
    else:
        return agent_response, "None"


def _format_player_info(player_config: Dict[str, Any]) -> str:
    """Format player information for display."""
    player_type = player_config["type"]
    if player_type == "llm":
        model = player_config.get("model", "unknown")
        return f"LLM ({model})"
    else:
        return player_type.replace("_", " ").title()


# For backward compatibility and easy integration
def create_gradio_compatible_config(
    game_name: str,
    player1_type: str,
    player2_type: str,
    player1_model: str = None,
    player2_model: str = None,
    rounds: int = 1
) -> Tuple[Dict[str, Any], str]:
    """
    Create both a config dict and a temp file for maximum compatibility.

    Returns:
        Tuple of (config_dict, temp_file_path)
    """
    config = create_config_for_gradio_game(
        game_name, player1_type, player2_type,
        player1_model, player2_model, rounds
    )
    temp_file = create_temporary_config_file(config)
    return config, temp_file


if __name__ == "__main__":
    # Example usage
    config = create_config_for_gradio_game(
        game_name="tic_tac_toe",
        player1_type="llm",
        player2_type="random",
        player1_model="litellm_groq/llama-3.1-8b-instant",
        rounds=3
    )

    print("Generated configuration:")
    print(yaml.dump(config, default_flow_style=False))
