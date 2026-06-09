"""Generic text-rendered OpenSpiel environment.

This environment is useful for OpenSpiel games that already provide readable
state and observation strings but do not need a custom board renderer yet.

Research context:
    Gin Rummy and OpenSpiel Solitaire are useful for early multi-game testing,
    but this branch does not yet include custom human-friendly renderers for
    those games. This wrapper keeps the games playable by exposing OpenSpiel's
    own observation/action strings directly to the prompt, UI, logs, and export.
"""

from typing import Any, Dict, Optional

from .open_spiel_env import OpenSpielEnv


class GenericTextEnv(OpenSpielEnv):
    """Environment wrapper that delegates rendering to OpenSpiel text output."""

    def __init__(
        self,
        game: Any,
        game_name: str,
        player_types: Dict[str, str],
        max_game_rounds: int = None,
        seed: Optional[int] = None,
    ):
        # All transition, chance-node, reward, and prompt-generation behavior
        # remains in the shared OpenSpielEnv base class. This class only
        # customizes how states and legal actions are rendered as text.
        super().__init__(game, game_name, player_types, max_game_rounds, seed)

    def get_player_symbol(self, agent_id: int) -> str:
        """Return a generic player label for prompts."""
        return f"Player {agent_id}"

    def describe_legal_actions(self, agent_id: int) -> str:
        """Render legal actions with OpenSpiel's action labels when available."""
        legal = self.state.legal_actions(agent_id)
        labelled_actions = []
        for action in legal:
            try:
                # Card games often have meaningful action labels. When
                # OpenSpiel can describe the action, include that text so the
                # LLM sees more than an opaque integer.
                label = self.state.action_to_string(agent_id, action)
            except Exception:
                # Some games/states may not implement action_to_string.
                # Falling back to the integer keeps the action space valid.
                label = str(action)
            labelled_actions.append(f"{action}: {label}")

        # Return an explicit empty list for states with no legal actions.
        if not labelled_actions:
            return "[]"
        return "\n".join(labelled_actions)

    def render_board(self, agent_id: int) -> str:
        """Return OpenSpiel's text observation for UI display and prompting."""
        try:
            # observation_string is player-relative and may hide private
            # information in imperfect-information games such as Gin Rummy.
            return self.state.observation_string(agent_id)
        except Exception:
            # Fallback to the full state string if player-specific observation
            # rendering is unavailable.
            return str(self.state)
