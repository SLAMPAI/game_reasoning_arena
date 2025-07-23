import pyspiel

# List all available games in OpenSpiel
available_games = pyspiel.registered_names()
print("\n".join(available_games))
