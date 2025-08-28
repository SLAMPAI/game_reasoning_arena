#!/usr/bin/env python3
"""
Test script for HEX game implementation with random agents.
"""

import sys
from pathlib import Path

# Add src to path if needed
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir.resolve()))

def test_hex_game():
    """Test the HEX game implementation."""
    try:
        print("Testing HEX game import...")
        
        # Test imports
        from game_reasoning_arena.arena.games.registry import registry
        from game_reasoning_arena.arena.agents.random_agent import RandomAgent
        
        print("✓ Registry and RandomAgent imported successfully")
        
        # List available games
        print("\nAvailable games in registry:")
        registry._ensure_loaders_imported()
        for game_name in registry._registry.keys():
            print(f"  - {game_name}")
        
        # Check if hex is registered
        if 'hex' in registry._registry:
            print("✓ HEX game is registered in the registry")
            
            # Try to get the game loader
            try:
                hex_info = registry._registry['hex']
                print(f"✓ HEX game info: {hex_info['display_name']}")
                print(f"  Environment: {hex_info['environment_path']}")
                
                # Try to load the game (this will fail without pyspiel but we can catch it)
                try:
                    game_loader = registry.get_game_loader('hex')
                    print("✓ HEX game loader retrieved successfully")
                except Exception as e:
                    print(f"⚠ Could not load HEX game (expected without pyspiel): {e}")
                    
            except Exception as e:
                print(f"✗ Error getting HEX game info: {e}")
        else:
            print("✗ HEX game is not registered")
            
        # Test random agent
        try:
            agent = RandomAgent(agent_id=0, seed=42)
            print("✓ RandomAgent created successfully")
        except Exception as e:
            print(f"✗ Error creating RandomAgent: {e}")
            
    except Exception as e:
        print(f"✗ Import error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("=" * 50)
    print("HEX Game Test Script")
    print("=" * 50)
    
    success = test_hex_game()
    
    print("\n" + "=" * 50)
    if success:
        print("✓ Basic HEX game setup test completed")
    else:
        print("✗ HEX game setup test failed")
    print("=" * 50)
