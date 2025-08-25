#!/usr/bin/env python3
"""Test script to verify the challenge system works correctly."""

import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from challenges import (
    Challenge,
    ChallengeDifficulty,
    ChallengeType,
    get_all_challenges_sorted,
    get_challenge_by_name,
    get_challenges_by_difficulty,
    get_challenges_by_type
)

def test_challenge_system():
    """Test the challenge system."""
    print("Testing challenge system...")
    
    # Test getting all challenges
    challenges = get_all_challenges_sorted()
    print(f"Found {len(challenges)} challenges")
    
    # Test getting challenges by difficulty
    beginner_challenges = get_challenges_by_difficulty(ChallengeDifficulty.BEGINNER)
    print(f"Found {len(beginner_challenges)} beginner challenges")
    
    # Test getting challenges by type
    synthetic_challenges = get_challenges_by_type(ChallengeType.SYNTHETIC)
    print(f"Found {len(synthetic_challenges)} synthetic challenges")
    
    # Test getting a specific challenge
    copy_challenge = get_challenge_by_name("copy_task_beginner")
    assert copy_challenge is not None, "Could not find copy_task_beginner challenge"
    assert copy_challenge.difficulty == ChallengeDifficulty.BEGINNER, "Wrong difficulty"
    assert copy_challenge.challenge_type == ChallengeType.SYNTHETIC, "Wrong type"
    print(f"Found challenge: {copy_challenge.name}")
    
    # Test challenge attributes
    assert hasattr(copy_challenge, 'name'), "Missing name attribute"
    assert hasattr(copy_challenge, 'description'), "Missing description attribute"
    assert hasattr(copy_challenge, 'difficulty'), "Missing difficulty attribute"
    assert hasattr(copy_challenge, 'challenge_type'), "Missing challenge_type attribute"
    assert hasattr(copy_challenge, 'data_config'), "Missing data_config attribute"
    assert hasattr(copy_challenge, 'computational_requirements'), "Missing computational_requirements attribute"
    assert hasattr(copy_challenge, 'expected_duration'), "Missing expected_duration attribute"
    assert hasattr(copy_challenge, 'recommended_hardware'), "Missing recommended_hardware attribute"
    
    print("All tests passed!")

if __name__ == "__main__":
    test_challenge_system()