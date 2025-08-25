"""Challenge system for the HRM/HREM project."""

from typing import Dict, List, Any, Optional
from enum import Enum
from pydantic import BaseModel, Field
from hrm_system.config import DataConfig


class ChallengeDifficulty(str, Enum):
    """Difficulty levels for challenges."""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    RESEARCH = "research"


class ChallengeType(str, Enum):
    """Types of challenges."""
    SYNTHETIC = "synthetic"
    ARC = "arc"
    SUDOKU = "sudoku"
    MAZE = "maze"


class Challenge(BaseModel):
    """A challenge configuration."""
    name: str
    description: str
    difficulty: ChallengeDifficulty
    challenge_type: ChallengeType
    data_config: DataConfig
    computational_requirements: str
    expected_duration: str
    recommended_hardware: str


# Define the challenges
CHALLENGES: Dict[str, Challenge] = {
    "copy_task_beginner": Challenge(
        name="Copy Task (Beginner)",
        description="Simple copy task with short sequences. Perfect for testing basic functionality.",
        difficulty=ChallengeDifficulty.BEGINNER,
        challenge_type=ChallengeType.SYNTHETIC,
        data_config=DataConfig(
            dataset="synthetic",
            synthetic_task="copy",
            num_aug=0
        ),
        computational_requirements="Minimal - runs on any hardware",
        expected_duration="Seconds",
        recommended_hardware="Any CPU or GPU"
    ),
    "reverse_task_beginner": Challenge(
        name="Reverse Task (Beginner)",
        description="Simple sequence reversal task. Tests basic memory and processing capabilities.",
        difficulty=ChallengeDifficulty.BEGINNER,
        challenge_type=ChallengeType.SYNTHETIC,
        data_config=DataConfig(
            dataset="synthetic",
            synthetic_task="reverse",
            num_aug=0
        ),
        computational_requirements="Minimal - runs on any hardware",
        expected_duration="Seconds",
        recommended_hardware="Any CPU or GPU"
    ),
    "copy_task_intermediate": Challenge(
        name="Copy Task (Intermediate)",
        description="Copy task with longer sequences and more complex patterns.",
        difficulty=ChallengeDifficulty.INTERMEDIATE,
        challenge_type=ChallengeType.SYNTHETIC,
        data_config=DataConfig(
            dataset="synthetic",
            synthetic_task="copy",
            num_aug=5
        ),
        computational_requirements="Moderate - requires GPU for reasonable training time",
        expected_duration="Minutes",
        recommended_hardware="GPU with 4GB+ VRAM"
    ),
    "arc_challenge": Challenge(
        name="ARC Challenge",
        description="Abstraction and Reasoning Corpus - measures artificial general intelligence capabilities.",
        difficulty=ChallengeDifficulty.ADVANCED,
        challenge_type=ChallengeType.ARC,
        data_config=DataConfig(
            dataset="arc",
            num_aug=100
        ),
        computational_requirements="High - requires significant GPU memory and compute",
        expected_duration="Hours",
        recommended_hardware="Multi-GPU setup with 24GB+ VRAM total"
    ),
    "sudoku_extreme": Challenge(
        name="Sudoku Extreme",
        description="Extremely difficult Sudoku puzzles requiring advanced reasoning.",
        difficulty=ChallengeDifficulty.ADVANCED,
        challenge_type=ChallengeType.SUDOKU,
        data_config=DataConfig(
            dataset="sudoku",
            num_aug=1000
        ),
        computational_requirements="High - requires significant GPU memory and compute",
        expected_duration="Hours",
        recommended_hardware="GPU with 8GB+ VRAM"
    ),
    "maze_hard": Challenge(
        name="Hard Maze Navigation",
        description="Complex maze navigation requiring path planning and memory.",
        difficulty=ChallengeDifficulty.ADVANCED,
        challenge_type=ChallengeType.MAZE,
        data_config=DataConfig(
            dataset="maze",
            num_aug=100
        ),
        computational_requirements="Moderate to High - requires GPU for reasonable training time",
        expected_duration="Hours",
        recommended_hardware="GPU with 6GB+ VRAM"
    ),
    "research_challenge": Challenge(
        name="Research Challenge",
        description="Full-scale ARC-AGI-2 dataset with maximum augmentation for breakthrough research.",
        difficulty=ChallengeDifficulty.RESEARCH,
        challenge_type=ChallengeType.ARC,
        data_config=DataConfig(
            dataset="arc",
            num_aug=1000
        ),
        computational_requirements="Very High - requires powerful multi-GPU setup",
        expected_duration="Days",
        recommended_hardware="Multi-GPU setup with 48GB+ VRAM total"
    )
}


def get_challenges_by_difficulty(difficulty: ChallengeDifficulty) -> List[Challenge]:
    """Get all challenges for a given difficulty level."""
    return [challenge for challenge in CHALLENGES.values() if challenge.difficulty == difficulty]


def get_challenges_by_type(challenge_type: ChallengeType) -> List[Challenge]:
    """Get all challenges for a given type."""
    return [challenge for challenge in CHALLENGES.values() if challenge.challenge_type == challenge_type]


def get_all_challenges_sorted() -> List[Challenge]:
    """Get all challenges sorted by difficulty."""
    difficulty_order = {
        ChallengeDifficulty.BEGINNER: 0,
        ChallengeDifficulty.INTERMEDIATE: 1,
        ChallengeDifficulty.ADVANCED: 2,
        ChallengeDifficulty.RESEARCH: 3
    }
    
    return sorted(CHALLENGES.values(), key=lambda c: (difficulty_order[c.difficulty], c.name))


def get_challenge_by_name(name: str) -> Optional[Challenge]:
    """Get a challenge by its name."""
    return CHALLENGES.get(name)