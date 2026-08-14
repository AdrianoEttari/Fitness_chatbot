from enum import Enum
from pydantic import BaseModel, Field


# Enum is used to return an error when the user inputs a FitnessGoal not available in the list
class FitnessGoal(str, Enum):
    FAT_LOSS = "fat_loss"
    MUSCLE_GAIN = "muscle_gain"
    MAINTENANCE = "maintenance"
    GENERAL_FITNESS = "general_fitness"
    HYROX_IMPROVEMENT = "hyrox_improvement"


class ExperienceLevel(str, Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


class UserProfile(BaseModel):
    # Physical data
    age: int = Field(ge=18, le=100)
    height_cm: float = Field(gt=100, lt=250)
    weight_kg: float = Field(gt=30, lt=300)

    # Fitness goals
    goal: FitnessGoal
    experience: ExperienceLevel

    # Training constraints
    training_days_per_week: int = Field(ge=1, le=7)
    training_duration_minutes: int = Field(ge=15, le=180)

    # Additional constraints
    dietary_preferences: list[str] = Field(default_factory=list)
    dietary_restrictions: list[str] = Field(default_factory=list)
    injuries_or_limitations: list[str] = Field(default_factory=list)