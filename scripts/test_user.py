from app.domain.user import (
    ExperienceLevel,
    FitnessGoal,
    UserProfile,
)

user = UserProfile(
    age=28,
    height_cm=180,
    weight_kg=80,
    goal=FitnessGoal.MUSCLE_GAIN,
    experience=ExperienceLevel.INTERMEDIATE,
    training_days_per_week=4,
    training_duration_minutes=60,
    # dietary_preferences=["omnivore"],
    # dietary_restrictions=[],
    # injuries_or_limitations=[],
)

print(user)

# python -m scripts.test_user