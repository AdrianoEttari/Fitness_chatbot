from pydantic import BaseModel
from langchain_core.tools import tool

class ExerciseInfo(BaseModel):
    name: str
    muscle_groups: list[str]
    equipment: str
    difficulty: str


EXERCISES = [
    ExerciseInfo(
        name="Barbell Bench Press",
        muscle_groups=["chest", "triceps", "shoulders"],
        equipment="barbell",
        difficulty="intermediate",
    ),
    ExerciseInfo(
        name="Barbell Squat",
        muscle_groups=["quadriceps", "glutes"],
        equipment="barbell",
        difficulty="intermediate",
    ),
    ExerciseInfo(
        name="Lat Pulldown",
        muscle_groups=["back", "biceps"],
        equipment="machine",
        difficulty="beginner",
    ),
    ExerciseInfo(
        name="Dumbbell Curl",
        muscle_groups=["biceps"],
        equipment="dumbbell",
        difficulty="beginner",
    ),
]

@tool
def search_exercises(
    muscle_group: str | None = None,
    difficulty: str | None = None,
) -> list[ExerciseInfo]:
    """
    Search the exercise database.

    Use this tool whenever you need to find exercises
    matching a specific muscle group or difficulty level.

    Args:
        muscle_group: Target muscle group, such as chest, back,
            shoulders, quadriceps or biceps.
        difficulty: Exercise difficulty, such as beginner,
            intermediate or advanced.
    """
    results = EXERCISES

    # filter out exercises that are not in the specified muscle group
    if muscle_group:
        results = [
            exercise
            for exercise in results
            if muscle_group.lower() in [
                muscle.lower()
                for muscle in exercise.muscle_groups
            ]
        ]
    # filter out exercises that have a different difficulty level than the specified one
    if difficulty:
        results = [
            exercise
            for exercise in results
            if exercise.difficulty.lower() == difficulty.lower()
        ]

    return results