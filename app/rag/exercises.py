from pydantic import BaseModel
from langchain_core.tools import tool
import json

class ExerciseInfo(BaseModel):
    name: str
    muscle_groups: list[str]
    equipment: str
    difficulty: str


EXERCISES = [
    # CHEST
    ExerciseInfo(
        name="Barbell Bench Press",
        muscle_groups=["chest", "triceps", "shoulders"],
        equipment="barbell",
        difficulty="intermediate",
    ),
    ExerciseInfo(
        name="Incline Barbell Bench Press",
        muscle_groups=["upper chest", "triceps", "shoulders"],
        equipment="barbell",
        difficulty="intermediate",
    ),
    ExerciseInfo(
        name="Dumbbell Bench Press",
        muscle_groups=["chest", "triceps", "shoulders"],
        equipment="dumbbell",
        difficulty="beginner",
    ),
    ExerciseInfo(
        name="Incline Dumbbell Press",
        muscle_groups=["upper chest", "triceps", "shoulders"],
        equipment="dumbbell",
        difficulty="beginner",
    ),
    ExerciseInfo(
        name="Chest Fly",
        muscle_groups=["chest"],
        equipment="dumbbell",
        difficulty="beginner",
    ),
    ExerciseInfo(
        name="Cable Fly",
        muscle_groups=["chest"],
        equipment="cable",
        difficulty="beginner",
    ),
    ExerciseInfo(
        name="Push-Up",
        muscle_groups=["chest", "triceps", "shoulders"],
        equipment="bodyweight",
        difficulty="beginner",
    ),

    # BACK
    ExerciseInfo(
        name="Lat Pulldown",
        muscle_groups=["back", "biceps"],
        equipment="machine",
        difficulty="beginner",
    ),
    ExerciseInfo(
        name="Pull-Up",
        muscle_groups=["back", "biceps"],
        equipment="bodyweight",
        difficulty="advanced",
    ),
    ExerciseInfo(
        name="Chin-Up",
        muscle_groups=["back", "biceps"],
        equipment="bodyweight",
        difficulty="intermediate",
    ),
    ExerciseInfo(
        name="Barbell Row",
        muscle_groups=["back", "biceps"],
        equipment="barbell",
        difficulty="intermediate",
    ),
    ExerciseInfo(
        name="T-Bar Row",
        muscle_groups=["back", "biceps"],
        equipment="machine",
        difficulty="intermediate",
    ),
    ExerciseInfo(
        name="Seated Cable Row",
        muscle_groups=["back", "biceps"],
        equipment="cable",
        difficulty="beginner",
    ),
    ExerciseInfo(
        name="Single-Arm Dumbbell Row",
        muscle_groups=["back", "biceps"],
        equipment="dumbbell",
        difficulty="beginner",
    ),
    ExerciseInfo(
        name="Deadlift",
        muscle_groups=["back", "glutes", "hamstrings"],
        equipment="barbell",
        difficulty="advanced",
    ),

    # SHOULDERS
    ExerciseInfo(
        name="Overhead Press",
        muscle_groups=["shoulders", "triceps"],
        equipment="barbell",
        difficulty="intermediate",
    ),
    ExerciseInfo(
        name="Dumbbell Shoulder Press",
        muscle_groups=["shoulders", "triceps"],
        equipment="dumbbell",
        difficulty="beginner",
    ),
    ExerciseInfo(
        name="Arnold Press",
        muscle_groups=["shoulders"],
        equipment="dumbbell",
        difficulty="intermediate",
    ),
    ExerciseInfo(
        name="Lateral Raise",
        muscle_groups=["shoulders"],
        equipment="dumbbell",
        difficulty="beginner",
    ),
    ExerciseInfo(
        name="Cable Lateral Raise",
        muscle_groups=["shoulders"],
        equipment="cable",
        difficulty="beginner",
    ),
    ExerciseInfo(
        name="Front Raise",
        muscle_groups=["shoulders"],
        equipment="dumbbell",
        difficulty="beginner",
    ),
    ExerciseInfo(
        name="Face Pull",
        muscle_groups=["rear delts", "upper back"],
        equipment="cable",
        difficulty="beginner",
    ),
    ExerciseInfo(
        name="Reverse Pec Deck",
        muscle_groups=["rear delts"],
        equipment="machine",
        difficulty="beginner",
    ),

    # BICEPS
    ExerciseInfo(
        name="Dumbbell Curl",
        muscle_groups=["biceps"],
        equipment="dumbbell",
        difficulty="beginner",
    ),
    ExerciseInfo(
        name="Barbell Curl",
        muscle_groups=["biceps"],
        equipment="barbell",
        difficulty="beginner",
    ),
    ExerciseInfo(
        name="Hammer Curl",
        muscle_groups=["biceps", "forearms"],
        equipment="dumbbell",
        difficulty="beginner",
    ),
    ExerciseInfo(
        name="Preacher Curl",
        muscle_groups=["biceps"],
        equipment="machine",
        difficulty="beginner",
    ),
    ExerciseInfo(
        name="Cable Curl",
        muscle_groups=["biceps"],
        equipment="cable",
        difficulty="beginner",
    ),
    ExerciseInfo(
        name="Concentration Curl",
        muscle_groups=["biceps"],
        equipment="dumbbell",
        difficulty="beginner",
    ),

    # TRICEPS
    ExerciseInfo(
        name="Tricep Pushdown",
        muscle_groups=["triceps"],
        equipment="cable",
        difficulty="beginner",
    ),
    ExerciseInfo(
        name="Overhead Tricep Extension",
        muscle_groups=["triceps"],
        equipment="dumbbell",
        difficulty="beginner",
    ),
    ExerciseInfo(
        name="Skull Crusher",
        muscle_groups=["triceps"],
        equipment="barbell",
        difficulty="intermediate",
    ),
    ExerciseInfo(
        name="Close-Grip Bench Press",
        muscle_groups=["triceps", "chest"],
        equipment="barbell",
        difficulty="intermediate",
    ),
    ExerciseInfo(
        name="Bench Dip",
        muscle_groups=["triceps"],
        equipment="bodyweight",
        difficulty="beginner",
    ),

    # QUADRICEPS
    ExerciseInfo(
        name="Barbell Squat",
        muscle_groups=["quadriceps", "glutes"],
        equipment="barbell",
        difficulty="intermediate",
    ),
    ExerciseInfo(
        name="Front Squat",
        muscle_groups=["quadriceps", "glutes"],
        equipment="barbell",
        difficulty="advanced",
    ),
    ExerciseInfo(
        name="Leg Press",
        muscle_groups=["quadriceps", "glutes"],
        equipment="machine",
        difficulty="beginner",
    ),
    ExerciseInfo(
        name="Hack Squat",
        muscle_groups=["quadriceps", "glutes"],
        equipment="machine",
        difficulty="beginner",
    ),
    ExerciseInfo(
        name="Bulgarian Split Squat",
        muscle_groups=["quadriceps", "glutes"],
        equipment="dumbbell",
        difficulty="intermediate",
    ),
    ExerciseInfo(
        name="Walking Lunge",
        muscle_groups=["quadriceps", "glutes"],
        equipment="dumbbell",
        difficulty="beginner",
    ),
    ExerciseInfo(
        name="Leg Extension",
        muscle_groups=["quadriceps"],
        equipment="machine",
        difficulty="beginner",
    ),

    # HAMSTRINGS & GLUTES
    ExerciseInfo(
        name="Romanian Deadlift",
        muscle_groups=["hamstrings", "glutes"],
        equipment="barbell",
        difficulty="intermediate",
    ),
    ExerciseInfo(
        name="Stiff-Leg Deadlift",
        muscle_groups=["hamstrings", "glutes"],
        equipment="barbell",
        difficulty="intermediate",
    ),
    ExerciseInfo(
        name="Hip Thrust",
        muscle_groups=["glutes", "hamstrings"],
        equipment="barbell",
        difficulty="beginner",
    ),
    ExerciseInfo(
        name="Glute Bridge",
        muscle_groups=["glutes"],
        equipment="bodyweight",
        difficulty="beginner",
    ),
    ExerciseInfo(
        name="Cable Kickback",
        muscle_groups=["glutes"],
        equipment="cable",
        difficulty="beginner",
    ),
    ExerciseInfo(
        name="Leg Curl",
        muscle_groups=["hamstrings"],
        equipment="machine",
        difficulty="beginner",
    ),
    ExerciseInfo(
        name="Good Morning",
        muscle_groups=["hamstrings", "glutes", "lower back"],
        equipment="barbell",
        difficulty="advanced",
    ),

    # CALVES
    ExerciseInfo(
        name="Standing Calf Raise",
        muscle_groups=["calves"],
        equipment="machine",
        difficulty="beginner",
    ),
    ExerciseInfo(
        name="Seated Calf Raise",
        muscle_groups=["calves"],
        equipment="machine",
        difficulty="beginner",
    ),
    ExerciseInfo(
        name="Single-Leg Calf Raise",
        muscle_groups=["calves"],
        equipment="bodyweight",
        difficulty="beginner",
    ),

    # CORE
    ExerciseInfo(
        name="Plank",
        muscle_groups=["core"],
        equipment="bodyweight",
        difficulty="beginner",
    ),
    ExerciseInfo(
        name="Side Plank",
        muscle_groups=["obliques", "core"],
        equipment="bodyweight",
        difficulty="beginner",
    ),
    ExerciseInfo(
        name="Crunch",
        muscle_groups=["abs"],
        equipment="bodyweight",
        difficulty="beginner",
    ),
    ExerciseInfo(
        name="Cable Crunch",
        muscle_groups=["abs"],
        equipment="cable",
        difficulty="beginner",
    ),
    ExerciseInfo(
        name="Hanging Leg Raise",
        muscle_groups=["abs", "hip flexors"],
        equipment="bodyweight",
        difficulty="intermediate",
    ),
    ExerciseInfo(
        name="Russian Twist",
        muscle_groups=["obliques"],
        equipment="bodyweight",
        difficulty="beginner",
    ),
    ExerciseInfo(
        name="Ab Wheel Rollout",
        muscle_groups=["core"],
        equipment="ab wheel",
        difficulty="advanced",
    ),

    # FULL BODY / FUNCTIONAL
    ExerciseInfo(
        name="Farmer's Walk",
        muscle_groups=["forearms", "traps", "core"],
        equipment="dumbbell",
        difficulty="intermediate",
    ),
    ExerciseInfo(
        name="Kettlebell Swing",
        muscle_groups=["glutes", "hamstrings", "core"],
        equipment="kettlebell",
        difficulty="intermediate",
    ),
    ExerciseInfo(
        name="Thruster",
        muscle_groups=["quadriceps", "shoulders", "core"],
        equipment="dumbbell",
        difficulty="advanced",
    ),
    ExerciseInfo(
        name="Burpee",
        muscle_groups=["full body"],
        equipment="bodyweight",
        difficulty="advanced",
    ),
]

@tool
def search_exercises(
    muscle_group: str | None = None, # it can be either str or None. The default value is None.
    difficulty: str | None = None,
) -> list[ExerciseInfo]:
    """
    Search the exercise database.

    Use muscle_group only when the user asks for exercises
    targeting a specific muscle group.

    If the user does not specify a muscle group, OMIT the
    muscle_group argument entirely.
    
    DO NOT USE VALUS SUCH AS "all", "any", or "full body" for muscle_group.
    JUST USE THE MUSCLE GROUPS AVAILABLE:
    [chest, back, shoulders, biceps, triceps, quadriceps, hamstrings, glutes, calves, core]

    Args:
        muscle_group:
            Optional target muscle group.
            Examples: "chest", "back", "quadriceps".
            Omit this argument to search across all muscle groups.

        difficulty:
            Optional difficulty filter.
            Examples: "beginner", "intermediate", "advanced".
            Omit this argument to include all difficulty levels.
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

    # return results
    return json.dumps(
        [exercise.model_dump() for exercise in results]
    )