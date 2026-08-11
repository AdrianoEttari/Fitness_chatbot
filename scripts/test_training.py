from app.domain.training import (
    Exercise,
    TrainingPlan,
    WorkoutDay,
)
from app.domain.user import FitnessGoal


bench_press = Exercise(
    name="Bench Press",
    sets=3,
    reps="8-10",
    rest_seconds=120,
)

lat_pulldown = Exercise(
    name="Lat Pulldown",
    sets=3,
    reps="10-12",
    rest_seconds=90,
)


day_1 = WorkoutDay(
    day=1,
    focus="Upper Body",
    exercises=[
        bench_press,
        lat_pulldown,
    ],
)


plan = TrainingPlan(
    goal=FitnessGoal.MUSCLE_GAIN,
    days_per_week=4,
    workouts=[day_1],
)

print(plan)

# python -m scripts.test_training