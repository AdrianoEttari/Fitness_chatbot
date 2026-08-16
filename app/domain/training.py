from pydantic import BaseModel, Field, model_validator

from app.domain.user import FitnessGoal


class Exercise(BaseModel):
    name: str
    sets: int = Field(ge=1, le=10)
    reps: str
    rest_seconds: int = Field(ge=0, le=600)
    notes: str | None = None


class WorkoutDay(BaseModel):
    day: int
    focus: str
    exercises: list[Exercise]


class TrainingPlan(BaseModel):
    goal: FitnessGoal
    days_per_week: int = Field(ge=1, le=7)
    workouts: list[WorkoutDay]
    
    @model_validator(mode="after")
    def validate_number_of_workouts(self):
        if len(self.workouts) != self.days_per_week:
            raise ValueError(
                "The number of workouts must match days_per_week"
            )
        
        return self