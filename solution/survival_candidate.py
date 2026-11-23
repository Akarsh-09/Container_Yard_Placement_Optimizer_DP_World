"""Zero-argument review candidate; MyStrategy's default is unchanged."""
from .my_strategy import MyStrategy


class SurvivalCandidate(MyStrategy):
    def __init__(self):
        super().__init__(mode="survival_risk",timing_key="s3_next_feasible",
                         survival_horizon_hours=120,survival_rotation_sigma_hours=0,
                         use_expanded_survival=True)
