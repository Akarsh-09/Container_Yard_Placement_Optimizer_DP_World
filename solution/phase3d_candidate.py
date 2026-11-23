"""Phase 3D review candidate; the submission default remains unchanged."""
from .my_strategy import MyStrategy


class Phase3DCandidate(MyStrategy):
    def __init__(self):
        super().__init__(mode="survival_risk",timing_key="s3_next_feasible",
                         survival_horizon_hours=float("inf"),
                         survival_discount_hours=240,
                         survival_rotation_sigma_hours=0,
                         use_expanded_survival=True,use_multi_rotation=False)
