from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATSBOMB_DATA_DIR = PROJECT_ROOT / "open-data-master" / "open-data-master" / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"


@dataclass(frozen=True)
class MomentumWeights:
    goal: float = 15.0
    shot_base: float = 1.0
    shot_xg: float = 6.0
    own_goal: float = 5.0
    final_third_pass: float = 0.35
    final_third_carry: float = 0.25
    complete_dribble: float = 0.25
    recovery: float = 0.15

