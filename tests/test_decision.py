from compassmind.decision import (
    BOX_BREATHING,
    GROUNDING,
    LIGHT_PLANNING,
    MOVEMENT,
    REST,
    recommend,
)


def test_uncertainty_or_low_conf_favors_safe_actions():
    row = {"stress_level": 2.0, "energy_level": 3.0}
    what, when = recommend("focused", 3, 1, 0.2, "morning", row)
    assert what in (GROUNDING, BOX_BREATHING)


def test_overwhelmed_high_energy_light_planning_or_rest():
    row = {"stress_level": 5.0, "energy_level": 4.0}
    what, when = recommend("overwhelmed", 5, 0, 0.9, "afternoon", row)
    assert what in (LIGHT_PLANNING, REST)


def test_restless_suggests_movement_when_confident():
    row = {"stress_level": 2.0, "energy_level": 4.0}
    what, _ = recommend("restless", 4, 0, 0.85, "morning", row)
    assert what == MOVEMENT
