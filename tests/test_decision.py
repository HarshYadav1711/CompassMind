from compassmind.decision import recommend


def test_low_confidence_forces_grounding():
    row = {"stress_level": 2.0, "energy_level": 3.0}
    what, when = recommend("focused", 3, True, 0.2, "morning", row)
    assert "grounding" in what


def test_overwhelmed_high_stress_light_plan():
    row = {"stress_level": 5.0, "energy_level": 4.0}
    what, when = recommend("overwhelmed", 5, False, 0.9, "afternoon", row)
    assert "planning" in what or "rest" in what
