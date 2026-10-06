import pytest

from simulations.environment import Environment
from simulations.room_model import step_room
from utils.rng import RNG


def test_rng_is_seeded_and_returns_values_in_expected_ranges():
    rng1 = RNG(42)
    rng2 = RNG(42)

    assert rng1.gauss(0.0, 1.0) == rng2.gauss(0.0, 1.0)
    assert 2.0 <= RNG(42).uniform(2.0, 3.0) < 3.0
    assert RNG(42).bernoulli(0.0) is False
    assert RNG(42).bernoulli(1.0) is True


def test_rng_rejects_invalid_probability_and_uniform_bounds():
    with pytest.raises(ValueError):
        RNG(42).bernoulli(1.1)
    with pytest.raises(ValueError):
        RNG(42).uniform(2.0, 1.0)


def test_environment_temperature_is_unaffected_by_door_event():
    env = Environment(
        base=5.0,
        amplitude=2.0,
        period_s=8.0,
        door_start_s=2.0,
        door_duration_s=2.0,
    )

    assert env.T_out(0.0) == pytest.approx(5.0)
    assert env.T_out(2.0) == pytest.approx(7.0)
    assert env.T_out(4.0) == pytest.approx(5.0)
    assert env.T_out(8.0) == pytest.approx(env.T_out(0.0))
    assert env.is_door_open(2.0)
    assert not env.is_door_open(4.0)


def test_environment_requires_positive_period():
    with pytest.raises(ValueError, match="period_s"):
        Environment(period_s=0.0).T_out(0.0)


def test_room_model_applies_one_euler_step():
    # dT/dt = (10 - 20)/(2*5) + 1*4/5 = -0.2 degrees C/s.
    assert step_room(20.0, 1, 10.0, 2.0, 5.0, 4.0, 1.0) == pytest.approx(19.8)


def test_room_model_cools_faster_when_door_is_open():
    normal = step_room(20.0, 0, 10.0, 2.0, 5.0, 4.0, 1.0)
    door_open = step_room(
        20.0, 0, 10.0, 2.0, 5.0, 4.0, 1.0,
        door_open=True,
        door_exchange_tau_s=10.0,
    )

    assert door_open < normal
    assert door_open == pytest.approx(18.0)


def test_room_model_adds_seeded_process_noise():
    inputs = (20.0, 0, 10.0, 2.0, 5.0, 4.0, 1.0)

    result1 = step_room(*inputs, process_sigma=1.0, rng=RNG(42))
    result2 = step_room(*inputs, process_sigma=1.0, rng=RNG(42))

    assert result1 == result2


def test_room_model_requires_rng_when_process_noise_is_enabled():
    with pytest.raises(ValueError, match="rng is required"):
        step_room(20.0, 0, 10.0, 2.0, 5.0, 4.0, 1.0, process_sigma=1.0)
