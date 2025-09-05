import jax
import jax.numpy as jnp

from adaptive_echo.synth import get_initial_settings, sigmoid_forward


def test_sigmoid_forward():
    params = get_initial_settings(jax.random.PRNGKey(0))
    times = jnp.linspace(0, 1, 100)
    outputs = sigmoid_forward(times, params)
    assert outputs.shape == (100,)
    print(outputs.mean(), outputs.std())
