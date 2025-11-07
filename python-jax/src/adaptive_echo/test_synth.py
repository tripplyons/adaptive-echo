import jax
import jax.numpy as jnp

from adaptive_echo.synth import get_initial_settings, sigmoid_forward


# make sure the sigmoid function can generate from random parameters
def test_sigmoid_forward():
    params = get_initial_settings(jax.random.PRNGKey(0))
    times = jnp.linspace(0, 1, 100)
    outputs = sigmoid_forward(times, params)

    # make sure the outputs are the right shape
    assert outputs.shape == (100,)
    # make sure the outputs aren't all the same
    assert outputs.std() > 0


# make sure it is differentiable
def test_sigmoid_forward_grad():
    params = get_initial_settings(jax.random.PRNGKey(0))
    times = jnp.linspace(0, 1, 100)

    # calculate some loss function
    def loss_fn(params):
        return sigmoid_forward(times, params).std()

    grad = jax.grad(loss_fn)(params)

    # make sure the gradient is not all zero
    def assert_nonzero(x):
        assert x.std() > 0

    jax.tree.map(assert_nonzero, grad)


# make sure the synth function can be compiled correctly
def test_sigmoid_forward_jit():
    params = get_initial_settings(jax.random.PRNGKey(0))
    times = jnp.linspace(0, 1, 100)

    @jax.jit
    def infer_fn(params):
        return sigmoid_forward(times, params)

    outputs = infer_fn(params)
    jit_outputs = infer_fn(params)

    assert jit_outputs.shape == outputs.shape
    assert jnp.allclose(outputs, jit_outputs)
