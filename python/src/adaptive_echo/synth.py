import jax
import jax.numpy as jnp

from adaptive_echo.audio import normalize_wav, save_wav


def linear_interp(a, b, t):
    return a + (b - a) * t


def exp_interp(a, b, t):
    return a * (b / a) ** t


# envolope generator
def env(time, length, attack, decay, sustain, release):
    value = jnp.where(
        time < attack,
        time / attack,
        jnp.where(
            time < attack + decay,
            1 - (1 - sustain) * (time - attack) / decay,
            jnp.where(
                time < length - release, sustain, sustain * (length - time) / release
            ),
        ),
    )
    value = jnp.where(value < 0, 0, value)
    value = jnp.where(value > 1, 1, value)
    return value


# waveform + noise generator
def osc(rng, time, freq, phase_shift, warmth, harshness, amplitude, noise_level):
    noise = jax.random.normal(rng, time.shape) * 0.25

    phase = (time * freq) % 1

    phase = 0.5 * (phase**warmth - (1 - phase) ** warmth + 1)

    phase += phase_shift
    phase *= 2 * jnp.pi

    sin = jnp.sin(phase)

    wave = jnp.sign(sin) * jnp.abs(sin) ** harshness * amplitude

    noise_interp = 0.25 * noise_level

    return linear_interp(wave, noise, noise_interp)


def env_uniform(time, length, attack, decay, sustain, release):
    min_length = 0.1
    max_length = 1.0
    length = exp_interp(min_length, max_length, length)

    min_attack = 0.05
    max_attack = 0.5
    attack = exp_interp(min_attack, max_attack, attack)

    min_decay = 0.05
    max_decay = 0.5
    decay = exp_interp(min_decay, max_decay, decay)

    min_sustain = 0.1
    max_sustain = 1.0
    sustain = linear_interp(min_sustain, max_sustain, sustain)

    min_release = 0.05
    max_release = 0.5
    release = exp_interp(min_release, max_release, release)

    return env(time, length, attack, decay, sustain, release)


def osc_uniform(
    rng, time, freq, phase_shift, warmth, harshness, amplitude, noise_level
):
    min_freq = jnp.log2(10) * 12
    max_freq = jnp.log2(20000) * 12
    semitones = linear_interp(min_freq, max_freq, freq)
    freq = 2 ** (semitones / 12)

    min_phase_shift = 0
    max_phase_shift = 1
    phase_shift = linear_interp(min_phase_shift, max_phase_shift, phase_shift)

    min_warmth = 1 / 5
    max_warmth = 5
    warmth = exp_interp(min_warmth, max_warmth, warmth)

    min_harshness = 1 / 5
    max_harshness = 5
    harshness = exp_interp(min_harshness, max_harshness, harshness)

    min_amplitude = 0.1
    max_amplitude = 1
    amplitude = linear_interp(min_amplitude, max_amplitude, amplitude)

    return osc(rng, time, freq, phase_shift, warmth, harshness, amplitude, noise_level)


def synth(
    rng,
    time,
    env_vol_settings,
    env_mod_settings,
    osc_a_settings,
    osc_b_settings,
    osc_a_mod_settings,
    osc_b_mod_settings,
):
    env_vol = env_uniform(time, *env_vol_settings)
    env_mod = env_uniform(time, *env_mod_settings)

    osc_a_settings_modulated = linear_interp(
        osc_a_settings, osc_a_mod_settings, env_mod
    )
    osc_b_settings_modulated = linear_interp(
        osc_b_settings, osc_b_mod_settings, env_mod
    )

    rng_a, rng_b = jax.random.split(rng, 2)

    osc_a = osc_uniform(rng_a, time, *osc_a_settings_modulated)
    osc_b = osc_uniform(rng_b, time, *osc_b_settings_modulated)

    return (osc_a + osc_b) * env_vol / 2


synth_parallel = jax.vmap(
    synth, in_axes=(None, 0, None, None, None, None, None, None), out_axes=0
)


def forward(times, params, seed=0):
    return synth_parallel(
        jax.random.PRNGKey(seed),
        times,
        params["env_vol_settings"],
        params["env_mod_settings"],
        params["osc_a_settings"],
        params["osc_b_settings"],
        params["osc_a_mod_settings"],
        params["osc_b_mod_settings"],
    )


def get_initial_settings(rng):
    env_vol_rng, env_mod_rng, osc_a_rng, osc_b_rng, osc_a_mod_rng, osc_b_mod_rng = (
        jax.random.split(rng, 6)
    )
    env_vol_settings = jax.random.normal(env_vol_rng, (5,))
    env_mod_settings = jax.random.normal(env_mod_rng, (5,))
    osc_a_settings = jax.random.normal(osc_a_rng, (6,))
    osc_b_settings = jax.random.normal(osc_b_rng, (6,))
    osc_a_mod_settings = jax.random.normal(osc_a_mod_rng, (6,))
    osc_b_mod_settings = jax.random.normal(osc_b_mod_rng, (6,))

    return {
        "env_vol_settings": env_vol_settings,
        "env_mod_settings": env_mod_settings,
        "osc_a_settings": osc_a_settings,
        "osc_b_settings": osc_b_settings,
        "osc_a_mod_settings": osc_a_mod_settings,
        "osc_b_mod_settings": osc_b_mod_settings,
    }


def sigmoid(x):
    return 0.001 + 0.998 / (1 + jnp.exp(-x))


def sigmoid_forward(times, params, seed=0):
    sigmoid_params = {
        "env_vol_settings": sigmoid(params["env_vol_settings"]),
        "env_mod_settings": sigmoid(params["env_mod_settings"]),
        "osc_a_settings": sigmoid(params["osc_a_settings"]),
        "osc_b_settings": sigmoid(params["osc_b_settings"]),
        "osc_a_mod_settings": sigmoid(params["osc_a_mod_settings"]),
        "osc_b_mod_settings": sigmoid(params["osc_b_mod_settings"]),
    }

    return forward(times, sigmoid_params, seed)


if __name__ == "__main__":
    for seed in range(10):
        params = get_initial_settings(jax.random.PRNGKey(seed))
        sr = 44100
        length = 2
        num_samples = int(sr * length)
        times = jnp.linspace(0, length, num_samples)
        print(times.shape)

        path = f"example_sound_{seed:02d}.wav"

        outputs = normalize_wav(sigmoid_forward(times, params))
        save_wav(path, outputs, sr)

        print(f"saved wav to {path}")
