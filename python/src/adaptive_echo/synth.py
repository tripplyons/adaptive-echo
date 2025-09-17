import jax
import jax.numpy as jnp

from adaptive_echo.audio import normalize_wav, save_wav


def linear_interp(a, b, t):
    return a + (b - a) * t


def exp_interp(a, b, t):
    return a * (b / a) ** t


# envolope generator
def env(
    # time
    time,
    # total time (after filling with sustain)
    length,
    # attack time
    attack,
    # decay time
    decay,
    # sustain level
    sustain,
    # release time
    release,
):
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
def osc(
    # random number generator (for noise)
    rng,
    # time
    time,
    # frequency
    freq,
    # phase shift
    phase_shift,
    # amount of even harmonics (one axis of the wavetable)
    warmth,
    # amount of higher harmonics (another axis of the wavetable)
    harshness,
    # overall volume
    amplitude,
    # amount of noise
    noise_level,
    # signal for frequency modulation
    modulation=None,
    # amount of frequency modulation
    fm_amount=0,
):
    noise = jax.random.normal(rng, time.shape) * 0.2

    phase = time * freq + phase_shift
    if modulation is not None:
        phase += modulation * fm_amount
    phase %= 1

    phase = 0.5 * (phase**warmth - (1 - phase) ** warmth + 1)

    phase *= 2 * jnp.pi

    sin = jnp.sin(phase)

    wave = jnp.sign(sin) * jnp.abs(sin) ** harshness * amplitude

    noise_interp = 0.2 * noise_level

    return linear_interp(wave, noise, noise_interp)


# use envelope generator with inputs from 0 to 1
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


# use oscillator generator with inputs from 0 to 1
def osc_uniform(
    rng,
    time,
    freq,
    phase_shift,
    warmth,
    harshness,
    amplitude,
    noise_level,
    modulation=None,
    fm_amount=0,
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

    return osc(
        rng,
        time,
        freq,
        phase_shift,
        warmth,
        harshness,
        amplitude,
        noise_level,
        modulation,
        fm_amount,
    )


# synthesize a single sample at a time
def synth(
    rng,  # random number generator
    time,  # time
    env_vol_a_settings,  # envelope for volume of osc_a
    env_vol_b_settings,  # envelope for volume of osc_b
    env_mod_settings,  # envelope for modulation amount
    osc_a_settings,  # settings for when osc_a is at no modulation
    osc_b_settings,  # settings for when osc_b is at no modulation
    osc_a_mod_settings,  # settings for when osc_a is at full modulation
    osc_b_mod_settings,  # settings for when osc_b is at full modulation
    env_fm_setting,  # envelope for frequency modulation amount
    fm_range,  # range of frequency modulation amount
):
    # calculate envelopes
    env_vol_a = env_uniform(time, *env_vol_a_settings)
    env_vol_b = env_uniform(time, *env_vol_b_settings)
    env_mod = env_uniform(time, *env_mod_settings)

    # interpolate settings from modulation
    osc_a_settings_modulated = linear_interp(
        osc_a_settings, osc_a_mod_settings, env_mod
    )
    osc_b_settings_modulated = linear_interp(
        osc_b_settings, osc_b_mod_settings, env_mod
    )

    # calculate frequency modulation amount
    min_fm = 0.005
    max_fm = 0.5
    start_fm = exp_interp(min_fm, max_fm, fm_range[0])
    end_fm = exp_interp(min_fm, max_fm, fm_range[1])
    start_fm = fm_range[0]
    end_fm = fm_range[1]
    env_fm = env_uniform(time, *env_fm_setting)
    fm_amount = linear_interp(start_fm, end_fm, env_fm)

    # calculate oscillators
    rng_a, rng_b = jax.random.split(rng, 2)
    osc_b = osc_uniform(rng_b, time, *osc_b_settings_modulated)
    # a is carrier, b is modulator for FM
    osc_a = osc_uniform(
        rng_a, time, *osc_a_settings_modulated, modulation=osc_b, fm_amount=fm_amount
    )

    # multiply by envelopes
    osc_a = osc_a * env_vol_a
    osc_b = osc_b * env_vol_b

    # add them together
    return osc_a + osc_b


# parallelize the function across multiple times/samples for the same parameters
synth_parallel = jax.vmap(
    synth,
    in_axes=(None, 0, None, None, None, None, None, None, None, None, None),
    out_axes=0,
)


# synthesize a sample from a set of parameters
def forward(times, params, seed=0):
    return synth_parallel(
        jax.random.PRNGKey(seed),
        times,
        params["env_vol_a_settings"],
        params["env_vol_b_settings"],
        params["env_mod_settings"],
        params["osc_a_settings"],
        params["osc_b_settings"],
        params["osc_a_mod_settings"],
        params["osc_b_mod_settings"],
        params["env_fm_setting"],
        params["fm_range"],
    )


# generate random settings for the synthesizer (with the range of all real numbers, using a normal distribution)
def get_initial_settings(rng):
    (
        env_vol_a_rng,
        env_vol_b_rng,
        env_mod_rng,
        osc_a_rng,
        osc_b_rng,
        osc_a_mod_rng,
        osc_b_mod_rng,
        env_fm_rng,
        fm_rng,
    ) = jax.random.split(rng, 9)

    env_vol_a_settings = jax.random.normal(env_vol_a_rng, (5,))
    env_vol_b_settings = jax.random.normal(env_vol_b_rng, (5,))
    env_mod_settings = jax.random.normal(env_mod_rng, (5,))
    osc_a_settings = jax.random.normal(osc_a_rng, (6,))
    osc_b_settings = jax.random.normal(osc_b_rng, (6,))
    osc_a_mod_settings = jax.random.normal(osc_a_mod_rng, (6,))
    osc_b_mod_settings = jax.random.normal(osc_b_mod_rng, (6,))
    env_fm_setting = jax.random.normal(env_fm_rng, (5,))
    fm_range = jax.random.normal(fm_rng, (2,))

    return {
        "env_vol_a_settings": env_vol_a_settings,
        "env_vol_b_settings": env_vol_b_settings,
        "env_mod_settings": env_mod_settings,
        "osc_a_settings": osc_a_settings,
        "osc_b_settings": osc_b_settings,
        "osc_a_mod_settings": osc_a_mod_settings,
        "osc_b_mod_settings": osc_b_mod_settings,
        "env_fm_setting": env_fm_setting,
        "fm_range": fm_range,
    }


# sigmoid function to convert from real numbers to 0 to 1
def sigmoid(x):
    return 0.001 + 0.998 / (1 + jnp.exp(-x))


# convert from real parameters to 0 to 1 parameters and generate a sample
def sigmoid_forward(times, params, seed=0):
    sigmoid_params = {
        "env_vol_a_settings": sigmoid(params["env_vol_a_settings"]),
        "env_vol_b_settings": sigmoid(params["env_vol_b_settings"]),
        "env_mod_settings": sigmoid(params["env_mod_settings"]),
        "osc_a_settings": sigmoid(params["osc_a_settings"]),
        "osc_b_settings": sigmoid(params["osc_b_settings"]),
        "osc_a_mod_settings": sigmoid(params["osc_a_mod_settings"]),
        "osc_b_mod_settings": sigmoid(params["osc_b_mod_settings"]),
        "env_fm_setting": sigmoid(params["env_fm_setting"]),
        "fm_range": sigmoid(params["fm_range"]),
    }

    return forward(times, sigmoid_params, seed)


if __name__ == "__main__":
    for seed in range(10):
        params = get_initial_settings(jax.random.PRNGKey(seed))
        # a standard sample rate
        sr = 44100
        # a length of 2 seconds
        length = 2
        # calculate the number of samples
        num_samples = int(sr * length)
        # create an array of the time of each sample
        times = jnp.linspace(0, length, num_samples)

        # save the wav file
        path = f"example_sound_{seed:02d}.wav"
        outputs = normalize_wav(sigmoid_forward(times, params))
        save_wav(path, outputs, sr)
        print(f"saved wav to {path}")
