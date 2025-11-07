import torch
from envelope import env, env_uniform
from oscillator import osc, osc_uniform

'''
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



def synth(
)
'''