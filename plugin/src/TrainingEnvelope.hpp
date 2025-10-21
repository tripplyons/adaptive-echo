#pragma once

#include "Parameters.hpp"
#include "Interpolation.hpp"
#include <autodiff/reverse/var.hpp>
#include <algorithm>

autodiff::var env(const autodiff::var& time, const autodiff::var& length,
                  const autodiff::var& attack, const autodiff::var& decay,
                  const autodiff::var& sustain, const autodiff::var& release);

autodiff::var env_uniform(const autodiff::var& time,
                                 const autodiff::var& length_norm,
                                 const autodiff::var& attack_norm,
                                 const autodiff::var& decay_norm,
                                 const autodiff::var& sustain_norm,
                                 const autodiff::var& release_norm)
{
    // Remap normalized length from 0.1s to 1.0s (exponentially)
    const autodiff::var length = exp_interp(0.1, 1.0, length_norm);

    // Remap normalized attack from 0.05s to 0.5s (exponentially)
    const autodiff::var attack = exp_interp(0.005, 0.5, attack_norm);

    // Remap normalized decay from 0.05s to 0.5s (exponentially)
    const autodiff::var decay = exp_interp(0.005, 0.5, decay_norm);

    // Remap normalized sustain from 0.1 to 1.0 (linearly)
    const autodiff::var sustain = linear_interp(0.1, 1.0, sustain_norm);

    // Remap normalized release from 0.05s to 0.5s (exponentially)
    const autodiff::var release = exp_interp(0.005, 0.5, release_norm);

    return env(time, length, attack, decay, sustain, release);
}

autodiff::var env_uniform(const autodiff::var& time,
                          const EnvelopeParameters& env_params)
{
    return env_uniform(time, env_params.length, env_params.attack,
                       env_params.decay, env_params.sustain, env_params.release);
}

autodiff::var env(const autodiff::var& time, const autodiff::var& length,
                  const autodiff::var& attack, const autodiff::var& decay,
                  const autodiff::var& sustain, const autodiff::var& release)
{
    // A small epsilon to prevent division by zero in derivatives
    const double epsilon = 1e-9;
    autodiff::var value;

    if (time < attack) {
        // Attack phase: linear ramp from 0 to 1
        value = time / (attack + epsilon);
    } else if (time < attack + decay) {
        // Decay phase: linear ramp from 1 to sustain level
        value = 1.0 - (1.0 - sustain) * (time - attack) / (decay + epsilon);
    } else if (time < length) { 
        // Sustain phase
        value = sustain;
    } else {
        // Release phase: linear ramp from sustain level to 0
        autodiff::var time_from_end = length + release - time;
        value = sustain * time_from_end / (release + epsilon);
    }

    // Clip the final value to be within the [0, 1] range
    return max(0.0, min(value, 1.0));
}
