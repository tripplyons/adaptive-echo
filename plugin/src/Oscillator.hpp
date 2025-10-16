#pragma once

#include <autodiff/reverse/var.hpp>

#include <cmath>
#include <random>

// Forward declarations
autodiff::var osc(std::mt19937 &rng, const autodiff::var &time,
                  const autodiff::var &freq, const autodiff::var &phase_shift,
                  const autodiff::var &warmth, const autodiff::var &harshness,
                  const autodiff::var &amplitude,
                  const autodiff::var &noise_level);

autodiff::var osc(std::mt19937 &rng, const autodiff::var &time,
                  const autodiff::var &freq, const autodiff::var &phase_shift,
                  const autodiff::var &warmth, const autodiff::var &harshness,
                  const autodiff::var &amplitude,
                  const autodiff::var &noise_level,
                  const autodiff::var &modulation,
                  const autodiff::var &fm_amount);

// Returns sign of x
inline autodiff::var sign(const autodiff::var &x) {
    if (x > 0.0)
        return 1.0;
    if (x < 0.0)
        return -1.0;
    return 0.0;
}

inline autodiff::var
osc(std::mt19937 &rng, const autodiff::var &time, const autodiff::var &freq,
    const autodiff::var &phase_shift, const autodiff::var &warmth,
    const autodiff::var &harshness, const autodiff::var &amplitude,
    const autodiff::var &noise_level, const autodiff::var &modulation,
    const autodiff::var &fm_amount) {
    // Generate normally distributed noise
    std::normal_distribution<double> normal(0.0, 0.2);
    autodiff::var noise = normal(rng);

    // Calculate phase with optional frequency modulation
    autodiff::var phase = time * freq + phase_shift;
    phase += modulation * fm_amount;

    // Wrap phase to the [0, 1) interval
    phase -= int(phase);
    if (phase < 0.0) {
        phase += 1.0;
    }

    // Apply warmth to shape the phase, affecting the waveform's duty cycle
    autodiff::var inversePhase = 1.0 - phase;
    phase = 0.5 * (pow(phase, warmth) - pow(inversePhase, warmth) + 1.0);

    // Convert phase to radians for sin function
    phase *= 2.0 * M_PI;

    autodiff::var s = sin(phase);

    // Apply harshness and amplitude
    autodiff::var absS = abs(s);
    autodiff::var wave = sign(s) * pow(absS, harshness) * amplitude;

    // Interpolate between the generated wave and noise
    autodiff::var noise_interp = 0.2 * noise_level;
    return linear_interp(wave, noise, noise_interp);
}

// Overloaded osc function without frequency modulation
inline autodiff::var
osc(std::mt19937 &rng, const autodiff::var &time, const autodiff::var &freq,
    const autodiff::var &phase_shift, const autodiff::var &warmth,
    const autodiff::var &harshness, const autodiff::var &amplitude,
    const autodiff::var &noise_level) {
    // Call the main function with zero modulation
    return osc(rng, time, freq, phase_shift, warmth, harshness, amplitude,
               noise_level, 0.0, 0.0);
}

inline autodiff::var osc_uniform(
    std::mt19937 &rng, const autodiff::var &time,
    const autodiff::var &freq_norm, const autodiff::var &phase_shift_norm,
    const autodiff::var &warmth_norm, const autodiff::var &harshness_norm,
    const autodiff::var &amplitude_norm, const autodiff::var &noise_level_norm,
    const autodiff::var &modulation, const autodiff::var &fm_amount) {
    // Map normalized frequency to a logarithmic scale from 10Hz to 10kHz.
    const autodiff::var min_freq_semitones = log2(10.0) * 12.0;
    const autodiff::var max_freq_semitones = log2(10000.0) * 12.0;
    autodiff::var semitones =
        linear_interp(min_freq_semitones, max_freq_semitones, freq_norm);
    autodiff::var freq = pow(2.0, semitones / 12.0);

    // Phase shift is already 0-1.
    autodiff::var phase_shift = phase_shift_norm;

    // Map normalized warmth and harshness to exponential scales.
    autodiff::var warmth = exp_interp(1.0 / 5.0, 5.0, warmth_norm);
    autodiff::var harshness = exp_interp(1.0 / 5.0, 5.0, harshness_norm);

    // Map normalized amplitude to a linear scale.
    autodiff::var amplitude = linear_interp(0.1, 1.0, amplitude_norm);

    return osc(rng, time, freq, phase_shift, warmth, harshness, amplitude,
               noise_level_norm, modulation, fm_amount);
}

inline autodiff::var osc_uniform(std::mt19937 &rng, const autodiff::var &time,
                                 const autodiff::var &freq_norm,
                                 const autodiff::var &phase_shift_norm,
                                 const autodiff::var &warmth_norm,
                                 const autodiff::var &harshness_norm,
                                 const autodiff::var &amplitude_norm,
                                 const autodiff::var &noise_level_norm) {
    return osc_uniform(rng, time, freq_norm, phase_shift_norm, warmth_norm,
                       harshness_norm, amplitude_norm, noise_level_norm, 0.0,
                       0.0);
}