#pragma once

/**
 * Oscillator functions for adaptive_echo.
 */

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include "adaptive_echo/interpolation.hpp"

namespace adaptive_echo {

namespace detail {
// Simple random number generator (similar to JAX's PRNGKey(42))
inline std::mt19937& get_rng() {
    static std::mt19937 rng(42);
    return rng;
}
}  // namespace detail

/**
 * Generate oscillator waveform with optional frequency modulation.
 */
template <typename T>
inline std::vector<T> osc(const std::vector<T>& time, T freq, T phase_shift, T warmth, T harshness,
                          T amplitude, T noise_level, const std::vector<T>* modulation = nullptr,
                          T* fm_amount = nullptr) {
    constexpr T EPSILON = static_cast<T>(1e-6);
    constexpr T TWO_PI = static_cast<T>(2.0 * M_PI);

    size_t n = time.size();
    std::vector<T> result(n);

    std::normal_distribution<T> dist(static_cast<T>(0.0), static_cast<T>(0.5));

    for (size_t i = 0; i < n; ++i) {
        // Generate noise (fixed seed for deterministic behavior)
        T noise = dist(detail::get_rng());

        // Calculate phase
        T phase = time[i] * freq + phase_shift;
        if (modulation != nullptr && fm_amount != nullptr) {
            phase += (*modulation)[i] * (*fm_amount);
        }
        phase = std::fmod(phase, static_cast<T>(1.0));
        if (phase < 0) phase += 1.0;

        phase = std::clamp(phase, EPSILON, static_cast<T>(1.0) - EPSILON);

        T phase_pow = std::pow(phase, warmth);
        T one_minus_phase_pow = std::pow(static_cast<T>(1.0) - phase, warmth);

        // Handle infinities
        if (!std::isfinite(phase_pow)) phase_pow = 0;
        if (!std::isfinite(one_minus_phase_pow)) one_minus_phase_pow = 0;

        phase = static_cast<T>(0.5) * (phase_pow - one_minus_phase_pow + static_cast<T>(1.0));
        phase *= TWO_PI;

        T sin_val = std::sin(phase);
        T abs_sin = std::abs(sin_val);
        abs_sin = std::clamp(abs_sin, EPSILON, static_cast<T>(1.0));
        T sin_pow = std::pow(abs_sin, harshness);

        if (!std::isfinite(sin_pow)) sin_pow = 0;

        T wave = (sin_val >= 0 ? 1 : -1) * sin_pow * amplitude;

        T noise_interp = static_cast<T>(0.1) * noise_level;

        result[i] = linear_interp(wave, noise, noise_interp);
    }

    return result;
}

/**
 * Generate oscillator waveform with time-varying parameters.
 * This version accepts vectors for parameters that change over time.
 */
template <typename T>
inline std::vector<T> osc_time_varying(
    const std::vector<T>& time, const std::vector<T>& freq, const std::vector<T>& phase_shift,
    const std::vector<T>& warmth, const std::vector<T>& harshness, const std::vector<T>& amplitude,
    const std::vector<T>& noise_level, const std::vector<T>* modulation = nullptr,
    const std::vector<T>* fm_amount = nullptr) {
    constexpr T EPSILON = static_cast<T>(1e-6);
    constexpr T TWO_PI = static_cast<T>(2.0 * M_PI);

    size_t n = time.size();
    std::vector<T> result(n);

    std::normal_distribution<T> dist(static_cast<T>(0.0), static_cast<T>(0.5));

    for (size_t i = 0; i < n; ++i) {
        // Generate noise (fixed seed for deterministic behavior)
        T noise = dist(detail::get_rng());

        // Calculate phase
        T phase = time[i] * freq[i] + phase_shift[i];
        if (modulation != nullptr && fm_amount != nullptr) {
            phase += (*modulation)[i] * (*fm_amount)[i];
        }
        phase = std::fmod(phase, static_cast<T>(1.0));
        if (phase < 0) phase += 1.0;

        phase = std::clamp(phase, EPSILON, static_cast<T>(1.0) - EPSILON);

        T phase_pow = std::pow(phase, warmth[i]);
        T one_minus_phase_pow = std::pow(static_cast<T>(1.0) - phase, warmth[i]);

        // Handle infinities
        if (!std::isfinite(phase_pow)) phase_pow = 0;
        if (!std::isfinite(one_minus_phase_pow)) one_minus_phase_pow = 0;

        phase = static_cast<T>(0.5) * (phase_pow - one_minus_phase_pow + static_cast<T>(1.0));
        phase *= TWO_PI;

        T sin_val = std::sin(phase);
        T abs_sin = std::abs(sin_val);
        abs_sin = std::clamp(abs_sin, EPSILON, static_cast<T>(1.0));
        T sin_pow = std::pow(abs_sin, harshness[i]);

        if (!std::isfinite(sin_pow)) sin_pow = 0;

        T wave = (sin_val >= 0 ? 1 : -1) * sin_pow * amplitude[i];

        T noise_interp = static_cast<T>(0.1) * noise_level[i];

        result[i] = linear_interp(wave, noise, noise_interp);
    }

    return result;
}

/**
 * Generate oscillator waveform with inputs normalized to [0, 1].
 */
template <typename T>
inline std::vector<T> osc_uniform(const std::vector<T>& time, T freq, T phase_shift, T warmth,
                                  T harshness, T amplitude, T noise_level,
                                  const std::vector<T>* modulation = nullptr,
                                  T* fm_amount = nullptr) {
    const T min_freq_log = static_cast<T>(12.0 * std::log2(50.0));    // log2(50) * 12
    const T max_freq_log = static_cast<T>(12.0 * std::log2(2000.0));  // log2(2000) * 12
    T semitones = linear_interp(min_freq_log, max_freq_log, freq);
    freq = std::pow(static_cast<T>(2.0), semitones / static_cast<T>(12.0));

    constexpr T min_phase_shift = static_cast<T>(0.0);
    constexpr T max_phase_shift = static_cast<T>(1.0);
    phase_shift = linear_interp(min_phase_shift, max_phase_shift, phase_shift);

    constexpr T min_warmth = static_cast<T>(1.0) / static_cast<T>(5.0);
    constexpr T max_warmth = static_cast<T>(5.0);
    warmth = exp_interp(min_warmth, max_warmth, warmth);

    constexpr T min_harshness = static_cast<T>(1.0) / static_cast<T>(5.0);
    constexpr T max_harshness = static_cast<T>(5.0);
    harshness = exp_interp(min_harshness, max_harshness, harshness);

    constexpr T min_amplitude = static_cast<T>(0.1);
    constexpr T max_amplitude = static_cast<T>(1.0);
    amplitude = linear_interp(min_amplitude, max_amplitude, amplitude);

    return osc(time, freq, phase_shift, warmth, harshness, amplitude, noise_level, modulation,
               fm_amount);
}

/**
 * Generate oscillator waveform with time-varying uniform inputs.
 * Maps [0,1] to actual parameter ranges for each time step.
 */
template <typename T>
inline std::vector<T> osc_uniform_time_varying(
    const std::vector<T>& time, const std::vector<T>& freq, const std::vector<T>& phase_shift,
    const std::vector<T>& warmth, const std::vector<T>& harshness, const std::vector<T>& amplitude,
    const std::vector<T>& noise_level, const std::vector<T>* modulation = nullptr,
    const std::vector<T>* fm_amount = nullptr) {
    const T min_freq_log = static_cast<T>(12.0 * std::log2(50.0));
    const T max_freq_log = static_cast<T>(12.0 * std::log2(2000.0));

    constexpr T min_phase_shift = static_cast<T>(0.0);
    constexpr T max_phase_shift = static_cast<T>(1.0);

    constexpr T min_warmth = static_cast<T>(1.0) / static_cast<T>(5.0);
    constexpr T max_warmth = static_cast<T>(5.0);

    constexpr T min_harshness = static_cast<T>(1.0) / static_cast<T>(5.0);
    constexpr T max_harshness = static_cast<T>(5.0);

    constexpr T min_amplitude = static_cast<T>(0.1);
    constexpr T max_amplitude = static_cast<T>(1.0);

    size_t n = time.size();
    std::vector<T> freq_mapped(n);
    std::vector<T> phase_shift_mapped(n);
    std::vector<T> warmth_mapped(n);
    std::vector<T> harshness_mapped(n);
    std::vector<T> amplitude_mapped(n);

    for (size_t i = 0; i < n; ++i) {
        T semitones = linear_interp(min_freq_log, max_freq_log, freq[i]);
        freq_mapped[i] = std::pow(static_cast<T>(2.0), semitones / static_cast<T>(12.0));
        phase_shift_mapped[i] = linear_interp(min_phase_shift, max_phase_shift, phase_shift[i]);
        warmth_mapped[i] = exp_interp(min_warmth, max_warmth, warmth[i]);
        harshness_mapped[i] = exp_interp(min_harshness, max_harshness, harshness[i]);
        amplitude_mapped[i] = linear_interp(min_amplitude, max_amplitude, amplitude[i]);
    }

    return osc_time_varying(time, freq_mapped, phase_shift_mapped, warmth_mapped, harshness_mapped,
                            amplitude_mapped, noise_level, modulation, fm_amount);
}

}  // namespace adaptive_echo
