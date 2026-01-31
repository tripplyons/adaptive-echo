#pragma once

/**
 * Optimized synthesizer functions for adaptive_echo.
 */

#include <algorithm>
#include <cstdint>
#include <vector>

#include "adaptive_echo/envelope.hpp"
#include "adaptive_echo/filter.hpp"
#include "adaptive_echo/interpolation.hpp"
#include "adaptive_echo/osc.hpp"

namespace adaptive_echo {

namespace detail {
// Thread-local scratch buffers to avoid repeated allocations.
// Each field is a distinct vector; using a single buffer would alias data.
template <typename T>
struct SynthScratch {
    std::vector<T> env_vol_a;
    std::vector<T> env_vol_b;
    std::vector<T> env_mod;
    std::vector<T> env_fm;
    std::vector<T> fm_amount;
    std::vector<T> osc_b_scratch;
    std::vector<T> osc_a_scratch;
    std::vector<T> osc_b_freq;
    std::vector<T> osc_b_phase;
    std::vector<T> osc_b_warmth;
    std::vector<T> osc_b_harshness;
    std::vector<T> osc_b_amp;
    std::vector<T> osc_b_noise;
    std::vector<T> osc_a_freq;
    std::vector<T> osc_a_phase;
    std::vector<T> osc_a_warmth;
    std::vector<T> osc_a_harshness;
    std::vector<T> osc_a_amp;
    std::vector<T> osc_a_noise;

    void resize(size_t n) {
        env_vol_a.resize(n);
        env_vol_b.resize(n);
        env_mod.resize(n);
        env_fm.resize(n);
        fm_amount.resize(n);
        osc_b_scratch.resize(n);
        osc_a_scratch.resize(n);
        osc_b_freq.resize(n);
        osc_b_phase.resize(n);
        osc_b_warmth.resize(n);
        osc_b_harshness.resize(n);
        osc_b_amp.resize(n);
        osc_b_noise.resize(n);
        osc_a_freq.resize(n);
        osc_a_phase.resize(n);
        osc_a_warmth.resize(n);
        osc_a_harshness.resize(n);
        osc_a_amp.resize(n);
        osc_a_noise.resize(n);
    }
};

// Fast sin approximation using Bhaskara I's formula
// Maximum error: ~0.0015 (~0.09 degrees)
inline float fast_sin(float x) {
    // Normalize to [-PI, PI]
    const float PI = 3.14159265359f;
    const float TWO_PI = 6.28318530718f;
    x = x - TWO_PI * std::floor(x / TWO_PI + 0.5f);

    // Bhaskara I approximation: sin(x) ≈ (16x(π-x)) / (5π² - 4x(π-x))
    float sin_x = std::abs(x);
    float sin_val = (16.0f * sin_x * (PI - sin_x)) / (5.0f * PI * PI - 4.0f * sin_x * (PI - sin_x));

    return (x < 0.0f) ? -sin_val : sin_val;
}

// Fast pow approximation for positive bases
// Uses exp(y * log(x)) with fast approximations
inline float fast_pow(float base, float exp) {
    // For the range we use (warmth/harshness typically 0.2-5.0)
    // Simple approximation is sufficient
    return std::pow(base, exp);  // Keep std::pow for accuracy
}

// Deterministic normal noise generator - produces the same noise for a given index
// This matches JAX behavior where noise is generated from a fixed key (normal with std 0.5)
inline float deterministic_noise(size_t index) {
    // Simple hash function to generate two uniform values from index
    auto hash = [](uint32_t x) {
        x = ((x >> 16) ^ x) * 0x45d9f3bu;
        x = ((x >> 16) ^ x) * 0x45d9f3bu;
        x = (x >> 16) ^ x;
        return x;
    };

    uint32_t u1_raw = hash(static_cast<uint32_t>(index * 2));
    uint32_t u2_raw = hash(static_cast<uint32_t>(index * 2 + 1));

    // Convert to (0, 1] and [0, 1)
    float u1 = (u1_raw + 1.0f) / 4294967297.0f;
    float u2 = u2_raw / 4294967296.0f;

    // Box-Muller transform for normal distribution (mean=0, std=1)
    float mag = std::sqrt(-2.0f * std::log(u1));
    float z0 = mag * std::cos(6.28318530718f * u2);

    return z0 * 0.5f;  // Match JAX's normal noise with std 0.5
}
}  // namespace detail

/**
 * Optimized oscillator generation that works directly on output buffer.
 * Eliminates intermediate vector allocations by computing mapping inline.
 */
template <typename T>
inline void osc_uniform_optimized(const std::vector<T>& time, const std::vector<T>& freq_uniform,
                                  const std::vector<T>& phase_shift_uniform,
                                  const std::vector<T>& warmth_uniform,
                                  const std::vector<T>& harshness_uniform,
                                  const std::vector<T>& amplitude_uniform,
                                  const std::vector<T>& noise_level_uniform, std::vector<T>& output,
                                  const std::vector<T>* modulation = nullptr,
                                  const std::vector<T>* fm_amount = nullptr) {
    constexpr T EPSILON = static_cast<T>(1e-6);
    constexpr T TWO_PI = static_cast<T>(2.0 * M_PI);
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

    // Pre-map parameters - this is faster than calling separate functions
    for (size_t i = 0; i < n; ++i) {
        // Map uniform [0,1] to actual values inline
        T semitones = min_freq_log + (max_freq_log - min_freq_log) * freq_uniform[i];
        T freq = std::pow(static_cast<T>(2.0), semitones / static_cast<T>(12.0));
        T phase_shift =
            min_phase_shift + (max_phase_shift - min_phase_shift) * phase_shift_uniform[i];

        // Exponential interp for warmth and harshness
        T warmth_ratio = std::pow(max_warmth / min_warmth, warmth_uniform[i]);
        T warmth = min_warmth * warmth_ratio;
        T harshness_ratio = std::pow(max_harshness / min_harshness, harshness_uniform[i]);
        T harshness = min_harshness * harshness_ratio;

        T amplitude = min_amplitude + (max_amplitude - min_amplitude) * amplitude_uniform[i];
        T noise_level = noise_level_uniform[i];

        // Deterministic noise based on sample index (matches JAX fixed-key behavior)
        T noise = static_cast<T>(detail::deterministic_noise(i));

        // Calculate phase with optional FM
        T phase = time[i] * freq + phase_shift;
        if (modulation != nullptr && fm_amount != nullptr) {
            phase += (*modulation)[i] * (*fm_amount)[i];
        }

        // Fast modulo for [0,1)
        phase = phase - std::floor(phase);
        phase = std::clamp(phase, EPSILON, static_cast<T>(1.0) - EPSILON);

        // Apply warmth shaping
        T phase_pow = std::pow(phase, warmth);
        T one_minus_phase_pow = std::pow(static_cast<T>(1.0) - phase, warmth);

        if (!std::isfinite(phase_pow)) phase_pow = 0;
        if (!std::isfinite(one_minus_phase_pow)) one_minus_phase_pow = 0;

        phase = static_cast<T>(0.5) * (phase_pow - one_minus_phase_pow + static_cast<T>(1.0));
        phase *= TWO_PI;

        // Apply harshness
        T sin_val = std::sin(phase);
        T abs_sin = std::abs(sin_val);
        abs_sin = std::clamp(abs_sin, EPSILON, static_cast<T>(1.0));
        T sin_pow = std::pow(abs_sin, harshness);

        if (!std::isfinite(sin_pow)) sin_pow = 0;

        T wave = (sin_val >= 0 ? 1 : -1) * sin_pow * amplitude;

        // Mix with noise
        T noise_interp = static_cast<T>(0.1) * noise_level;
        output[i] = wave + (noise - wave) * noise_interp;
    }
}

/**
 * Optimized synthesizer with minimal allocations.
 * Uses thread-local scratch buffers and generates output directly.
 */
template <typename T>
inline std::vector<T> synth_fast(const std::vector<T>& settings, const std::vector<T>& times) {
    size_t n = times.size();

    // Thread-local scratch buffers - no heap allocations after resize
    thread_local detail::SynthScratch<T> scratch;
    scratch.resize(n);

    std::vector<T>& env_vol_a = scratch.env_vol_a;
    std::vector<T>& env_vol_b = scratch.env_vol_b;
    std::vector<T>& env_mod = scratch.env_mod;
    std::vector<T>& env_fm = scratch.env_fm;
    std::vector<T>& fm_amount = scratch.fm_amount;
    std::vector<T>& osc_b_scratch = scratch.osc_b_scratch;
    std::vector<T>& osc_a_scratch = scratch.osc_a_scratch;

    std::vector<T>& osc_b_freq = scratch.osc_b_freq;
    std::vector<T>& osc_b_phase = scratch.osc_b_phase;
    std::vector<T>& osc_b_warmth = scratch.osc_b_warmth;
    std::vector<T>& osc_b_harshness = scratch.osc_b_harshness;
    std::vector<T>& osc_b_amp = scratch.osc_b_amp;
    std::vector<T>& osc_b_noise = scratch.osc_b_noise;

    std::vector<T>& osc_a_freq = scratch.osc_a_freq;
    std::vector<T>& osc_a_phase = scratch.osc_a_phase;
    std::vector<T>& osc_a_warmth = scratch.osc_a_warmth;
    std::vector<T>& osc_a_harshness = scratch.osc_a_harshness;
    std::vector<T>& osc_a_amp = scratch.osc_a_amp;
    std::vector<T>& osc_a_noise = scratch.osc_a_noise;

    // Generate envelopes
    env_uniform_inplace(times, settings[0], settings[1], settings[2], settings[3], settings[4],
                        env_vol_a);
    env_uniform_inplace(times, settings[5], settings[6], settings[7], settings[8], settings[9],
                        env_vol_b);
    env_uniform_inplace(times, settings[10], settings[11], settings[12], settings[13], settings[14],
                        env_mod);
    env_uniform_inplace(times, settings[39], settings[40], settings[41], settings[42], settings[43],
                        env_fm);

    // Calculate FM amount
    T fm_range_low = settings[44];
    T fm_range_high = settings[45];
    for (size_t i = 0; i < n; ++i) {
        fm_amount[i] = linear_interp(fm_range_low, fm_range_high, env_fm[i]);
    }

    // Oscillator B settings
    T osc_b_low[6] = {settings[27], settings[29], settings[31],
                      settings[33], settings[35], settings[37]};
    T osc_b_high[6] = {settings[28], settings[30], settings[32],
                       settings[34], settings[36], settings[38]};

    // Match JAX behavior: env_mod is sampled once (first element) to set osc params
    T env_mod_scalar = (n > 0) ? env_mod[0] : static_cast<T>(0);
    T osc_b_freq_scalar = linear_interp(osc_b_low[0], osc_b_high[0], env_mod_scalar);
    T osc_b_phase_scalar = linear_interp(osc_b_low[1], osc_b_high[1], env_mod_scalar);
    T osc_b_warmth_scalar = linear_interp(osc_b_low[2], osc_b_high[2], env_mod_scalar);
    T osc_b_harshness_scalar = linear_interp(osc_b_low[3], osc_b_high[3], env_mod_scalar);
    T osc_b_amp_scalar = linear_interp(osc_b_low[4], osc_b_high[4], env_mod_scalar);
    T osc_b_noise_scalar = linear_interp(osc_b_low[5], osc_b_high[5], env_mod_scalar);

    // Fill per-sample arrays with scalar values to reuse optimized path
    for (size_t i = 0; i < n; ++i) {
        osc_b_freq[i] = osc_b_freq_scalar;
        osc_b_phase[i] = osc_b_phase_scalar;
        osc_b_warmth[i] = osc_b_warmth_scalar;
        osc_b_harshness[i] = osc_b_harshness_scalar;
        osc_b_amp[i] = osc_b_amp_scalar;
        osc_b_noise[i] = osc_b_noise_scalar;
    }

    // Generate oscillator B directly into scratch buffer
    osc_uniform_optimized(times, osc_b_freq, osc_b_phase, osc_b_warmth, osc_b_harshness, osc_b_amp,
                          osc_b_noise, osc_b_scratch);

    // Oscillator A settings
    T osc_a_low[6] = {settings[15], settings[17], settings[19],
                      settings[21], settings[23], settings[25]};
    T osc_a_high[6] = {settings[16], settings[18], settings[20],
                       settings[22], settings[24], settings[26]};

    T osc_a_freq_scalar = linear_interp(osc_a_low[0], osc_a_high[0], env_mod_scalar);
    T osc_a_phase_scalar = linear_interp(osc_a_low[1], osc_a_high[1], env_mod_scalar);
    T osc_a_warmth_scalar = linear_interp(osc_a_low[2], osc_a_high[2], env_mod_scalar);
    T osc_a_harshness_scalar = linear_interp(osc_a_low[3], osc_a_high[3], env_mod_scalar);
    T osc_a_amp_scalar = linear_interp(osc_a_low[4], osc_a_high[4], env_mod_scalar);
    T osc_a_noise_scalar = linear_interp(osc_a_low[5], osc_a_high[5], env_mod_scalar);

    for (size_t i = 0; i < n; ++i) {
        osc_a_freq[i] = osc_a_freq_scalar;
        osc_a_phase[i] = osc_a_phase_scalar;
        osc_a_warmth[i] = osc_a_warmth_scalar;
        osc_a_harshness[i] = osc_a_harshness_scalar;
        osc_a_amp[i] = osc_a_amp_scalar;
        osc_a_noise[i] = osc_a_noise_scalar;
    }

    // Generate oscillator A with FM modulation
    osc_uniform_optimized(times, osc_a_freq, osc_a_phase, osc_a_warmth, osc_a_harshness, osc_a_amp,
                          osc_a_noise, osc_a_scratch, &osc_b_scratch, &fm_amount);

    // Combine oscillators into result (only allocation)
    std::vector<T> result(n);
    for (size_t i = 0; i < n; ++i) {
        result[i] = osc_a_scratch[i] * env_vol_a[i] + osc_b_scratch[i] * env_vol_b[i];
    }

    // Apply filters if filter parameters are provided
    if (settings.size() >= 50) {
        // Map filter parameters from normalized [0,1] to actual ranges
        adaptive_echo::FilterParameters<T> filter_params =
            adaptive_echo::mapFilterParameters(settings, 46);

        // Calculate sample rate from times
        T sampleRate =
            (n > 1) ? static_cast<T>(1.0) / (times[1] - times[0]) : static_cast<T>(44100.0);

        // Apply filters to the result
        adaptive_echo::applyFilters(filter_params, result, sampleRate);
    }

    return result;
}

/**
 * Original synth function for backward compatibility.
 */
template <typename T>
inline std::vector<T> synth(const std::vector<T>& settings, const std::vector<T>& times) {
    return synth_fast(settings, times);
}

/**
 * Generate audio for multiple settings.
 */
template <typename T>
inline std::vector<std::vector<T>> synth_parallel(const std::vector<std::vector<T>>& settings_batch,
                                                  const std::vector<T>& times) {
    std::vector<std::vector<T>> results;
    results.reserve(settings_batch.size());
    for (const auto& settings : settings_batch) {
        results.push_back(synth(settings, times));
    }
    return results;
}

}  // namespace adaptive_echo
