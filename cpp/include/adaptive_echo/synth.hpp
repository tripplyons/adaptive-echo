#pragma once

/**
 * Optimized synthesizer functions for adaptive_echo.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "adaptive_echo/constants.hpp"
#include "adaptive_echo/envelope.hpp"
#include "adaptive_echo/filter.hpp"
#include "adaptive_echo/interpolation.hpp"

namespace adaptive_echo {

namespace detail {
// Thread-local scratch buffers to avoid repeated allocations.
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
[[gnu::always_inline]] inline float fast_sin(float x) {
    const float PI = 3.14159265359f;
    const float TWO_PI = 6.28318530718f;
    x = x - TWO_PI * std::floor(x / TWO_PI + 0.5f);
    float sin_x = std::abs(x);
    float sin_val = (16.0f * sin_x * (PI - sin_x)) / (5.0f * PI * PI - 4.0f * sin_x * (PI - sin_x));
    return (x < 0.0f) ? -sin_val : sin_val;
}

// Fast pow approximation for positive bases
[[gnu::always_inline]] inline float fast_pow(float base, float exp) {
    return std::exp2(exp * std::log2(base));
}

// Deterministic normal noise generator with pre-calculation
struct NoiseProvider {
    static constexpr size_t SIZE = 65536;
    static constexpr size_t MASK = SIZE - 1;
    float table[SIZE];

    NoiseProvider() {
        auto hash = [](uint32_t x) {
            x = ((x >> 16) ^ x) * 0x45d9f3bu;
            x = ((x >> 16) ^ x) * 0x45d9f3bu;
            x = (x >> 16) ^ x;
            return x;
        };
        for (size_t i = 0; i < SIZE; ++i) {
            uint32_t u1_raw = hash(static_cast<uint32_t>(i * 2));
            uint32_t u2_raw = hash(static_cast<uint32_t>(i * 2 + 1));
            float u1 = (u1_raw + 1.0f) / 4294967297.0f;
            float u2 = u2_raw / 4294967296.0f;
            float mag = std::sqrt(-2.0f * std::log(u1));
            float z0 = mag * std::cos(6.28318530718f * u2);
            table[i] = z0 * 0.5f;
        }
    }

    static const NoiseProvider& instance() {
        static NoiseProvider inst;
        return inst;
    }
};
}  // namespace detail

/**
 * Optimized oscillator generation with scalar parameters.
 */
template <typename T>
inline void osc_optimized(const std::vector<T>& time, T freq_scalar, T phase_shift_scalar,
                          T warmth_scalar, T harshness_scalar, T amplitude_scalar,
                          T noise_level_scalar, std::vector<T>& output,
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

    T semitones = min_freq_log + (max_freq_log - min_freq_log) * freq_scalar;
    T freq = std::pow(static_cast<T>(2.0), semitones / static_cast<T>(12.0));
    T phase_shift = min_phase_shift + (max_phase_shift - min_phase_shift) * phase_shift_scalar;

    T warmth_ratio = std::pow(max_warmth / min_warmth, warmth_scalar);
    T warmth = min_warmth * warmth_ratio;
    T harshness_ratio = std::pow(max_harshness / min_harshness, harshness_scalar);
    T harshness = min_harshness * harshness_ratio;

    T amplitude = min_amplitude + (max_amplitude - min_amplitude) * amplitude_scalar;
    T noise_interp = static_cast<T>(0.1) * noise_level_scalar;

    size_t n = time.size();
    const float* noise_table = detail::NoiseProvider::instance().table;

#if defined(_OPENMP)
#pragma omp simd
#endif
    for (size_t i = 0; i < n; ++i) {
        T phase = time[i] * freq + phase_shift;
        if (modulation != nullptr && fm_amount != nullptr) {
            phase += (*modulation)[i] * (*fm_amount)[i];
        }
        phase = phase - std::floor(phase);
        phase = std::clamp(phase, EPSILON, static_cast<T>(1.0) - EPSILON);

        T phase_pow = detail::fast_pow(static_cast<float>(phase), static_cast<float>(warmth));
        T one_minus_phase_pow = detail::fast_pow(static_cast<float>(static_cast<T>(1.0) - phase),
                                                 static_cast<float>(warmth));

        phase = static_cast<T>(0.5) * (phase_pow - one_minus_phase_pow + static_cast<T>(1.0));
        phase *= TWO_PI;

        T sin_val = detail::fast_sin(static_cast<float>(phase));
        T abs_sin = std::abs(sin_val);
        abs_sin = std::clamp(abs_sin, EPSILON, static_cast<T>(1.0));
        T sin_pow = detail::fast_pow(static_cast<float>(abs_sin), static_cast<float>(harshness));

        T wave = (sin_val >= 0 ? 1 : -1) * sin_pow * amplitude;
        T noise = static_cast<T>(noise_table[i & detail::NoiseProvider::MASK]);
        output[i] = wave + (noise - wave) * noise_interp;
    }
}

/**
 * Compatibility wrapper for osc_uniform_optimized.
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
    if (!freq_uniform.empty()) {
        osc_optimized(time, freq_uniform[0], phase_shift_uniform[0], warmth_uniform[0],
                      harshness_uniform[0], amplitude_uniform[0], noise_level_uniform[0], output,
                      modulation, fm_amount);
    }
}

/**
 * Optimized synthesizer with minimal allocations.
 */
template <typename T>
inline std::vector<T> synth_fast(const std::vector<T>& settings, const std::vector<T>& times) {
    size_t n = times.size();
    thread_local detail::SynthScratch<T> scratch;
    scratch.resize(n);

    env_uniform_inplace(times, settings[0], settings[1], settings[2], settings[3], settings[4],
                        scratch.env_vol_a);
    env_uniform_inplace(times, settings[5], settings[6], settings[7], settings[8], settings[9],
                        scratch.env_vol_b);
    env_uniform_inplace(times, settings[10], settings[11], settings[12], settings[13], settings[14],
                        scratch.env_mod);
    env_uniform_inplace(times, settings[39], settings[40], settings[41], settings[42], settings[43],
                        scratch.env_fm);

    T fm_range_low = settings[44];
    T fm_range_high = settings[45];
    for (size_t i = 0; i < n; ++i) {
        scratch.fm_amount[i] = linear_interp(fm_range_low, fm_range_high, scratch.env_fm[i]);
    }

    T env_mod_scalar = (n > 0) ? scratch.env_mod[0] : static_cast<T>(0);

    // Osc B
    T osc_b_freq_scalar = linear_interp(settings[27], settings[28], env_mod_scalar);
    T osc_b_phase_scalar = linear_interp(settings[29], settings[30], env_mod_scalar);
    T osc_b_warmth_scalar = linear_interp(settings[31], settings[32], env_mod_scalar);
    T osc_b_harshness_scalar = linear_interp(settings[33], settings[34], env_mod_scalar);
    T osc_b_amp_scalar = linear_interp(settings[35], settings[36], env_mod_scalar);
    T osc_b_noise_scalar = linear_interp(settings[37], settings[38], env_mod_scalar);

    osc_optimized(times, osc_b_freq_scalar, osc_b_phase_scalar, osc_b_warmth_scalar,
                  osc_b_harshness_scalar, osc_b_amp_scalar, osc_b_noise_scalar,
                  scratch.osc_b_scratch);

    // Osc A
    T osc_a_freq_scalar = linear_interp(settings[15], settings[16], env_mod_scalar);
    T osc_a_phase_scalar = linear_interp(settings[17], settings[18], env_mod_scalar);
    T osc_a_warmth_scalar = linear_interp(settings[19], settings[20], env_mod_scalar);
    T osc_a_harshness_scalar = linear_interp(settings[21], settings[22], env_mod_scalar);
    T osc_a_amp_scalar = linear_interp(settings[23], settings[24], env_mod_scalar);
    T osc_a_noise_scalar = linear_interp(settings[25], settings[26], env_mod_scalar);

    osc_optimized(times, osc_a_freq_scalar, osc_a_phase_scalar, osc_a_warmth_scalar,
                  osc_a_harshness_scalar, osc_a_amp_scalar, osc_a_noise_scalar,
                  scratch.osc_a_scratch, &scratch.osc_b_scratch, &scratch.fm_amount);

    std::vector<T> result(n);
    for (size_t i = 0; i < n; ++i) {
        result[i] = scratch.osc_a_scratch[i] * scratch.env_vol_a[i] +
                    scratch.osc_b_scratch[i] * scratch.env_vol_b[i];
    }

    applyDistortion(settings[50], result);
    FilterParameters<T> filter_params = mapFilterParameters(settings, 46);
    applyFilters(filter_params, result,
                 static_cast<T>(adaptive_echo::constants::OUTPUT_SAMPLE_RATE));

    return result;
}

template <typename T>
inline std::vector<T> synth(const std::vector<T>& settings, const std::vector<T>& times) {
    return synth_fast(settings, times);
}

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
