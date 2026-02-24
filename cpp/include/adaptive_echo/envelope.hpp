#pragma once

/**
 * Envelope generator functions for adaptive_echo.
 */

#include <vector>

#include "adaptive_echo/interpolation.hpp"

namespace adaptive_echo {

/**
 * Generate an ADSR envelope.
 */
template <typename T>
inline std::vector<T> env(const std::vector<T>& time, T length, T attack, T decay, T sustain,
                          T release) {
    size_t n = time.size();
    std::vector<T> value(n);

    for (size_t i = 0; i < n; ++i) {
        T t = time[i];
        if (t < attack) {
            value[i] = t / attack;
        } else if (t < attack + decay) {
            value[i] = static_cast<T>(1.0) - (static_cast<T>(1.0) - sustain) * (t - attack) / decay;
        } else if (t < length - release) {
            value[i] = sustain;
        } else if (t < length) {
            value[i] = sustain * (length - t) / release;
        } else {
            value[i] = static_cast<T>(0.0);
        }
    }

    return value;
}

/**
 * Generate an ADSR envelope with inputs normalized to [0, 1].
 */
template <typename T>
inline std::vector<T> env_uniform(const std::vector<T>& time, T length, T attack, T decay,
                                  T sustain, T release) {
    constexpr T min_length = static_cast<T>(0.2);
    constexpr T max_length = static_cast<T>(2.0);
    length = exp_interp(min_length, max_length, length);

    constexpr T min_attack = static_cast<T>(0.05);
    constexpr T max_attack = static_cast<T>(0.5);
    attack = exp_interp(min_attack, max_attack, attack);

    constexpr T min_decay = static_cast<T>(0.05);
    constexpr T max_decay = static_cast<T>(0.5);
    decay = exp_interp(min_decay, max_decay, decay);

    constexpr T min_sustain = static_cast<T>(0.1);
    constexpr T max_sustain = static_cast<T>(1.0);
    sustain = linear_interp(min_sustain, max_sustain, sustain);

    constexpr T min_release = static_cast<T>(0.05);
    constexpr T max_release = static_cast<T>(0.5);
    release = exp_interp(min_release, max_release, release);

    return env(time, length, attack, decay, sustain, release);
}

/**
 * Generate an ADSR envelope in-place to avoid allocation.
 */
template <typename T>
inline void env_inplace(const std::vector<T>& time, T length, T attack, T decay, T sustain,
                        T release, std::vector<T>& output) {
    size_t n = time.size();
    for (size_t i = 0; i < n; ++i) {
        T t = time[i];
        if (t < attack) {
            output[i] = t / attack;
        } else if (t < attack + decay) {
            output[i] =
                static_cast<T>(1.0) - (static_cast<T>(1.0) - sustain) * (t - attack) / decay;
        } else if (t < length - release) {
            output[i] = sustain;
        } else if (t < length) {
            output[i] = sustain * (length - t) / release;
        } else {
            output[i] = static_cast<T>(0.0);
        }
    }
}

/**
 * Generate an ADSR envelope with uniform inputs in-place.
 */
template <typename T>
inline void env_uniform_inplace(const std::vector<T>& time, T length, T attack, T decay, T sustain,
                                T release, std::vector<T>& output) {
    constexpr T min_length = static_cast<T>(0.2);
    constexpr T max_length = static_cast<T>(2.0);
    length = exp_interp(min_length, max_length, length);

    constexpr T min_attack = static_cast<T>(0.05);
    constexpr T max_attack = static_cast<T>(0.5);
    attack = exp_interp(min_attack, max_attack, attack);

    constexpr T min_decay = static_cast<T>(0.05);
    constexpr T max_decay = static_cast<T>(0.5);
    decay = exp_interp(min_decay, max_decay, decay);

    constexpr T min_sustain = static_cast<T>(0.1);
    constexpr T max_sustain = static_cast<T>(1.0);
    sustain = linear_interp(min_sustain, max_sustain, sustain);

    constexpr T min_release = static_cast<T>(0.05);
    constexpr T max_release = static_cast<T>(0.5);
    release = exp_interp(min_release, max_release, release);

    env_inplace(time, length, attack, decay, sustain, release, output);
}

}  // namespace adaptive_echo
