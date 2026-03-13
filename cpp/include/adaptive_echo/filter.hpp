#pragma once
#include <algorithm>
#include <cmath>
#include <vector>

namespace adaptive_echo {

// Filter implementations for the synth

// Biquad filter coefficient calculation
template <typename T>
struct BiquadCoefficients {
    T b0, b1, b2;  // Numerator coefficients
    T a0, a1, a2;  // Denominator coefficients
};

// Calculate high-pass filter coefficients
template <typename T>
BiquadCoefficients<T> calculateHighPass(T cutoff, T q, T sampleRate) {
    BiquadCoefficients<T> coeffs;

    const T two = static_cast<T>(2.0);
    const T one = static_cast<T>(1.0);
    T w0 = two * static_cast<T>(M_PI) * cutoff / sampleRate;
    T cos_w0 = std::cos(w0);
    T alpha = std::sin(w0) / (two * q);

    coeffs.b0 = (one + cos_w0) / two;
    coeffs.b1 = -(one + cos_w0);
    coeffs.b2 = (one + cos_w0) / two;
    coeffs.a0 = one + alpha;
    coeffs.a1 = -two * cos_w0;
    coeffs.a2 = one - alpha;

    // Normalize coefficients
    coeffs.b0 /= coeffs.a0;
    coeffs.b1 /= coeffs.a0;
    coeffs.b2 /= coeffs.a0;
    coeffs.a1 /= coeffs.a0;
    coeffs.a2 /= coeffs.a0;
    coeffs.a0 = one;

    return coeffs;
}

// Calculate low-pass filter coefficients
template <typename T>
BiquadCoefficients<T> calculateLowPass(T cutoff, T q, T sampleRate) {
    BiquadCoefficients<T> coeffs;

    const T two = static_cast<T>(2.0);
    const T one = static_cast<T>(1.0);
    T w0 = two * static_cast<T>(M_PI) * cutoff / sampleRate;
    T cos_w0 = std::cos(w0);
    T alpha = std::sin(w0) / (two * q);

    coeffs.b0 = (one - cos_w0) / two;
    coeffs.b1 = one - cos_w0;
    coeffs.b2 = (one - cos_w0) / two;
    coeffs.a0 = one + alpha;
    coeffs.a1 = -two * cos_w0;
    coeffs.a2 = one - alpha;

    // Normalize coefficients
    coeffs.b0 /= coeffs.a0;
    coeffs.b1 /= coeffs.a0;
    coeffs.b2 /= coeffs.a0;
    coeffs.a1 /= coeffs.a0;
    coeffs.a2 /= coeffs.a0;
    coeffs.a0 = one;

    return coeffs;
}

// Apply biquad filter to audio signal
template <typename T>
void applyBiquadFilter(const BiquadCoefficients<T>& coeffs, std::vector<T>& audio) {
    T x1 = 0.0, x2 = 0.0;  // Previous input samples
    T y1 = 0.0, y2 = 0.0;  // Previous output samples

    for (size_t i = 0; i < audio.size(); ++i) {
        T x0 = audio[i];
        T y0 = coeffs.b0 * x0 + coeffs.b1 * x1 + coeffs.b2 * x2 - coeffs.a1 * y1 - coeffs.a2 * y2;

        audio[i] = y0;

        // Shift samples
        x2 = x1;
        x1 = x0;
        y2 = y1;
        y1 = y0;
    }
}

// Filter parameters mapping from normalized [0,1] to actual ranges
template <typename T>
struct FilterParameters {
    T highPassCutoff;  // 20Hz to 20kHz
    T highPassSlope;   // 6 to 48 dB/octave
    T lowPassCutoff;   // 20Hz to 20kHz
    T lowPassSlope;    // 6 to 48 dB/octave
};

// Map normalized parameters to actual filter ranges
template <typename T>
FilterParameters<T> mapFilterParameters(const std::vector<T>& settings, size_t baseIndex) {
    FilterParameters<T> params;
    const T min_frequency = static_cast<T>(20.0);
    const T frequency_ratio = static_cast<T>(1000.0);
    const T min_slope = static_cast<T>(6.0);
    const T slope_span = static_cast<T>(42.0);

    // High-pass filter parameters
    // Exponential mapping for frequency: 20Hz to 20kHz
    params.highPassCutoff = min_frequency * std::pow(frequency_ratio, settings[baseIndex]);
    params.highPassSlope = min_slope + slope_span * settings[baseIndex + 1];

    // Low-pass filter parameters
    params.lowPassCutoff = min_frequency * std::pow(frequency_ratio, settings[baseIndex + 2]);
    params.lowPassSlope = min_slope + slope_span * settings[baseIndex + 3];

    return params;
}

/**
 * Apply a soft-clipping distortion effect.
 * amount: Normalized [0, 1] parameter controlling the drive/gain.
 */
template <typename T>
void applyDistortion(T amount, std::vector<T>& audio) {
    if (amount <= static_cast<T>(0)) return;

    // Map amount [0, 1] to gain [1, 20] for a noticeable effect
    const T one = static_cast<T>(1.0);
    T gain = one + amount * static_cast<T>(19.0);

    for (size_t i = 0; i < audio.size(); ++i) {
        T x = audio[i] * gain;
        // Soft clipping using the algebraic approximation of tanh: x / (1 + |x|)
        audio[i] = x / (one + std::abs(x));
    }
}

// Apply all filters to audio signal
template <typename T>
void applyFilters(const FilterParameters<T>& params, std::vector<T>& audio, T sampleRate) {
    // High-pass filter
    if (params.highPassCutoff > 21.0) {  // Only apply if above minimum
        // Number of passes to achieve desired slope
        // Each biquad pass is 12dB/octave.
        int hp_passes = static_cast<int>(std::round(params.highPassSlope / 12.0));
        if (hp_passes < 1) hp_passes = 1;  // At least one pass

        // Clamp cutoff to safe range (avoid Nyquist)
        T hp_cutoff = std::clamp(params.highPassCutoff, static_cast<T>(20.0),
                                 sampleRate * static_cast<T>(0.45));

        for (int i = 0; i < hp_passes; ++i) {
            BiquadCoefficients<T> coeffs =
                calculateHighPass(hp_cutoff, static_cast<T>(0.707), sampleRate);
            applyBiquadFilter(coeffs, audio);
        }
    }

    // Low-pass filter
    if (params.lowPassCutoff < 19000.0) {  // Only apply if below maximum
        int lp_passes = static_cast<int>(std::round(params.lowPassSlope / 12.0));
        if (lp_passes < 1) lp_passes = 1;

        T lp_cutoff = std::clamp(params.lowPassCutoff, static_cast<T>(20.0),
                                 sampleRate * static_cast<T>(0.45));

        for (int i = 0; i < lp_passes; ++i) {
            BiquadCoefficients<T> coeffs =
                calculateLowPass(lp_cutoff, static_cast<T>(0.707), sampleRate);
            applyBiquadFilter(coeffs, audio);
        }
    }
}

}  // namespace adaptive_echo
