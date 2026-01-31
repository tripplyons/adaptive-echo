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

    T w0 = 2.0 * M_PI * cutoff / sampleRate;
    T cos_w0 = std::cos(w0);
    T alpha = std::sin(w0) / (2.0 * q);

    coeffs.b0 = (1.0 + cos_w0) / 2.0;
    coeffs.b1 = -(1.0 + cos_w0);
    coeffs.b2 = (1.0 + cos_w0) / 2.0;
    coeffs.a0 = 1.0 + alpha;
    coeffs.a1 = -2.0 * cos_w0;
    coeffs.a2 = 1.0 - alpha;

    // Normalize coefficients
    coeffs.b0 /= coeffs.a0;
    coeffs.b1 /= coeffs.a0;
    coeffs.b2 /= coeffs.a0;
    coeffs.a1 /= coeffs.a0;
    coeffs.a2 /= coeffs.a0;
    coeffs.a0 = 1.0;

    return coeffs;
}

// Calculate low-pass filter coefficients
template <typename T>
BiquadCoefficients<T> calculateLowPass(T cutoff, T q, T sampleRate) {
    BiquadCoefficients<T> coeffs;

    T w0 = 2.0 * M_PI * cutoff / sampleRate;
    T cos_w0 = std::cos(w0);
    T alpha = std::sin(w0) / (2.0 * q);

    coeffs.b0 = (1.0 - cos_w0) / 2.0;
    coeffs.b1 = (1.0 - cos_w0);
    coeffs.b2 = (1.0 - cos_w0) / 2.0;
    coeffs.a0 = 1.0 + alpha;
    coeffs.a1 = -2.0 * cos_w0;
    coeffs.a2 = 1.0 - alpha;

    // Normalize coefficients
    coeffs.b0 /= coeffs.a0;
    coeffs.b1 /= coeffs.a0;
    coeffs.b2 /= coeffs.a0;
    coeffs.a1 /= coeffs.a0;
    coeffs.a2 /= coeffs.a0;
    coeffs.a0 = 1.0;

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

    // High-pass filter parameters
    // Exponential mapping for frequency: 20Hz to 20kHz
    params.highPassCutoff = 20.0 * std::pow(1000.0, settings[baseIndex]);
    params.highPassSlope = 6.0 + 42.0 * settings[baseIndex + 1];

    // Low-pass filter parameters
    params.lowPassCutoff = 20.0 * std::pow(1000.0, settings[baseIndex + 2]);
    params.lowPassSlope = 6.0 + 42.0 * settings[baseIndex + 3];

    return params;
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
