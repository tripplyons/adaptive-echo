#pragma once

/**
 * Optimized STFT-based audio loss functions for adaptive_echo.
 * Parallelized and vectorized for maximum performance.
 *
 * Features:
 * - OpenMP parallelization across STFT frames
 * - Fixed-position STFT with 1/4 window hop for deterministic evaluation
 * - Precomputed target STFTs at multiple scales and resolutions
 * - SIMD vectorization using compiler intrinsics
 * - Thread-safe loss computation for parallel DE evaluation
 * - Precomputed FFT twiddle factors and bit-reversal tables
 */

#include <pocketfft_hdronly.h>

#include <cmath>
#include <complex>
#include <cstring>
#include <vector>

#include "adaptive_echo/constants.hpp"
#include "constants.hpp"

// SIMD intrinsics support
#if defined(__AVX2__)
#include <immintrin.h>
#define HAS_AVX2 1
#elif defined(__AVX__)
#include <immintrin.h>
#define HAS_AVX 1
#elif defined(__SSE4_2__)
#include <nmmintrin.h>
#define HAS_SSE42 1
#elif defined(__SSE2__)
#include <emmintrin.h>
#define HAS_SSE2 1
#endif

// OpenMP support
#if defined(_OPENMP)
#include <omp.h>
#define HAS_OPENMP 1
#endif

namespace adaptive_echo {

namespace detail {
/**
 * Thread-local scratch buffer pool for STFT computation.
 * Prevents allocation overhead during parallel processing.
 */
template <typename T>
class ScratchBufferPool {
   public:
    static ScratchBufferPool& instance() {
        static ScratchBufferPool pool;
        return pool;
    }

    std::vector<std::complex<T>> acquire(size_t size) {
        thread_local std::vector<std::complex<T>> buffer;
        if (buffer.size() < size) {
            buffer.resize(size);
        }
        return buffer;
    }

   private:
    ScratchBufferPool() = default;
};

/**
 * Precomputed Hann window cache with aligned memory.
 */
template <typename T>
class WindowCache {
   public:
    static const std::vector<T>& get(size_t size) {
        static std::vector<std::vector<T>> cache;
        if (cache.size() <= size) {
            cache.resize(size + 1);
        }
        if (cache[size].empty()) {
            cache[size].resize(size);
            const T TWO_PI = static_cast<T>(2.0 * M_PI);
            for (size_t i = 0; i < size; ++i) {
                cache[size][i] = static_cast<T>(0.5) *
                                 (static_cast<T>(1.0) -
                                  std::cos(TWO_PI * static_cast<T>(i) / static_cast<T>(size - 1)));
            }
        }
        return cache[size];
    }

    static T get_sum(size_t size) {
        const auto& window = get(size);
        T sum = 0;
        for (T v : window) sum += v;
        return sum;
    }
};

/**
 * Vectorized mean calculation using SIMD where available.
 */
template <typename T>
inline T vectorized_mean(const std::vector<T>& data) {
    if (data.empty()) return static_cast<T>(0);

    T sum = 0;
    const size_t n = data.size();

#if defined(HAS_AVX2) && defined(__AVX2__)
    // AVX2: Process 8 floats at a time
    if constexpr (std::is_same_v<T, float>) {
        __m256 sum_vec = _mm256_setzero_ps();
        size_t i = 0;
        for (; i + 7 < n; i += 8) {
            __m256 vec = _mm256_loadu_ps(&data[i]);
            sum_vec = _mm256_add_ps(sum_vec, vec);
        }

        // Horizontal sum
        float sum_arr[8];
        _mm256_storeu_ps(sum_arr, sum_vec);
        for (int j = 0; j < 8; ++j) sum += sum_arr[j];

        // Remainder
        for (; i < n; ++i) sum += data[i];
        return sum / static_cast<T>(n);
    }
#elif defined(HAS_SSE2) && defined(__SSE2__)
    // SSE2: Process 4 floats at a time
    if constexpr (std::is_same_v<T, float>) {
        __m128 sum_vec = _mm_setzero_ps();
        size_t i = 0;
        for (; i + 3 < n; i += 4) {
            __m128 vec = _mm_loadu_ps(&data[i]);
            sum_vec = _mm_add_ps(sum_vec, vec);
        }

        // Horizontal sum
        float sum_arr[4];
        _mm_storeu_ps(sum_arr, sum_vec);
        for (int j = 0; j < 4; ++j) sum += sum_arr[j];

        // Remainder
        for (; i < n; ++i) sum += data[i];
        return sum / static_cast<T>(n);
    }
#endif

// Fallback: OpenMP SIMD
#if defined(HAS_OPENMP)
#pragma omp simd reduction(+ : sum)
#endif
    for (size_t i = 0; i < n; ++i) {
        sum += data[i];
    }

    return sum / static_cast<T>(n);
}

/**
 * Vectorized variance calculation.
 */
template <typename T>
inline T vectorized_variance(const std::vector<T>& data, T mean) {
    if (data.size() < 2) return static_cast<T>(0);

    T var = 0;
    const size_t n = data.size();

#if defined(HAS_OPENMP)
#pragma omp simd reduction(+ : var)
#endif
    for (size_t i = 0; i < n; ++i) {
        T diff = data[i] - mean;
        var += diff * diff;
    }

    return var / static_cast<T>(n);
}

/**
 * Single STFT frame computation using pocketfft.
 * Thread-safe: each call allocates its own buffer to avoid thread conflicts.
 */
template <typename T>
inline void compute_stft_frame(const std::vector<T>& x, const std::vector<T>& window, size_t frame,
                               size_t start, size_t win_length, size_t n_fft, size_t num_freqs,
                               T* result) {
    using complex_t = std::complex<T>;

    // Allocate buffer locally - OpenMP ensures each thread has its own
    std::vector<complex_t> frame_data(n_fft);
    std::vector<complex_t> fft_output(n_fft);

    // Window and zero-pad
    for (size_t i = 0; i < n_fft; ++i) {
        if (i < win_length && start + i < x.size()) {
            frame_data[i] = complex_t(x[start + i] * window[i], 0);
        } else {
            frame_data[i] = complex_t(0, 0);
        }
    }

    // FFT using pocketfft
    pocketfft::shape_t shape{n_fft};
    pocketfft::stride_t stride_in{sizeof(complex_t)};
    pocketfft::stride_t stride_out{sizeof(complex_t)};
    pocketfft::shape_t axes{0};  // Transform along axis 0
    pocketfft::c2c<T>(shape, stride_in, stride_out, axes, true, frame_data.data(),
                      fft_output.data(), T(1));

    // Store magnitudes with window normalization
    // FFT normalization: 1.0 / sum(window) for magnitude spectrum
    T window_sum = WindowCache<T>::get_sum(win_length);
    T scale = static_cast<T>(1.0) / window_sum;
    size_t offset = frame * num_freqs;
    for (size_t freq = 0; freq < num_freqs; ++freq) {
        result[offset + freq] = std::abs(fft_output[freq]) * scale;
    }
}

/**
 * Generate fixed sampling positions for a given window size and signal length.
 * Uses hop = win_length / 4 for evenly spaced coverage.
 * Positions are deterministic and consistent across calls.
 */
template <typename T>
inline std::vector<size_t> generate_fixed_positions(size_t signal_length, size_t win_length) {
    std::vector<size_t> positions;

    if (signal_length < win_length) {
        return positions;
    }

    // Hop size = win_length / 4 (25% overlap)
    size_t hop = win_length / 4;
    if (hop == 0) hop = 1;

    // Generate positions starting at 0, with fixed hop
    for (size_t start = 0; start + win_length <= signal_length; start += hop) {
        positions.push_back(start);
    }

    return positions;
}

/**
 * STFT with fixed sampling positions (hop = win_length / 4).
 * Layout: [frame][freq] for cache locality.
 * Deterministic and consistent across calls.
 */
template <typename T>
inline std::vector<T> stft_fixed(const std::vector<T>& x, size_t win_length, size_t n_fft,
                                 std::vector<size_t>& out_positions) {
    const auto& window = WindowCache<T>::get(win_length);
    size_t num_freqs = n_fft / 2 + 1;

    // Generate fixed positions
    out_positions = generate_fixed_positions<T>(x.size(), win_length);
    size_t num_frames = out_positions.size();

    if (num_frames == 0) {
        return std::vector<T>();
    }

    std::vector<T> result(num_frames * num_freqs);

#if defined(HAS_OPENMP)
#pragma omp parallel for schedule(dynamic, 4)
#endif
    for (size_t frame = 0; frame < num_frames; ++frame) {
        size_t start = out_positions[frame];
        compute_stft_frame(x, window, frame, start, win_length, n_fft, num_freqs, result.data());
    }

    return result;
}

/**
 * Compute STFT with explicit positions.
 * Used to ensure generated uses same positions as precomputed target.
 */
template <typename T>
inline std::vector<T> stft_with_positions(const std::vector<T>& x, size_t win_length, size_t n_fft,
                                          const std::vector<size_t>& positions) {
    const auto& window = WindowCache<T>::get(win_length);
    size_t num_freqs = n_fft / 2 + 1;
    size_t num_frames = positions.size();

    if (num_frames == 0) {
        return std::vector<T>();
    }

    std::vector<T> result(num_frames * num_freqs);

#if defined(HAS_OPENMP)
#pragma omp parallel for schedule(dynamic, 4)
#endif
    for (size_t frame = 0; frame < num_frames; ++frame) {
        size_t start = positions[frame];
        // Clamp start position to valid range
        size_t max_start = (x.size() >= win_length) ? x.size() - win_length : 0;
        if (start > max_start) start = max_start;
        compute_stft_frame(x, window, frame, start, win_length, n_fft, num_freqs, result.data());
    }

    return result;
}

/**
 * Compute perceptual frequency weights (A-weighting + Mel-scale density).
 * Used to give "fair weight" to each frequency in audio similarity loss.
 */
template <typename T>
inline std::vector<T> compute_frequency_weights(size_t num_freqs, size_t n_fft, T sample_rate) {
    std::vector<T> weights(num_freqs);
    const T f_s = sample_rate;
    const T n_f = static_cast<T>(n_fft);

    for (size_t i = 0; i < num_freqs; ++i) {
        T freq = static_cast<T>(i) * f_s / n_f;

        // A-weighting approximation
        T f2 = freq * freq;
        T f4 = f2 * f2;
        T c1 = static_cast<T>(12194.217 * 12194.217);
        T c2 = static_cast<T>(20.601103 * 20.601103);
        T c3 = static_cast<T>(107.65265 * 107.65265);
        T c4 = static_cast<T>(737.86223 * 737.86223);

        T num = c1 * f4;
        T den = (f2 + c2) * std::sqrt((f2 + c3) * (f2 + c4)) * (f2 + c1);
        T ra = num / (den + static_cast<T>(1e-8));

        // Mel-scale slope to balance linear bin density
        // dMel/df = 1 / (1 + f/700)
        T mel_slope = static_cast<T>(1.0) / (static_cast<T>(1.0) + freq / static_cast<T>(700.0));

        weights[i] = ra * mel_slope;
    }

    // Normalize weights to average 1.0 to maintain loss scale
    T sum = 0;
    for (T w : weights) sum += w;
    T inv_mean = static_cast<T>(num_freqs) / (sum + static_cast<T>(1e-8));
    for (size_t i = 0; i < num_freqs; ++i) {
        weights[i] *= inv_mean;
    }

    return weights;
}

/**
 * Vectorized zero-crossing rate computation.
 */
template <typename T>
inline T zero_crossing_rate_fast(const std::vector<T>& x) {
    if (x.size() < 2) return static_cast<T>(0);

    size_t count = 0;
    const size_t n = x.size();

#if defined(HAS_OPENMP)
#pragma omp simd reduction(+ : count)
#endif
    for (size_t i = 1; i < n; ++i) {
        count += (x[i] * x[i - 1] < 0) ? 1 : 0;
    }

    return static_cast<T>(count) / static_cast<T>(n - 1);
}

/**
 * Vectorized spectral convergence loss with frequency weighting.
 */
template <typename T>
inline T spectral_convergence_loss(const std::vector<T>& x_mag, const std::vector<T>& y_mag,
                                   T y_mag_mean, const std::vector<T>& weights, size_t num_freqs) {
    const size_t n = x_mag.size();
    if (n == 0 || num_freqs == 0) return static_cast<T>(0);

    T inv_y_mean = static_cast<T>(1.0) / (y_mag_mean + static_cast<T>(1e-8));
    size_t num_frames = n / num_freqs;

    T sc_loss = 0;
    for (size_t frame = 0; frame < num_frames; ++frame) {
        const T* x_ptr = &x_mag[frame * num_freqs];
        const T* y_ptr = &y_mag[frame * num_freqs];
#if defined(HAS_OPENMP)
#pragma omp simd reduction(+ : sc_loss)
#endif
        for (size_t freq = 0; freq < num_freqs; ++freq) {
            sc_loss += weights[freq] * std::abs(y_ptr[freq] - x_ptr[freq]) * inv_y_mean;
        }
    }

    return sc_loss / static_cast<T>(n);
}

/**
 * Vectorized log-magnitude loss with frequency weighting.
 */
template <typename T>
inline T log_magnitude_loss(const std::vector<T>& x_mag, const std::vector<T>& y_mag,
                            const std::vector<T>& weights, size_t num_freqs) {
    const size_t n = x_mag.size();
    if (n == 0 || num_freqs == 0) return static_cast<T>(0);

    constexpr T EPSILON = static_cast<T>(1e-8);
    size_t num_frames = n / num_freqs;

    T mag_loss = 0;
    for (size_t frame = 0; frame < num_frames; ++frame) {
        const T* x_ptr = &x_mag[frame * num_freqs];
        const T* y_ptr = &y_mag[frame * num_freqs];
#if defined(HAS_OPENMP)
#pragma omp simd reduction(+ : mag_loss)
#endif
        for (size_t freq = 0; freq < num_freqs; ++freq) {
            T log_y = std::log(y_ptr[freq] + EPSILON);
            T log_x = std::log(x_ptr[freq] + EPSILON);
            mag_loss += weights[freq] * std::abs(log_y - log_x);
        }
    }

    return mag_loss / static_cast<T>(n);
}

template <typename T>
inline T weighted_l1_loss(const std::vector<T>& x, const std::vector<T>& y,
                          const std::vector<T>& weights, size_t num_features) {
    const size_t n = x.size();
    if (n == 0 || n != y.size() || num_features == 0) return static_cast<T>(0);

    const size_t num_frames = n / num_features;
    T loss = static_cast<T>(0);
    for (size_t frame = 0; frame < num_frames; ++frame) {
        const T* x_ptr = &x[frame * num_features];
        const T* y_ptr = &y[frame * num_features];
#if defined(HAS_OPENMP)
#pragma omp simd reduction(+ : loss)
#endif
        for (size_t feature = 0; feature < num_features; ++feature) {
            loss += weights[feature] * std::abs(x_ptr[feature] - y_ptr[feature]);
        }
    }

    return loss / static_cast<T>(n);
}

template <typename T>
inline T l1_loss(const std::vector<T>& x, const std::vector<T>& y) {
    const size_t n = std::min(x.size(), y.size());
    if (n == 0) return static_cast<T>(0);

    T loss = static_cast<T>(0);
#if defined(HAS_OPENMP)
#pragma omp simd reduction(+ : loss)
#endif
    for (size_t i = 0; i < n; ++i) {
        loss += std::abs(x[i] - y[i]);
    }

    return loss / static_cast<T>(n);
}

template <typename T>
inline std::vector<T> normalize_shape(std::vector<T> values) {
    if (values.empty()) {
        return values;
    }

    T peak = static_cast<T>(0);
    for (const T value : values) {
        peak = std::max(peak, std::abs(value));
    }

    if (peak <= static_cast<T>(1e-8)) {
        return values;
    }

    const T inv_peak = static_cast<T>(1) / peak;
#if defined(HAS_OPENMP)
#pragma omp simd
#endif
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] *= inv_peak;
    }

    return values;
}

template <typename T>
inline std::vector<T> compute_envelope(const std::vector<T>& x, size_t window = 1024,
                                       size_t hop = 256) {
    if (x.empty()) {
        return {};
    }

    window = std::max<size_t>(1, std::min(window, x.size()));
    hop = std::max<size_t>(1, hop);
    const size_t num_frames = 1 + (x.size() - 1) / hop;
    std::vector<T> envelope(num_frames, static_cast<T>(0));

    for (size_t frame = 0; frame < num_frames; ++frame) {
        const size_t start = frame * hop;
        const size_t end = std::min(start + window, x.size());
        T sum = static_cast<T>(0);

#if defined(HAS_OPENMP)
#pragma omp simd reduction(+ : sum)
#endif
        for (size_t i = start; i < end; ++i) {
            sum += x[i] * x[i];
        }

        const T mean = sum / static_cast<T>(std::max<size_t>(1, end - start));
        envelope[frame] = std::sqrt(mean);
    }

    return normalize_shape(std::move(envelope));
}

template <typename T>
inline std::vector<T> compute_delta(const std::vector<T>& x) {
    if (x.size() < 2) {
        return {};
    }

    std::vector<T> delta(x.size() - 1, static_cast<T>(0));
#if defined(HAS_OPENMP)
#pragma omp simd
#endif
    for (size_t i = 1; i < x.size(); ++i) {
        delta[i - 1] = x[i] - x[i - 1];
    }

    return delta;
}

template <typename T>
inline std::vector<T> compute_windowed_zcr(const std::vector<T>& x, size_t window = 1024,
                                           size_t hop = 256) {
    if (x.size() < 2) {
        return {};
    }

    window = std::max<size_t>(2, std::min(window, x.size()));
    hop = std::max<size_t>(1, hop);
    const size_t num_frames = 1 + (x.size() - 1) / hop;
    std::vector<T> zcr(num_frames, static_cast<T>(0));

    for (size_t frame = 0; frame < num_frames; ++frame) {
        const size_t start = frame * hop;
        const size_t end = std::min(start + window, x.size());
        if (end - start < 2) {
            continue;
        }

        size_t crossings = 0;
#if defined(HAS_OPENMP)
#pragma omp simd reduction(+ : crossings)
#endif
        for (size_t i = start + 1; i < end; ++i) {
            const bool crossed = (x[i - 1] >= static_cast<T>(0) && x[i] < static_cast<T>(0)) ||
                                 (x[i - 1] < static_cast<T>(0) && x[i] >= static_cast<T>(0));
            crossings += crossed ? 1u : 0u;
        }

        zcr[frame] = static_cast<T>(crossings) / static_cast<T>(end - start - 1);
    }

    return zcr;
}

template <typename T>
inline std::vector<T> compute_band_centroid_trajectory(const std::vector<T>& band_spec,
                                                       size_t num_bands) {
    if (band_spec.empty() || num_bands == 0) {
        return {};
    }

    const size_t num_frames = band_spec.size() / num_bands;
    std::vector<T> centroids(num_frames, static_cast<T>(0));

    for (size_t frame = 0; frame < num_frames; ++frame) {
        const T* band_ptr = &band_spec[frame * num_bands];
        T weighted_sum = static_cast<T>(0);
        T magnitude_sum = static_cast<T>(0);
        for (size_t band = 0; band < num_bands; ++band) {
            const T magnitude = std::max(band_ptr[band], static_cast<T>(0));
            weighted_sum += static_cast<T>(band) * magnitude;
            magnitude_sum += magnitude;
        }
        if (magnitude_sum > static_cast<T>(1e-8)) {
            centroids[frame] = weighted_sum /
                               (magnitude_sum * static_cast<T>(std::max<size_t>(1, num_bands - 1)));
        }
    }

    return centroids;
}

template <typename T>
inline std::vector<T> compute_spectral_flux_trajectory(const std::vector<T>& spec,
                                                       size_t num_features) {
    if (spec.empty() || num_features == 0) {
        return {};
    }

    const size_t num_frames = spec.size() / num_features;
    if (num_frames < 2) {
        return {};
    }

    std::vector<T> flux(num_frames - 1, static_cast<T>(0));
    constexpr T epsilon = static_cast<T>(1e-8);

    for (size_t frame = 1; frame < num_frames; ++frame) {
        const T* prev_ptr = &spec[(frame - 1) * num_features];
        const T* curr_ptr = &spec[frame * num_features];
        T frame_flux = static_cast<T>(0);
#if defined(HAS_OPENMP)
#pragma omp simd reduction(+ : frame_flux)
#endif
        for (size_t feature = 0; feature < num_features; ++feature) {
            const T prev_log = std::log(prev_ptr[feature] + epsilon);
            const T curr_log = std::log(curr_ptr[feature] + epsilon);
            frame_flux += std::max(static_cast<T>(0), curr_log - prev_log);
        }
        flux[frame - 1] = frame_flux / static_cast<T>(num_features);
    }

    return normalize_shape(std::move(flux));
}

template <typename T>
inline std::vector<size_t> compute_log_band_edges(size_t num_freqs, size_t num_bands) {
    if (num_freqs == 0) {
        return {0};
    }

    num_bands = std::max<size_t>(1, std::min(num_bands, num_freqs));
    std::vector<size_t> edges(num_bands + 1);
    edges[0] = 0;
    edges[num_bands] = num_freqs;

    if (num_bands == 1) {
        return edges;
    }

    const T min_bin = static_cast<T>(1);
    const T max_bin = static_cast<T>(std::max<size_t>(1, num_freqs - 1));

    for (size_t band = 1; band < num_bands; ++band) {
        const T alpha = static_cast<T>(band) / static_cast<T>(num_bands);
        const T log_bin = std::exp(std::log(min_bin) +
                                   alpha * (std::log(max_bin) - std::log(min_bin)));
        const size_t edge = static_cast<size_t>(std::round(log_bin));
        edges[band] = std::clamp(edge, edges[band - 1] + 1, num_freqs - (num_bands - band));
    }

    return edges;
}

template <typename T>
inline std::vector<T> band_average_spectrogram(const std::vector<T>& spec, size_t num_freqs,
                                               const std::vector<size_t>& band_edges,
                                               const std::vector<T>& freq_weights) {
    if (spec.empty() || num_freqs == 0 || band_edges.size() < 2 || freq_weights.size() != num_freqs) {
        return {};
    }

    const size_t num_frames = spec.size() / num_freqs;
    const size_t num_bands = band_edges.size() - 1;
    std::vector<T> band_spec(num_frames * num_bands, static_cast<T>(0));

    for (size_t frame = 0; frame < num_frames; ++frame) {
        const T* frame_ptr = &spec[frame * num_freqs];
        T* band_ptr = &band_spec[frame * num_bands];

        for (size_t band = 0; band < num_bands; ++band) {
            const size_t start = band_edges[band];
            const size_t end = band_edges[band + 1];
            T sum = static_cast<T>(0);
            T weight_sum = static_cast<T>(0);
            for (size_t freq = start; freq < end; ++freq) {
                sum += frame_ptr[freq] * freq_weights[freq];
                weight_sum += freq_weights[freq];
            }
            if (weight_sum > static_cast<T>(0)) {
                band_ptr[band] = sum / weight_sum;
            } else {
                band_ptr[band] = sum / static_cast<T>(std::max<size_t>(1, end - start));
            }
        }
    }

    return band_spec;
}

template <typename T>
inline std::vector<T> aggregate_band_weights(const std::vector<T>& freq_weights,
                                             const std::vector<size_t>& band_edges) {
    if (band_edges.size() < 2) {
        return {};
    }

    const size_t num_bands = band_edges.size() - 1;
    std::vector<T> band_weights(num_bands, static_cast<T>(0));
    for (size_t band = 0; band < num_bands; ++band) {
        const size_t start = band_edges[band];
        const size_t end = band_edges[band + 1];
        T sum = static_cast<T>(0);
        for (size_t freq = start; freq < end; ++freq) {
            sum += freq_weights[freq];
        }
        // Preserve the full perceptual mass of each band so the coarse losses
        // reflect how much weighted frequency content the band represents.
        band_weights[band] = sum;
    }

    T total = std::accumulate(band_weights.begin(), band_weights.end(), static_cast<T>(0));
    if (total > static_cast<T>(0)) {
        const T scale = static_cast<T>(num_bands) / total;
        for (T& weight : band_weights) {
            weight *= scale;
        }
    }

    return band_weights;
}

template <typename T>
inline std::vector<T> perceptual_cepstral_weights(const std::vector<T>& source_weights,
                                                  size_t num_coeffs) {
    if (source_weights.empty() || num_coeffs == 0) {
        return {};
    }

    const size_t num_bins = source_weights.size();
    const T dct_scale = static_cast<T>(M_PI) / static_cast<T>(num_bins);
    std::vector<T> weights(num_coeffs, static_cast<T>(0));

    for (size_t coeff = 0; coeff < num_coeffs; ++coeff) {
        T weighted_basis_energy = static_cast<T>(0);
        for (size_t bin = 0; bin < num_bins; ++bin) {
            const T angle = dct_scale * (static_cast<T>(bin) + static_cast<T>(0.5)) *
                            static_cast<T>(coeff);
            weighted_basis_energy += source_weights[bin] * std::abs(std::cos(angle));
        }
        weights[coeff] = weighted_basis_energy / static_cast<T>(num_bins);
    }

    T total = std::accumulate(weights.begin(), weights.end(), static_cast<T>(0));
    if (total > static_cast<T>(0)) {
        const T scale = static_cast<T>(num_coeffs) / total;
        for (T& weight : weights) {
            weight *= scale;
        }
    }

    return weights;
}

template <typename T>
inline std::vector<T> weighted_cepstrum_spectrogram(const std::vector<T>& band_spec, size_t num_bands,
                                                    const std::vector<T>& band_weights,
                                                    size_t num_coeffs) {
    if (band_spec.empty() || num_bands == 0 || num_coeffs == 0) {
        return {};
    }

    const size_t num_frames = band_spec.size() / num_bands;
    std::vector<T> cepstra(num_frames * num_coeffs, static_cast<T>(0));
    constexpr T epsilon = static_cast<T>(1e-8);
    const T dct_scale = static_cast<T>(M_PI) / static_cast<T>(num_bands);

    for (size_t frame = 0; frame < num_frames; ++frame) {
        const T* band_ptr = &band_spec[frame * num_bands];
        T* cep_ptr = &cepstra[frame * num_coeffs];

        for (size_t coeff = 0; coeff < num_coeffs; ++coeff) {
            T sum = static_cast<T>(0);
            for (size_t band = 0; band < num_bands; ++band) {
                const T weighted_log =
                    std::log(band_ptr[band] * band_weights[band] + epsilon);
                const T angle =
                    dct_scale * (static_cast<T>(band) + static_cast<T>(0.5)) * static_cast<T>(coeff);
                sum += weighted_log * std::cos(angle);
            }
            cep_ptr[coeff] = sum / static_cast<T>(num_bands);
        }
    }

    return cepstra;
}

}  // namespace detail

/**
 * Precomputed target features for the four-part perceptual loss:
 * - fine exact-frequency spectrogram matching over time
 * - coarse band-energy spectrogram matching over time
 * - fine cepstral matching over time
 * - coarse cepstral matching over time
 */
template <typename T>
struct TargetFeaturesFast {
    // Fine-grained target STFTs at each scale [scale][frame * freq]
    std::vector<std::vector<T>> stfts;
    std::vector<size_t> fft_sizes;
    std::vector<size_t> num_freqs;
    std::vector<size_t> num_frames;
    std::vector<std::vector<size_t>> positions;
    std::vector<std::vector<T>> fine_weights;
    std::vector<std::vector<T>> fine_cepstra;
    std::vector<std::vector<T>> fine_cepstral_weights;
    std::vector<size_t> num_fine_cepstra;

    // Coarse-grained band trajectories and cepstra at each scale
    std::vector<std::vector<T>> band_stfts;
    std::vector<std::vector<size_t>> band_edges;
    std::vector<std::vector<T>> band_weights;
    std::vector<size_t> num_bands;
    std::vector<std::vector<T>> cepstra;
    std::vector<std::vector<T>> cepstral_weights;
    std::vector<size_t> num_cepstra;

    // Global envelope-shape features, normalized to focus on contour.
    std::vector<T> envelope;
    std::vector<T> envelope_delta;
    std::vector<T> zcr;
    std::vector<T> zcr_delta;
    std::vector<std::vector<T>> band_centroids;
    std::vector<std::vector<T>> band_flux;
};

/**
 * Precompute all target features including STFTs at multiple scales.
 * Uses fixed hop positions (win_length/4) for deterministic evaluation.
 * All computationally expensive operations happen once during construction.
 */
template <typename T>
inline TargetFeaturesFast<T> precompute_target_features_fast(
    const std::vector<T>& target, const std::vector<size_t>& fft_sizes = {2048, 1024, 512}) {
    TargetFeaturesFast<T> features;
    const size_t num_scales = fft_sizes.size();

    features.fft_sizes = fft_sizes;
    features.num_freqs.resize(num_scales);
    features.num_frames.resize(num_scales);
    features.stfts.resize(num_scales);
    features.positions.resize(num_scales);
    features.fine_weights.resize(num_scales);
    features.fine_cepstra.resize(num_scales);
    features.fine_cepstral_weights.resize(num_scales);
    features.num_fine_cepstra.resize(num_scales);
    features.band_stfts.resize(num_scales);
    features.band_edges.resize(num_scales);
    features.band_weights.resize(num_scales);
    features.num_bands.resize(num_scales);
    features.cepstra.resize(num_scales);
    features.cepstral_weights.resize(num_scales);
    features.num_cepstra.resize(num_scales);
    features.band_centroids.resize(num_scales);
    features.band_flux.resize(num_scales);
    features.envelope = detail::compute_envelope(target);
    features.envelope_delta = detail::compute_delta(features.envelope);
    features.zcr = detail::compute_windowed_zcr(target);
    features.zcr_delta = detail::compute_delta(features.zcr);

    for (size_t scale = 0; scale < num_scales; ++scale) {
        size_t n_fft = fft_sizes[scale];
        size_t win_length = n_fft;
        features.num_freqs[scale] = n_fft / 2 + 1;
        features.fine_weights[scale] = detail::compute_frequency_weights(
            features.num_freqs[scale], n_fft, static_cast<T>(constants::TRAINING_SAMPLE_RATE));

        auto positions = detail::generate_fixed_positions<T>(target.size(), win_length);
        features.positions[scale] = positions;
        features.num_frames[scale] = positions.size();

        if (!positions.empty()) {
            // Precompute the exact-frequency target spectrogram once, then derive the
            // fine cepstral and coarse representations from that same spectrogram.
            features.stfts[scale] =
                detail::stft_with_positions(target, win_length, n_fft, positions);
            features.num_fine_cepstra[scale] = std::min<size_t>(16, features.num_freqs[scale]);
            features.fine_cepstral_weights[scale] = detail::perceptual_cepstral_weights<T>(
                features.fine_weights[scale], features.num_fine_cepstra[scale]);
            features.fine_cepstra[scale] = detail::weighted_cepstrum_spectrogram(
                features.stfts[scale], features.num_freqs[scale], features.fine_weights[scale],
                features.num_fine_cepstra[scale]);
            const size_t band_count = std::min<size_t>(16, std::max<size_t>(8, n_fft / 128));
            features.band_edges[scale] =
                detail::compute_log_band_edges<T>(features.num_freqs[scale], band_count);
            features.num_bands[scale] = features.band_edges[scale].size() - 1;
            features.band_stfts[scale] = detail::band_average_spectrogram(
                features.stfts[scale], features.num_freqs[scale], features.band_edges[scale],
                features.fine_weights[scale]);
            features.band_weights[scale] = detail::aggregate_band_weights(
                features.fine_weights[scale], features.band_edges[scale]);
            features.num_cepstra[scale] = std::min<size_t>(8, features.num_bands[scale]);
            features.cepstral_weights[scale] = detail::perceptual_cepstral_weights<T>(
                features.band_weights[scale], features.num_cepstra[scale]);
            features.band_centroids[scale] = detail::compute_band_centroid_trajectory(
                features.band_stfts[scale], features.num_bands[scale]);
            features.band_flux[scale] = detail::compute_spectral_flux_trajectory(
                features.band_stfts[scale], features.num_bands[scale]);
            features.cepstra[scale] = detail::weighted_cepstrum_spectrogram(
                features.band_stfts[scale], features.num_bands[scale], features.band_weights[scale],
                features.num_cepstra[scale]);
        }
    }

    return features;
}

/**
 * Fast single audio loss computation.
 * Thread-safe: uses thread-local buffers internally.
 * Compares generated STFT against precomputed target STFTs at fixed positions.
 */
template <typename T>
inline T compute_audio_loss_fast(const std::vector<T>& generated,
                                 const TargetFeaturesFast<T>& target_features) {
    T fine_spectral_loss = static_cast<T>(0);
    T coarse_spectral_loss = static_cast<T>(0);
    T fine_cepstral_loss = static_cast<T>(0);
    T coarse_cepstral_loss = static_cast<T>(0);
    T envelope_loss = static_cast<T>(0);
    T envelope_delta_loss = static_cast<T>(0);
    T zcr_loss = static_cast<T>(0);
    T zcr_delta_loss = static_cast<T>(0);
    T centroid_loss = static_cast<T>(0);
    T flux_loss = static_cast<T>(0);
    size_t active_scales = 0;
    const size_t num_scales = target_features.fft_sizes.size();

    for (size_t scale = 0; scale < num_scales; ++scale) {
        size_t n_fft = target_features.fft_sizes[scale];
        size_t win_length = n_fft;
        size_t num_freqs = target_features.num_freqs[scale];
        if (target_features.num_frames[scale] == 0 || target_features.stfts[scale].empty()) {
            continue;
        }

        const auto& positions = target_features.positions[scale];
        // Compute the generated spectrogram once per scale, then reuse it for both
        // the fine exact-frequency loss and the coarse band-over-time loss.
        auto x_stft = detail::stft_with_positions(generated, win_length, n_fft, positions);
        if (x_stft.size() != target_features.stfts[scale].size()) {
            continue;
        }

        const auto& y_stft = target_features.stfts[scale];
        fine_spectral_loss += detail::log_magnitude_loss(x_stft, y_stft,
                                                         target_features.fine_weights[scale],
                                                         num_freqs);

        const auto x_fine_cepstra = detail::weighted_cepstrum_spectrogram(
            x_stft, num_freqs, target_features.fine_weights[scale],
            target_features.num_fine_cepstra[scale]);
        if (!x_fine_cepstra.empty() &&
            x_fine_cepstra.size() == target_features.fine_cepstra[scale].size()) {
            fine_cepstral_loss += detail::weighted_l1_loss(
                x_fine_cepstra, target_features.fine_cepstra[scale],
                target_features.fine_cepstral_weights[scale],
                target_features.num_fine_cepstra[scale]);
        }

        const auto x_band_stft = detail::band_average_spectrogram(
            x_stft, num_freqs, target_features.band_edges[scale],
            target_features.fine_weights[scale]);
        if (!x_band_stft.empty() && x_band_stft.size() == target_features.band_stfts[scale].size()) {
            coarse_spectral_loss += detail::log_magnitude_loss(
                x_band_stft, target_features.band_stfts[scale],
                target_features.band_weights[scale], target_features.num_bands[scale]);
            const auto x_band_centroids = detail::compute_band_centroid_trajectory(
                x_band_stft, target_features.num_bands[scale]);
            if (!x_band_centroids.empty() &&
                x_band_centroids.size() == target_features.band_centroids[scale].size()) {
                centroid_loss +=
                    detail::l1_loss(x_band_centroids, target_features.band_centroids[scale]);
            }
            const auto x_band_flux = detail::compute_spectral_flux_trajectory(
                x_band_stft, target_features.num_bands[scale]);
            if (!x_band_flux.empty() &&
                x_band_flux.size() == target_features.band_flux[scale].size()) {
                flux_loss += detail::l1_loss(x_band_flux, target_features.band_flux[scale]);
            }
            const auto x_cepstra = detail::weighted_cepstrum_spectrogram(
                x_band_stft, target_features.num_bands[scale], target_features.band_weights[scale],
                target_features.num_cepstra[scale]);
            if (!x_cepstra.empty() && x_cepstra.size() == target_features.cepstra[scale].size()) {
                coarse_cepstral_loss += detail::weighted_l1_loss(
                    x_cepstra, target_features.cepstra[scale],
                    target_features.cepstral_weights[scale], target_features.num_cepstra[scale]);
            }
        }

        ++active_scales;
    }

    if (active_scales == 0) {
        return static_cast<T>(0);
    }

    fine_spectral_loss /= static_cast<T>(active_scales);
    coarse_spectral_loss /= static_cast<T>(active_scales);
    fine_cepstral_loss /= static_cast<T>(active_scales);
    coarse_cepstral_loss /= static_cast<T>(active_scales);
    centroid_loss /= static_cast<T>(active_scales);
    flux_loss /= static_cast<T>(active_scales);

    if (!target_features.envelope.empty()) {
        const auto generated_envelope = detail::compute_envelope(generated);
        envelope_loss = detail::l1_loss(generated_envelope, target_features.envelope);
        envelope_delta_loss = detail::l1_loss(detail::compute_delta(generated_envelope),
                                              target_features.envelope_delta);
    }
    if (!target_features.zcr.empty()) {
        const auto generated_zcr = detail::compute_windowed_zcr(generated);
        zcr_loss = detail::l1_loss(generated_zcr, target_features.zcr);
        zcr_delta_loss = detail::l1_loss(detail::compute_delta(generated_zcr),
                                         target_features.zcr_delta);
    }

    const T spectral_loss = static_cast<T>(0.36) * fine_spectral_loss +
                            static_cast<T>(0.18) * coarse_spectral_loss +
                            static_cast<T>(0.18) * fine_cepstral_loss +
                            static_cast<T>(0.08) * coarse_cepstral_loss;
    const T shape_loss = static_cast<T>(0.05) * envelope_loss +
                         static_cast<T>(0.02) * envelope_delta_loss +
                         static_cast<T>(0.02) * zcr_loss +
                         static_cast<T>(0.01) * zcr_delta_loss +
                         static_cast<T>(0.05) * centroid_loss +
                         static_cast<T>(0.05) * flux_loss;
    return spectral_loss + shape_loss;
}

/**
 * Batch loss computation - process multiple generated signals in parallel.
 */
template <typename T>
inline std::vector<T> compute_audio_loss_batch(const std::vector<std::vector<T>>& generated_batch,
                                               const TargetFeaturesFast<T>& target_features) {
    size_t batch_size = generated_batch.size();
    std::vector<T> losses(batch_size);

#if defined(HAS_OPENMP)
#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < batch_size; ++i) {
        losses[i] = compute_audio_loss_fast(generated_batch[i], target_features);
    }
#else
    for (size_t i = 0; i < batch_size; ++i) {
        losses[i] = compute_audio_loss_fast(generated_batch[i], target_features);
    }
#endif

    return losses;
}

/**
 * Thread-safe loss function class.
 * Safe to call from multiple threads during parallel DE evaluation.
 * Precomputes all target features at construction time.
 */
template <typename T>
class LossFunction {
   public:
    LossFunction(const std::vector<T>& target,
                 const std::vector<size_t>& fft_sizes = {2048, 1024, 512})
        : target_(target), fft_sizes_(fft_sizes) {
        features_ = precompute_target_features_fast(target, fft_sizes);
    }

    // Single loss computation - thread safe
    T operator()(const std::vector<T>& generated) const {
        return compute_audio_loss_fast(generated, features_);
    }

    // Batch loss computation - parallel processing
    std::vector<T> compute_batch(const std::vector<std::vector<T>>& generated_batch) const {
        return compute_audio_loss_batch(generated_batch, features_);
    }

    // Access precomputed features (for debugging/analysis)
    const TargetFeaturesFast<T>& features() const { return features_; }

   private:
    std::vector<T> target_;
    std::vector<size_t> fft_sizes_;
    TargetFeaturesFast<T> features_;
};

}  // namespace adaptive_echo
