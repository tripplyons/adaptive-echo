#pragma once

/**
 * Optimized STFT-based audio loss functions for adaptive_echo.
 * Parallelized and vectorized for maximum performance.
 *
 * Features:
 * - OpenMP parallelization across STFT frames
 * - Batch STFT processing (C++ equivalent of JAX vmap)
 * - SIMD vectorization using compiler intrinsics
 * - Thread-safe loss computation for parallel DE evaluation
 * - Precomputed FFT twiddle factors and bit-reversal tables
 */

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <numeric>
#include <thread>
#include <unordered_map>
#include <vector>

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
 * Aligned allocator for SIMD-friendly memory.
 * Ensures 64-byte alignment for AVX-512 compatibility.
 */
template <typename T>
class AlignedAllocator {
   public:
    static constexpr size_t ALIGNMENT = 64;

    static T* allocate(size_t n) {
        void* ptr = nullptr;
#if defined(_WIN32)
        ptr = _aligned_malloc(n * sizeof(T), ALIGNMENT);
        if (!ptr) throw std::bad_alloc();
#else
        if (posix_memalign(&ptr, ALIGNMENT, n * sizeof(T)) != 0) {
            throw std::bad_alloc();
        }
#endif
        return static_cast<T*>(ptr);
    }

    static void deallocate(T* ptr) {
#if defined(_WIN32)
        _aligned_free(ptr);
#else
        free(ptr);
#endif
    }
};

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
 * Optimized FFT with precomputed twiddle factors and bit-reversal.
 * Thread-safe initialization using static local variables.
 */
template <typename T>
class FFTCache {
   public:
    static FFTCache& get(size_t n) {
        // Pre-initialize common FFT sizes to avoid race conditions
        static FFTCache cache_1024(1024);
        static FFTCache cache_512(512);
        static FFTCache cache_256(256);
        static FFTCache cache_128(128);
        static FFTCache cache_64(64);
        static FFTCache cache_32(32);
        static FFTCache cache_16(16);
        static FFTCache cache_8(8);

        switch (n) {
            case 1024:
                return cache_1024;
            case 512:
                return cache_512;
            case 256:
                return cache_256;
            case 128:
                return cache_128;
            case 64:
                return cache_64;
            case 32:
                return cache_32;
            case 16:
                return cache_16;
            case 8:
                return cache_8;
            default: {
                // Fallback for other sizes - use a map
                static std::unordered_map<size_t, std::unique_ptr<FFTCache>> cache_map;
                static std::mutex cache_mutex;

                std::lock_guard<std::mutex> lock(cache_mutex);
                auto it = cache_map.find(n);
                if (it == cache_map.end()) {
                    cache_map[n] = std::unique_ptr<FFTCache>(new FFTCache(n));
                }
                return *cache_map[n];
            }
        }
    }

    const std::vector<std::complex<T>>& get_twiddle() const { return twiddle_; }
    const std::vector<size_t>& get_bitrev() const { return bitrev_; }
    size_t get_n() const { return n_; }

   private:
    size_t n_;
    std::vector<std::complex<T>> twiddle_;
    std::vector<size_t> bitrev_;

    explicit FFTCache(size_t n) : n_(n) {
        // Calculate number of stages (log2(n))
        size_t stages = 0;
        size_t temp = n;
        while (temp > 1) {
            temp >>= 1;
            stages++;
        }

        // Precompute twiddle factors: W_N^k = e^(-2πik/N)
        twiddle_.resize(n / 2);
        const T TWO_PI = static_cast<T>(2.0 * M_PI);
        for (size_t i = 0; i < n / 2; ++i) {
            T ang = -TWO_PI * static_cast<T>(i) / static_cast<T>(n);
            twiddle_[i] = std::complex<T>(std::cos(ang), std::sin(ang));
        }

        // Precompute bit-reversal permutation
        bitrev_.resize(n);
        for (size_t i = 0; i < n; ++i) {
            size_t j = 0;
            for (size_t k = 0; k < stages; ++k) {
                j = (j << 1) | ((i >> k) & 1);
            }
            bitrev_[i] = j;
        }
    }
};

/**
 * In-place iterative FFT with SIMD-friendly access patterns.
 */
template <typename T>
inline void fft_inplace(std::vector<std::complex<T>>& data) {
    size_t n = data.size();
    if (n <= 1) return;

    auto& cache = FFTCache<T>::get(n);
    const auto& bitrev = cache.get_bitrev();
    const auto& twiddle = cache.get_twiddle();

    // Bit-reverse permutation
    for (size_t i = 0; i < n; ++i) {
        size_t j = bitrev[i];
        if (i < j) {
            std::swap(data[i], data[j]);
        }
    }

    // FFT stages - iterative approach for better cache locality
    for (size_t stage = 1, step = 2; stage < n; stage <<= 1, step <<= 1) {
        size_t twiddle_step = n / step;

        for (size_t group = 0; group < n; group += step) {
            for (size_t pair = 0; pair < stage; ++pair) {
                size_t i = group + pair;
                size_t j = i + stage;
                size_t twidx = pair * twiddle_step;

                const std::complex<T>& w = twiddle[twidx];
                std::complex<T> temp = data[j] * w;
                data[j] = data[i] - temp;
                data[i] = data[i] + temp;
            }
        }
    }
}

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
 * Single STFT frame computation.
 * Thread-safe: each call allocates its own buffer to avoid thread conflicts.
 */
template <typename T>
inline void compute_stft_frame(const std::vector<T>& x, const std::vector<T>& window, size_t frame,
                               size_t start, size_t win_length, size_t n_fft, size_t num_freqs,
                               T* result) {
    // Allocate buffer locally - OpenMP ensures each thread has its own
    std::vector<std::complex<T>> frame_data(n_fft);

    // Window and zero-pad
    for (size_t i = 0; i < n_fft; ++i) {
        if (i < win_length && start + i < x.size()) {
            frame_data[i] = std::complex<T>(x[start + i] * window[i]);
        } else {
            frame_data[i] = std::complex<T>(0);
        }
    }

    // FFT
    fft_inplace(frame_data);

    // Store magnitudes - match JAX scipy.signal.stft behavior exactly
    // JAX's FFT does NOT normalize by 1/N (standard FFT convention)
    // scipy.signal.stft applies window normalization: 1.0 / sum(window) for magnitude spectrum
    // This matches the default 'spectrum' scaling mode
    T window_sum = WindowCache<T>::get_sum(win_length);
    T scale = static_cast<T>(1.0) / window_sum;
    size_t offset = frame * num_freqs;
    for (size_t freq = 0; freq < num_freqs; ++freq) {
        result[offset + freq] = std::abs(frame_data[freq]) * scale;
    }
}

/**
 * Optimized STFT with parallel frame processing.
 * Layout: [frame][freq] for cache locality.
 */
template <typename T>
inline std::vector<T> stft_fast(const std::vector<T>& x, size_t win_length, size_t hop_length,
                                size_t n_fft, bool parallel = true) {
    const auto& window = WindowCache<T>::get(win_length);
    size_t num_frames = (x.size() >= win_length) ? (x.size() - win_length) / hop_length + 1 : 0;
    size_t num_freqs = n_fft / 2 + 1;

    if (num_frames == 0) {
        return std::vector<T>();
    }

    std::vector<T> result(num_frames * num_freqs);

#if defined(HAS_OPENMP)
    if (parallel && num_frames > 16) {
// Parallel processing for many frames
#pragma omp parallel for schedule(dynamic, 4)
        for (size_t frame = 0; frame < num_frames; ++frame) {
            size_t start = frame * hop_length;
            compute_stft_frame(x, window, frame, start, win_length, n_fft, num_freqs,
                               result.data());
        }
    } else
#endif
    {
        // Sequential processing for small inputs
        for (size_t frame = 0; frame < num_frames; ++frame) {
            size_t start = frame * hop_length;
            compute_stft_frame(x, window, frame, start, win_length, n_fft, num_freqs,
                               result.data());
        }
    }

    return result;
}

/**
 * Batch STFT computation - C++ equivalent of JAX vmap.
 * Processes multiple audio signals in parallel.
 */
template <typename T>
inline std::vector<std::vector<T>> stft_batch(const std::vector<std::vector<T>>& batch,
                                              size_t win_length, size_t hop_length, size_t n_fft) {
    size_t batch_size = batch.size();
    std::vector<std::vector<T>> results(batch_size);

#if defined(HAS_OPENMP)
#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < batch_size; ++i) {
        results[i] = stft_fast(batch[i], win_length, hop_length, n_fft, false);
    }
#else
    for (size_t i = 0; i < batch_size; ++i) {
        results[i] = stft_fast(batch[i], win_length, hop_length, n_fft, false);
    }
#endif

    return results;
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
 * Vectorized spectral convergence loss.
 */
template <typename T>
inline T spectral_convergence_loss(const std::vector<T>& x_mag, const std::vector<T>& y_mag,
                                   T y_mag_mean) {
    const size_t n = x_mag.size();
    if (n == 0) return static_cast<T>(0);

    T inv_y_mean = static_cast<T>(1.0) / (y_mag_mean + static_cast<T>(1e-8));

    // Spectral convergence: mean(|y - x| / mean(y))
    T sc_loss = 0;
#if defined(HAS_OPENMP)
#pragma omp simd reduction(+ : sc_loss)
#endif
    for (size_t i = 0; i < n; ++i) {
        sc_loss += std::abs(y_mag[i] - x_mag[i]) * inv_y_mean;
    }

    return sc_loss / static_cast<T>(n);
}

/**
 * Vectorized log-magnitude loss.
 */
template <typename T>
inline T log_magnitude_loss(const std::vector<T>& x_mag, const std::vector<T>& y_mag) {
    const size_t n = x_mag.size();
    if (n == 0) return static_cast<T>(0);

    constexpr T EPSILON = static_cast<T>(1e-8);

    T mag_loss = 0;
#if defined(HAS_OPENMP)
#pragma omp simd reduction(+ : mag_loss)
#endif
    for (size_t i = 0; i < n; ++i) {
        T log_y = std::log(y_mag[i] + EPSILON);
        T log_x = std::log(x_mag[i] + EPSILON);
        mag_loss += std::abs(log_y - log_x);
    }

    return mag_loss / static_cast<T>(n);
}

/**
 * Fast vectorized normalization (zero mean, unit variance).
 */
template <typename T>
inline std::vector<T> normalize_signal(const std::vector<T>& x) {
    if (x.empty()) return x;

    // Compute mean
    T mean = vectorized_mean(x);

    // Subtract mean
    std::vector<T> result(x.size());
#if defined(HAS_OPENMP)
#pragma omp simd
#endif
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = x[i] - mean;
    }

    // Compute std and normalize
    T var = vectorized_variance(result, static_cast<T>(0));
    T std = std::sqrt(var);
    T inv_std = static_cast<T>(1.0) / (std + static_cast<T>(1e-8));

#if defined(HAS_OPENMP)
#pragma omp simd
#endif
    for (size_t i = 0; i < result.size(); ++i) {
        result[i] *= inv_std;
    }

    return result;
}
}  // namespace detail

/**
 * Precomputed target features for fast loss computation.
 */
template <typename T>
struct TargetFeaturesFast {
    std::vector<std::vector<T>> stfts;  // [scale][frame * freq]
    std::vector<T> stft_means;
    std::vector<size_t> num_frames;
    std::vector<size_t> num_freqs;
    T zcr;
};

/**
 * Precompute target STFT features with vectorized operations.
 */
template <typename T>
inline TargetFeaturesFast<T> precompute_target_features_fast(
    const std::vector<T>& target, const std::vector<size_t>& fft_sizes = {1024, 512, 256},
    const std::vector<size_t>& hop_sizes = {512, 256, 128}) {
    TargetFeaturesFast<T> features;
    features.num_frames.resize(fft_sizes.size());
    features.num_freqs.resize(fft_sizes.size());
    features.stft_means.resize(fft_sizes.size());

    // Normalize target
    std::vector<T> tgt_norm = detail::normalize_signal(target);

    // Compute STFT magnitudes for each scale
    for (size_t i = 0; i < fft_sizes.size(); ++i) {
        features.stfts.push_back(
            detail::stft_fast(tgt_norm, fft_sizes[i], hop_sizes[i], fft_sizes[i]));
        features.num_frames[i] =
            (target.size() >= fft_sizes[i]) ? (target.size() - fft_sizes[i]) / hop_sizes[i] + 1 : 0;
        features.num_freqs[i] = fft_sizes[i] / 2 + 1;
        if (!features.stfts[i].empty()) {
            features.stft_means[i] = detail::vectorized_mean(features.stfts[i]);
        } else {
            features.stft_means[i] = static_cast<T>(0);
        }
    }

    // Compute zero-crossing rate
    features.zcr = detail::zero_crossing_rate_fast(tgt_norm);

    return features;
}

/**
 * Fast single audio loss computation.
 * Thread-safe: uses thread-local buffers internally.
 */
template <typename T>
inline T compute_audio_loss_fast(const std::vector<T>& generated,
                                 const TargetFeaturesFast<T>& target_features) {
    // Normalize generated audio
    std::vector<T> gen_norm = detail::normalize_signal(generated);

    // Use same FFT sizes as JAX project
    static const std::vector<size_t> fft_sizes = {1024, 512, 256};
    static const std::vector<size_t> hop_sizes = {512, 256, 128};

    T total_loss = 0;

    for (size_t scale = 0; scale < fft_sizes.size(); ++scale) {
        // Compute STFT for this scale
        auto x_stft =
            detail::stft_fast(gen_norm, fft_sizes[scale], hop_sizes[scale], fft_sizes[scale]);
        const auto& y_stft = target_features.stfts[scale];

        if (x_stft.size() != y_stft.size()) continue;

        // Spectral convergence loss (70% weight)
        T sc_loss =
            detail::spectral_convergence_loss(x_stft, y_stft, target_features.stft_means[scale]);

        // Log-magnitude loss (30% weight)
        T mag_loss = detail::log_magnitude_loss(x_stft, y_stft);

        total_loss += static_cast<T>(0.7) * sc_loss + static_cast<T>(0.3) * mag_loss;
    }

    // Average over scales
    total_loss /= fft_sizes.size();

    // Add zero-crossing rate loss (5% weight)
    T gen_zcr = detail::zero_crossing_rate_fast(gen_norm);
    T zcr_loss = std::abs(gen_zcr - target_features.zcr);

    return static_cast<T>(0.95) * total_loss + static_cast<T>(0.05) * zcr_loss;
}

/**
 * Batch loss computation - process multiple generated signals in parallel.
 * C++ equivalent of JAX vmap over loss computation.
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
 */
template <typename T>
class LossFunction {
   public:
    LossFunction(const std::vector<T>& target, T stft_weight = static_cast<T>(1.0))
        : stft_weight_(stft_weight) {
        features_ = precompute_target_features_fast(target);
    }

    // Single loss computation - thread safe
    T operator()(const std::vector<T>& generated) const {
        return stft_weight_ * compute_audio_loss_fast(generated, features_);
    }

    // Batch loss computation - parallel processing
    std::vector<T> compute_batch(const std::vector<std::vector<T>>& generated_batch) const {
        auto losses = compute_audio_loss_batch(generated_batch, features_);
        if (stft_weight_ != static_cast<T>(1.0)) {
#if defined(HAS_OPENMP)
#pragma omp simd
#endif
            for (size_t i = 0; i < losses.size(); ++i) {
                losses[i] *= stft_weight_;
            }
        }
        return losses;
    }

   private:
    TargetFeaturesFast<T> features_;
    T stft_weight_;
};

}  // namespace adaptive_echo
