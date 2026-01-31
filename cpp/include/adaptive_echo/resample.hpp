#pragma once

/**
 * FFT-based resampling using pocketfft (matches scipy.signal.resample).
 */

#include <pocketfft_hdronly.h>

#include <algorithm>
#include <complex>
#include <cstddef>
#include <vector>

namespace adaptive_echo {

template <typename T>
inline std::vector<T> resample_fft(const std::vector<T>& input, std::size_t num) {
    const std::size_t n = input.size();
    if (n == 0 || num == 0 || num == n) {
        return input;
    }

    using complex_t = std::complex<T>;

    // Forward FFT (complex)
    std::vector<complex_t> x_time(n);
    for (std::size_t i = 0; i < n; ++i) {
        x_time[i] = complex_t(input[i], static_cast<T>(0));
    }

    std::vector<complex_t> x_freq(n);
    pocketfft::shape_t shape_in = {n};
    pocketfft::stride_t stride_in = {sizeof(complex_t)};
    pocketfft::stride_t stride_out = {sizeof(complex_t)};
    pocketfft::shape_t axes = {0};

    pocketfft::c2c<T>(shape_in, stride_in, stride_out, axes, pocketfft::FORWARD, x_time.data(),
                      x_freq.data(), static_cast<T>(1.0), 1);

    // Frequency-domain truncation/zero-padding
    std::vector<complex_t> y_freq(num, complex_t(0, 0));
    const std::size_t k = std::min(n, num);
    const std::size_t half = (k + 1) / 2;  // includes DC and Nyquist if present
    const std::size_t neg = k / 2;

    for (std::size_t i = 0; i < half; ++i) {
        y_freq[i] = x_freq[i];
    }
    for (std::size_t i = 0; i < neg; ++i) {
        y_freq[num - neg + i] = x_freq[n - neg + i];
    }

    // Inverse FFT to time domain
    std::vector<complex_t> y_time(num);
    pocketfft::shape_t shape_out = {num};
    pocketfft::c2c<T>(shape_out, stride_out, stride_out, axes, pocketfft::BACKWARD, y_freq.data(),
                      y_time.data(), static_cast<T>(1.0) / static_cast<T>(num), 1);

    // Match scipy.signal.resample scaling
    const T scale = static_cast<T>(num) / static_cast<T>(n);
    std::vector<T> output(num);
    for (std::size_t i = 0; i < num; ++i) {
        output[i] = static_cast<T>(y_time[i].real()) * scale;
    }

    return output;
}

}  // namespace adaptive_echo
