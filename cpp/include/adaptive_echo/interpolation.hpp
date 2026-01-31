#pragma once

/**
 * Interpolation functions for adaptive_echo.
 */

#include <algorithm>
#include <cmath>
#include <vector>

namespace adaptive_echo {

/**
 * Linear interpolation between a and b using t as the interpolation factor.
 */
template <typename T>
inline T linear_interp(T a, T b, T t) {
    return a + (b - a) * t;
}

/**
 * Exponential interpolation between a and b using t as the interpolation factor.
 */
template <typename T>
inline T exp_interp(T a, T b, T t) {
    constexpr T EPSILON = static_cast<T>(1e-6);

    T a_clamped = std::max(std::abs(a), EPSILON) * (a >= 0 ? 1 : -1);
    T b_clamped = std::max(std::abs(b), EPSILON) * (b >= 0 ? 1 : -1);

    T ratio = b_clamped / a_clamped;
    ratio = std::clamp(ratio, EPSILON, static_cast<T>(1.0) / EPSILON);

    return a_clamped * std::pow(ratio, t);
}

/**
 * Linear interpolation for vectors.
 */
template <typename T>
inline std::vector<T> linear_interp(const std::vector<T>& a, const std::vector<T>& b,
                                    const std::vector<T>& t) {
    size_t n = std::min({a.size(), b.size(), t.size()});
    std::vector<T> result(n);
    for (size_t i = 0; i < n; ++i) {
        result[i] = linear_interp(a[i], b[i], t[i]);
    }
    return result;
}

/**
 * Scalar-vector linear interpolation.
 */
template <typename T>
inline std::vector<T> linear_interp(const std::vector<T>& a, const std::vector<T>& b, T t) {
    size_t n = std::min(a.size(), b.size());
    std::vector<T> result(n);
    for (size_t i = 0; i < n; ++i) {
        result[i] = linear_interp(a[i], b[i], t);
    }
    return result;
}

}  // namespace adaptive_echo
