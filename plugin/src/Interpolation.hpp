#pragma once

#include <autodiff/reverse/var.hpp>
#include <cmath>
using namespace std;

inline autodiff::var linear_interp(const autodiff::var &a,
                                   const autodiff::var &b,
                                   const autodiff::var &t) {
    return a + (b - a) * t;
}

inline autodiff::var exp_interp(const autodiff::var &a, const autodiff::var &b,
                                const autodiff::var &t) {
    autodiff::var ratio = b / a;
    return a * pow(ratio, t);
}

inline autodiff::var sigmoid(const autodiff::var &x) {
    if (x < -5.0) {
        return 0.01;
    }
    if (x > 5.0) {
        return 0.99;
    }
    return 0.01 + 0.98 / (1.0 + exp(-x));
}