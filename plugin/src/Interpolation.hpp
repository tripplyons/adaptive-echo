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
    if (x < -10.0) {
        return 0.001;
    }
    if (x > 10.0) {
        return 0.999;
    }
    return 0.001 + 0.998 / (1.0 + exp(-x));
}