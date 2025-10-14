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
    return a * std::pow(b / a, t);
}