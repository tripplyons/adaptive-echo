#ifndef INTERPOLATION_HPP
#define INTERPOLATION_HPP

#include <autodiff/reverse/var.hpp>
#include <cmath>

inline autodiff::var linear_interp(const autodiff::var &a, const autodiff::var &b, const autodiff::var &t){
    return a + (b - a) * t;
}

inline autodiff::var exp_interp(const autodiff::var &a, const autodiff::var &b, const autodiff::var &t){
    return a * pow(b / a, t);
}

#endif