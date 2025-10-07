#include <autodiff/reverse/var.hpp>
#include <iostream>

autodiff::var forward(autodiff::var x) { return 2.0 * x; }

double backward(double x_val) {
    autodiff::var x = x_val;
    autodiff::var y = forward(x);
    auto [grad_x] = autodiff::derivatives(y, autodiff::wrt(x));
    return grad_x;
}

int main() {
    std::cout << backward(1) << std::endl;
    return 0;
}
