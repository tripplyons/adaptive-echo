#include <autodiff/reverse/var.hpp>
#include <iostream>

using namespace autodiff;

var forward(var x) { return 2 * x; }

double backward(double x_val) {
    var x = x_val;
    var y = forward(x);
    auto [grad_x] = derivatives(y, wrt(x));
    return grad_x;
}

int main() {
    std::cout << backward(1) << std::endl;
    return 0;
}
