#pragma once

#include <vector>

using namespace std;

vector<int32_t> normalize(const vector<double> &x) {
    double headroom = 0.95;
    double maxValue = double(INT32_MAX) * headroom;
    double scale = maxValue * headroom;
    double largestAbs = 0;
    for (double x : x) {
        largestAbs = max(largestAbs, abs(x));
    }
    vector<int32_t> normalized(x.size());
    for (size_t i = 0; i < x.size(); i++) {
        normalized[i] = int32_t(x[i] / largestAbs * scale);
    }
    return normalized;
}
