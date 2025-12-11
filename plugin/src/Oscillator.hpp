#pragma once

#include <JuceHeader.h>

class WavetableOscillator {
public:
    WavetableOscillator(float warmth, float harshness)
        : warmth(warmth), harshness(harshness) {}

    WavetableOscillator() = default;

    void updateParameters(float warmth, float harshness) {
        this->harshness = harshness;
        this->warmth = warmth;
    }

    float sample(float ph) noexcept {
        float ph01 = ph * (1.0f / (2.0f * M_PI)); // Convert from radians to 0-1
        float x = ph01 - floor(ph01);
        float p = ((std::pow(x, warmth)-std::pow(1-x, warmth))/2);
        float f = std::sin(p * 2 * M_PI);
        return std::pow(std::abs(f), harshness) * sign(f);
    }

private:
    float warmth;
    float harshness;

    // Returns -1 for negative values, 0 for 0, and 1 for positive values
    int sign(float x) {
        return (x > 0.0) - (x < 0.0);
    }

    float modf(float a, float b) {
        return a-((int)(a/b))*b;
    }
};