#include <JuceHeader.h>

class WavetableOscillator {
public:
    WavetableOscillator(float warmth, float harshness)
        : warmth(warmth), harshness(harshness) {}

    WavetableOscillator() = default;

    float sample(float ph) noexcept {
        float x = modf(ph, 1.0);
        float p = ((std::pow(x, warmth)-std::pow(1-x, warmth))/2);
        float f = std::sin(p);
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