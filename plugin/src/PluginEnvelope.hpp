#pragma once
#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

class EnvelopeGenerator {
  public:
    struct Segment {
        float initialValue;
        float finalValue;
        float curvature;
        unsigned int lengthSamples; // 0 for sustain

        Segment(float iv, float fv, float c, unsigned int l)
            : initialValue(iv), finalValue(fv), curvature(c), lengthSamples(l) {
        }

        float fx(unsigned int x) const {
            if (lengthSamples == 0)
                return finalValue; // sustain
            if (x > lengthSamples)
                return finalValue;
            float t = float(x) / float(lengthSamples);
            return (finalValue - initialValue) * std::pow(t, curvature) +
                   initialValue;
        }
    };

    std::vector<Segment> segments;
    EnvelopeGenerator() = default;

    float get_sample(int segment, unsigned int dt) const {
        if (segment < 0 || segment >= (int)segments.size())
            return 0.0f;
        float out = segments[segment].fx(dt);
        return out;
    }
};

class ADSREnvelope : public EnvelopeGenerator {
  public:
    ADSREnvelope()
        : ADSREnvelope(0.01f, 0.1f, 1.0f, 0.1f, 1.0f, 1.0f, 1.0f, 44100) {}

    ADSREnvelope(float attackSeconds, float decaySeconds, float sustainLevel,
                 float releaseSeconds, float attack_c, float decay_c,
                 float release_c, int sampleRate)
        : a(attackSeconds), d(decaySeconds), s(sustainLevel), r(releaseSeconds),
          ac(attack_c), dc(decay_c), rc(release_c), sr(sampleRate) {
        auto toSamples = [this](float seconds) -> unsigned int {
            if (seconds <= 0.0f)
                return 0u;
            return static_cast<unsigned int>(
                std::max(0.0f, seconds * float(sr)));
        };

        unsigned int aSamples = toSamples(a);
        unsigned int dSamples = toSamples(d);
        unsigned int rSamples = toSamples(r);

        segments.clear();
        segments.emplace_back(0.0f, 1.0f, ac, aSamples); // Attack
        segments.emplace_back(1.0f, s, dc, dSamples);    // Decay
        segments.emplace_back(s, s, 1.0f, 0u);           // Sustain
        segments.emplace_back(1.0, 0.0f, rc, rSamples);  // Release
    }

    float get_a() const { return a; }
    float get_d() const { return d; }
    float get_s() const { return s; }
    float get_r() const { return r; }
    float get_ac() const { return ac; }
    float get_dc() const { return dc; }
    float get_rc() const { return rc; }
    float get_sr() const { return sr; }

  private:
    float a, d, s, r;
    float ac, dc, rc;
    int sr;
};