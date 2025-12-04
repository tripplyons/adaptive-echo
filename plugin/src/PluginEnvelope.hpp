#pragma once
#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

/*
  Revised Envelope / ADSR implementation.

  Key behavior changes to address the jump/drop problem:

  - The release segment initialValue is computed at the exact moment of note-off
    (i.e. at the sample index where release begins). This handles note-offs that
    occur during attack or decay (when the envelope hasn't reached sustain).
  - A small, clear API is provided: trigger() to (re)start the envelope and
    noteOff(samplesSinceNoteOn) to begin the release based on how many samples
    have elapsed since the trigger.
  - Segment.fx() is robust to boundary conditions and uses [0..1] t mapping.
*/

class EnvelopeGenerator {
  public:
    struct Segment {
        float initialValue;
        float finalValue;
        float curvature;         // exponent: >0.0 (1.0 = linear, >1 convex, <1 concave)
        unsigned int lengthSamples; // 0 for sustain (infinite)

        Segment(float iv = 0.0f, float fv = 0.0f, float c = 1.0f, unsigned int l = 0u)
            : initialValue(iv), finalValue(fv), curvature(c), lengthSamples(l) {}

        // Evaluate the segment at sample index x where 0 <= x <= lengthSamples
        // If lengthSamples == 0 we treat it as sustain and return finalValue (stable)
        float fx(unsigned int x) const {
            if (lengthSamples == 0u)
                return finalValue; // sustain: stable value

            // clamp x to [0, lengthSamples]
            if (x >= lengthSamples)
                return finalValue;
            if (x == 0u)
                return initialValue;

            // normalized position in segment [0..1]
            float t = float(x) / float(lengthSamples);
            // protect curvature
            float c = curvature;
            if (c <= 0.0f)
                c = 1.0f;

            // compute eased value
            float v = std::pow(t, c);
            return (finalValue - initialValue) * v + initialValue;
        }
    };

    std::vector<Segment> segments;

    EnvelopeGenerator() = default;

    // get_sample: segment = index into segments vector (0..n-1),
    // dt = sample offset from start of that segment
    float get_sample(int segment, unsigned int dt) const {
        if (segment < 0 || segment >= (int)segments.size())
            return 0.0f;
        return segments[segment].fx(dt);
    }

    // Utility: evaluate envelope value at a given absolute sample offset since trigger,
    // ignoring any release segment. This is used to compute the correct release start value.
    // The mapping assumes segments are laid out sequentially in segments[] order.
    float value_at_samples_since_trigger(unsigned int samplesSinceTrigger) const {
        unsigned int cursor = 0u;

        for (size_t i = 0; i < segments.size(); ++i) {
            const Segment& seg = segments[i];
            if (seg.lengthSamples == 0u) {
                // sustain: if we reached sustain, value remains at seg.finalValue
                return seg.finalValue;
            }

            if (samplesSinceTrigger <= cursor + seg.lengthSamples) {
                unsigned int localDt = samplesSinceTrigger > cursor ? samplesSinceTrigger - cursor : 0u;
                // clamp localDt to seg.lengthSamples
                if (localDt >= seg.lengthSamples)
                    localDt = seg.lengthSamples;
                return seg.fx(localDt);
            }

            cursor += seg.lengthSamples;
        }

        // If after all defined segments, return final value of last segment
        if (!segments.empty())
            return segments.back().finalValue;

        return 0.0f;
    }
};

class ADSREnvelope : public EnvelopeGenerator {
  public:
    ADSREnvelope()
        : ADSREnvelope(0.01f, 0.1f, 1.0f, 0.1f, 1.0f, 1.0f, 1.0f, 44100) {}

    ADSREnvelope(float attackSeconds,
                 float decaySeconds,
                 float sustainLevel,
                 float releaseSeconds,
                 float attack_c,
                 float decay_c,
                 float release_c,
                 int sampleRate)
        : a(attackSeconds), d(decaySeconds), s(sustainLevel), r(releaseSeconds),
          ac(attack_c), dc(decay_c), rc(release_c), sr(sampleRate),
          aSamples(0u), dSamples(0u), rSamples(0u), running(false), samplesSinceTrigger(0u)
    {
        rebuildSegments();
    }

    // (Re)initialize envelope segments based on current parameter settings.
    void rebuildSegments() {
        auto toSamples = [this](float seconds) -> unsigned int {
            if (seconds <= 0.0f)
                return 0u;
            double v = double(seconds) * double(sr);
            if (v < 0.0)
                v = 0.0;
            return static_cast<unsigned int>(std::max(0.0, std::floor(v + 0.5)));
        };

        aSamples = toSamples(a);
        dSamples = toSamples(d);
        rSamples = toSamples(r);

        segments.clear();
        segments.reserve(4);

        // Attack: from 0.0 -> 1.0 over aSamples
        segments.emplace_back(0.0f, 1.0f, ac <= 0.0f ? 1.0f : ac, aSamples);

        // Decay: from 1.0 -> sustain over dSamples
        segments.emplace_back(1.0f, s, dc <= 0.0f ? 1.0f : dc, dSamples);

        // Sustain: length 0 (infinite), value s
        segments.emplace_back(s, s, 1.0f, 0u);

        // Release: set up with an initial guess; actual initialValue is set during noteOff()
        // We set initialValue to s by default to be reasonable for long-held notes.
        segments.emplace_back(s, 0.0f, rc <= 0.0f ? 1.0f : rc, rSamples);

        // reset runtime state
        running = false;
        samplesSinceTrigger = 0u;
    }

    // Trigger (note on). Resets the runtime counter and marks envelope running.
    void trigger() {
        running = true;
        samplesSinceTrigger = 0u;
        // reset release's initial in case it was changed before
        if (segments.size() >= 4) {
            segments[3].initialValue = s;
            segments[3].finalValue = 0.0f;
            segments[3].curvature = rc <= 0.0f ? 1.0f : rc;
            segments[3].lengthSamples = rSamples;
        }
    }

    // Call this when note-off occurs. samplesSinceTrigger is the number of samples
    // that elapsed since trigger() was called (i.e. current time - trigger time).
    // This computes the envelope value at that instant and assigns it as the release's initial.
    void noteOff(unsigned int samplesSinceTrigger_) {
        // Compute the current value at the moment of note-off using the attack/decay/sustain segments
        float valueAtRelease = value_at_samples_since_trigger(samplesSinceTrigger_);

        // Ensure segments vector has a release segment (index 3)
        if (segments.size() < 4) {
            // rebuild to ensure layout then set value
            rebuildSegments();
        }

        // Set release initial value to the precise current value so release starts smoothly
        segments[3].initialValue = valueAtRelease;
        segments[3].finalValue = 0.0f;
        segments[3].curvature = rc <= 0.0f ? 1.0f : rc;
        segments[3].lengthSamples = rSamples;

        // Mark envelope as no longer in "held" state; subsequent sampling should use release segment timing
        // To help consumers, we set a running flag and keep samplesSinceTrigger for reference if needed.
        running = false;
        samplesSinceTrigger = samplesSinceTrigger_;
    }

    // Sample the envelope given a segment index and sample offset inside that segment.
    // Common segment indices: 0=attack,1=decay,2=sustain,3=release.
    // For convenience, a helper that advances sample counters and returns the current value in typical usage:
    // - If the envelope is in the "held" portion (attack/decay/sustain), call advance() repeatedly until noteOff.
    // - After noteOff(), call get_release_sample(releaseDt) using dt since release start.
    float get_sample(int segment, unsigned int dt) const {
        return EnvelopeGenerator::get_sample(segment, dt);
    }

    // Convenience helper: compute value at absolute samples-since-trigger for the full envelope,
    // taking into account a release that may have been set via noteOff(). If release has been set,
    // and the requested time is after the release start, it will compute from the release segment.
    float value_at_absolute_sample(unsigned int absoluteSamplesSinceTrigger) const {
        // If release has been armed (we use segments[3].initialValue as indicator)
        // and absoluteSamplesSinceTrigger >= samplesSinceTrigger (the moment of noteOff),
        // then compute position inside release.
        if (segments.size() >= 4 && absoluteSamplesSinceTrigger >= samplesSinceTrigger && segments[3].lengthSamples > 0u) {
            unsigned int releaseDt = absoluteSamplesSinceTrigger - samplesSinceTrigger;
            // clamp releaseDt
            if (releaseDt >= segments[3].lengthSamples)
                return segments[3].finalValue;
            return segments[3].fx(releaseDt);
        }

        // otherwise, evaluate normally across attack/decay/sustain
        return value_at_samples_since_trigger(absoluteSamplesSinceTrigger);
    }

    // parameter accessors
    float get_a() const { return a; }
    float get_d() const { return d; }
    float get_s() const { return s; }
    float get_r() const { return r; }
    float get_ac() const { return ac; }
    float get_dc() const { return dc; }
    float get_rc() const { return rc; }
    int   get_sr() const { return sr; }

    // setters (if you change parameters at runtime, call rebuildSegments() to apply)
    void set_attack(float v) { a = v; }
    void set_decay(float v) { d = v; }
    void set_sustain(float v) { s = v; }
    void set_release(float v) { r = v; }
    void set_attack_curvature(float v) { ac = v; }
    void set_decay_curvature(float v) { dc = v; }
    void set_release_curvature(float v) { rc = v; }
    void set_sample_rate(int v) { sr = v; }

  private:
    float a, d, s, r;
    float ac, dc, rc;
    int sr;

    unsigned int aSamples, dSamples, rSamples;

    // runtime state
    bool running;
    unsigned int samplesSinceTrigger;
};
