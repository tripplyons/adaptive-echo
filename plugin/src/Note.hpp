#include "PluginEnvelope.hpp"

class Note {
  protected:
    int ds; // Samples since envelope segment started
    std::shared_ptr<ADSREnvelope> env;
    int currentSegment;
    int nextSegmentAt;
    float envSample;
    float releaseLevel; // Level when release started
    bool expired;
    bool releaseStarted;

  public:
    int num;

    Note(int _num, std::shared_ptr<ADSREnvelope> _env)
        : ds(0), env(_env), currentSegment(0), nextSegmentAt(0),
          envSample(0.0f), releaseLevel(1.0f), expired(false), num(_num) {
        nextSegmentAt = env->segments[0].lengthSamples;
    }

    Note(int _num) : Note(_num, std::make_shared<ADSREnvelope>()) {}
    Note() : Note(0) { expired = true; }
    Note(const Note &other) = default;

    bool is_expired() const { return expired; }

    void set_env(std::shared_ptr<ADSREnvelope> _env) { env = _env; }

    void start_release() {
        if (env) currentSegment = env->segments.size() - 1;
        else currentSegment = 0;
        ds = 0;
        nextSegmentAt = 0;
        expired = false;
        releaseStarted = true;
        releaseLevel = envSample;
    }

    void reset() {
        currentSegment = 0;
        ds = 0;
        expired = false;
        releaseLevel = 1.0;
    }

    float update_env() {
        ds++;
        if (nextSegmentAt != 0 && ds > nextSegmentAt) {
            ds = 0;
            currentSegment++;
            nextSegmentAt = env->segments[currentSegment].lengthSamples;
        }
        envSample = env->get_sample(currentSegment, ds) * releaseLevel;
        return envSample;
    }

    // Peek at the envelope value at a specific segment, ds samples into the
    // segment
    float peek_env(int segment, int ds) const {
        return env->get_sample(segment, ds);
    }

    // Peek at the envelope value in any segment (except release) at sample dt
    float peek_env(int dt) const {
        if (dt < env->get_a() * env->get_sr())
            return env->get_sample(0, dt);
        else if (dt < env->get_d() * env->get_sr())
            return env->get_sample(1, dt - env->get_a());
        else
            return env->get_sample(2, 0);
    }

    void applyEnvelopeToBuffer(juce::AudioBuffer<float>& buffer,
                               int startSample,
                               int numSamples) {
        if (expired || env->segments.size() == 0)
            return;

        const int numChannels = buffer.getNumChannels();

        for (int i = 0; i < numSamples; ++i) {
            if (currentSegment >= (int)env->segments.size()) {
                expired = true;
                for (int ch = 0; ch < numChannels; ++ch)
                    buffer.getWritePointer(ch)[startSample + i] = 0.0;
            }

            envSample = env->segments[currentSegment].fx(ds);
            if (releaseStarted) envSample *= releaseLevel;

            for (int ch = 0; ch < numChannels; ++ch)
                buffer.getWritePointer(ch)[startSample + i] *= envSample;

            ds++;
            if (env->segments[currentSegment].lengthSamples > 0 && ds >= env->segments[currentSegment].lengthSamples) {
                currentSegment++;
                ds = 0;
            }
        }
    }
};