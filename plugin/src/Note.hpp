#include "Envelope.hpp"

class Note {
  protected:
    int ds; // Samples since envelope segment started
    std::shared_ptr<ADSREnvelope> env;
    int active_segment;
    int next_segment_at;
    float env_sample;
    float release_level; // Level when release started
    bool expired;

  public:
    int num;

    Note(int _num, std::shared_ptr<ADSREnvelope> _env)
        : ds(0), env(_env), active_segment(0), next_segment_at(0),
          env_sample(0.0f), release_level(1.0f), expired(false), num(_num) {
        next_segment_at = env->segments[0].lengthSamples;
    }

    Note(int _num) : Note(_num, std::make_shared<ADSREnvelope>()) {}
    Note() : Note(0) {}
    Note(const Note &other) = default;

    bool is_expired() const { return expired; }

    void set_env(std::shared_ptr<ADSREnvelope> _env) { env = _env; }

    void start_release() {
        active_segment = env->segments.size() - 1;
        ds = 0;
        next_segment_at = 0;
        expired = true;
        release_level = env_sample;
    }

    float update_env() {
        ds++;
        if (next_segment_at != 0 && ds > next_segment_at) {
            ds = 0;
            active_segment++;
            next_segment_at = env->segments[active_segment].lengthSamples;
        }
        env_sample = env->get_sample(active_segment, ds) * release_level;
        return env_sample;
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
};