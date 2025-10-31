#include <autodiff/reverse/var.hpp>
#include <cmath>
#include <stdexcept>
#include <vector>

std::vector<autodiff::var> normalize(std::vector<autodiff::var> vec) {
    double total = 0;
    for (autodiff::var elem : vec) {
        total += pow((double)elem, 2);
    }
    autodiff::var norm = sqrt(total);
    for (autodiff::var elem : vec) {
        elem /= total;
    }
    return vec;
}

autodiff::var dotprod(const std::vector<autodiff::var> &sounds,
                      const std::vector<autodiff::var> &settings) {
    if (sounds.size() != settings.size()) {
        throw std::runtime_error("Input sizes do not match");
    }

    autodiff::var sum = 0;
    const int N = sounds.size();
    for (int i = 0; i < N; i++) {
        sum += sounds[i] * settings[i];
    }
    return sum;
}

autodiff::var forward(const autodiff::var &tau,
                      const std::vector<std::vector<autodiff::var>> &sounds,
                      const std::vector<std::vector<autodiff::var>> &settings) {
    if (sounds.size() != settings.size()) {
        throw std::runtime_error("Input sizes do not match");
    }

    const int N = sounds.size();

    std::vector S(N, std::vector<autodiff::var>(N));

    autodiff::var sound_to_settings = 0;
    autodiff::var settings_to_sound = 0;

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            S[i][j] = dotprod(normalize(sounds[i]), normalize(settings[j]));
        }
    }

    for (int i = 0; i < N; i++) {
        autodiff::var sndtemp = 0;
        autodiff::var settemp = 0;

        for (int j = 0; j < N; j++) {
            sndtemp += exp(S[i][j] / tau);
            settemp += exp(S[j][i] / tau);
        }

        sound_to_settings += log(exp(S[i][i] / tau) / sndtemp);
        settings_to_sound += log(exp(S[i][i] / tau) / settemp);
    }

    sound_to_settings /= -N;
    settings_to_sound /= -N;

    autodiff::var total_loss = (sound_to_settings + settings_to_sound) / 2;
    return total_loss;
}
