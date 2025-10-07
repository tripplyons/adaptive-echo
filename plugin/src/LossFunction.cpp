#include <cmath>
#include <stdexcept>
#include <vector>

typedef unsigned short ushort;

double cosine(const std::vector<double> sounds,
              const std::vector<double> settings) {
    if (sounds.size() != settings.size()) {
        throw std::runtime_error("Input sizes do not match!");
    }

    double sum = 0;
    ushort N = sounds.size();
    for (ushort i = 0; i < N; i++) {
        for (ushort j = 0; j < N; j++) {
            sum += sounds[i] * settings[j];
        }
    }
    return sum;
}

double CLIP_Loss(double tau, std::vector<std::vector<double>> sounds,
                 std::vector<std::vector<double>> settings) {

    if (sounds.size() != settings.size()) {
        throw std::runtime_error("Input sizes do not match!");
    }

    ushort N = sounds.size();

    std::vector<std::vector<double>> S(N, std::vector<double>(N));

    double sound_to_settings = 0;
    double settings_to_sound = 0;

    for (ushort i = 0; i < N; i++) {
        for (ushort j = 0; j < N; j++) {
            S[i][j] = cosine(sounds[i], settings[j]);
        }
    }

    for (ushort i = 0; i < N; i++) {
        double sndtemp = 0;
        double settemp = 0;

        for (ushort j = 0; j < N; j++) {
            sndtemp += std::exp(S[i][j] / tau);
            settemp += std::exp(S[j][i] / tau);
        }

        sound_to_settings += std::log(std::exp(S[i][i] / tau) / sndtemp);
        settings_to_sound += std::log(std::exp(S[i][i] / tau) / settemp);
    }

    sound_to_settings /= -N;
    settings_to_sound /= -N;

    double total_loss = (sound_to_settings + settings_to_sound) / 2;
    return total_loss;
}
