#include <autodiff/reverse/var.hpp>
#include <iostream>
#include <vector>
#include <stdexcept>
#include <cmath>


autodiff::var cosine(std::vector<autodiff::var> sounds, std::vector<autodiff::var> settings){
  if (sounds.size() != settings.size()){
    throw std::runtime_error("Input sizes do not match");
  }

  autodiff::var sum = 0;
  int N = sounds.size();
  for (int i = 0;i<N;i++){
    for(int j=0;j<N;j++){
      sum+=sounds[i]*settings[j];
    }
  }
  return sum;
}

autodiff::var forward(autodiff::var tau, std::vector<std::vector<autodiff::var>> sounds, std::vector<std::vector<autodiff::var>> settings) {
  if (sounds.size() != settings.size()){
    throw std::runtime_error("Input sizes do not match");
  }

  int N = sounds.size();

  std::vector<std::vector<autodiff::var>> S(N, std::vector<autodiff::var>(N));

  autodiff::var sound_to_settings = 0;
  autodiff::var settings_to_sound = 0;

  for (int i = 0; i < N; i++){
    for (int j = 0; j < N; j++){
      S[i][j] = cosine(sounds[i],settings[j]);
    }
  }

  for (int i = 0; i < N; i++){
    autodiff::var sndtemp = 0;
    autodiff::var settemp = 0;

    for (int j = 0; j < N; j++){
      sndtemp += std::exp(S[i][j] / tau);
      settemp += std::exp(S[j][i] / tau);
    }

    sound_to_settings += std::log(std::exp(S[i][i] / tau)/sndtemp);
    settings_to_sound += std::log(std::exp(S[i][i]/tau)/settemp);
  }

  sound_to_settings /= -N;
  settings_to_sound /= -N;

  autodiff::var total_loss = (sound_to_settings+settings_to_sound)/2;
  return total_loss;
}
