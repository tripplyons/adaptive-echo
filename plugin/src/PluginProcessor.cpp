#include "PluginProcessor.hpp"
#include "PluginEditor.hpp"

AdaptiveEchoAudioProcessor::AdaptiveEchoAudioProcessor()
    : AudioProcessor(BusesProperties().withOutput(
          "Output", juce::AudioChannelSet::stereo(), true)),
      apvts(*this, nullptr, "PARAMS", createParameterLayout()) {
  volumeSmoothed.reset(currentSampleRate, 0.02); // 20ms smoothing
}

juce::AudioProcessorValueTreeState::ParameterLayout
AdaptiveEchoAudioProcessor::createParameterLayout() {
  std::vector<std::unique_ptr<juce::RangedAudioParameter>> params;

  params.push_back(std::make_unique<juce::AudioParameterFloat>(
      "volume", "Volume",
      juce::NormalisableRange<float>(0.0f, 1.0f, 0.0f, 1.0f), 0.5f));
  params.push_back(std::make_unique<juce::AudioParameterFloat>(
      "freq", "Frequency",
      juce::NormalisableRange<float>(20.0f, 2000.0f, 0.01f, 0.3f), 440.0f));

  return {params.begin(), params.end()};
}

void AdaptiveEchoAudioProcessor::prepareToPlay(double sampleRate,
                                               int /*samplesPerBlock*/) {
  currentSampleRate = sampleRate;

  auto *freqParam = apvts.getRawParameterValue("freq");
  const double frequency = freqParam != nullptr ? freqParam->load() : 440.0;
  phaseInc = juce::MathConstants<double>::twoPi * frequency / currentSampleRate;

  phase = {0.0, 0.0};

  volumeSmoothed.reset(currentSampleRate, 0.02);
  auto *volParam = apvts.getRawParameterValue("volume");
  volumeSmoothed.setCurrentAndTargetValue(volParam != nullptr ? volParam->load()
                                                              : 0.5f);
}

void AdaptiveEchoAudioProcessor::releaseResources() {}

#ifndef JucePlugin_PreferredChannelConfigurations
bool AdaptiveEchoAudioProcessor::isBusesLayoutSupported(
    const BusesLayout &layouts) const {
  // Only allow mono or stereo outputs
  const auto &mainOut = layouts.getMainOutputChannelSet();
  return mainOut == juce::AudioChannelSet::mono() ||
         mainOut == juce::AudioChannelSet::stereo();
}
#endif

void AdaptiveEchoAudioProcessor::processBlock(juce::AudioBuffer<float> &buffer,
                                              juce::MidiBuffer &midi) {
  juce::ignoreUnused(midi);

  juce::ScopedNoDenormals noDenormals;
  const int numSamples = buffer.getNumSamples();
  const int numChans =
      juce::jmin(2, buffer.getNumChannels()); // we track phase for 2 chans

  auto *volParam = apvts.getRawParameterValue("volume");
  if (volParam != nullptr)
    volumeSmoothed.setTargetValue(volParam->load());

  auto *freqParam = apvts.getRawParameterValue("freq");
  if (freqParam != nullptr) {
    const double frequency = freqParam->load();
    phaseInc = juce::MathConstants<double>::twoPi * frequency / currentSampleRate;
  }

  for (int ch = 0; ch < numChans; ++ch) {
    float *out = buffer.getWritePointer(ch);
    double ph = phase[(size_t)ch];

    for (int n = 0; n < numSamples; ++n) {
      const float amp = volumeSmoothed.getNextValue();
      const float s = (float)std::sin(ph);
      out[n] = s * amp;

      ph += phaseInc;
      if (ph >= juce::MathConstants<double>::twoPi)
        ph -= juce::MathConstants<double>::twoPi;
    }

    phase[(size_t)ch] = ph;
  }

  // Clear any extra channels (e.g., if host created more)
  for (int ch = numChans; ch < buffer.getNumChannels(); ++ch)
    buffer.clear(ch, 0, numSamples);
}

void AdaptiveEchoAudioProcessor::getStateInformation(
    juce::MemoryBlock &destData) {
  if (auto state = apvts.copyState(); state.isValid()) {
    juce::MemoryOutputStream mos(destData, true);
    state.writeToStream(mos);
  }
}

void AdaptiveEchoAudioProcessor::setStateInformation(const void *data,
                                                     int sizeInBytes) {
  juce::ValueTree tree =
      juce::ValueTree::readFromData(data, (size_t)sizeInBytes);
  if (tree.isValid())
    apvts.replaceState(tree);

  // Nudge smoothed value to loaded state
  if (auto *volParam = apvts.getRawParameterValue("volume"))
    volumeSmoothed.setCurrentAndTargetValue(volParam->load());
}

juce::AudioProcessorEditor *AdaptiveEchoAudioProcessor::createEditor() {
  return new AdaptiveEchoAudioProcessorEditor(*this);
}

// This factory must be present in the TU with the processor class.
juce::AudioProcessor *JUCE_CALLTYPE createPluginFilter() {
  return new AdaptiveEchoAudioProcessor();
}
