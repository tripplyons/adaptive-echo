#include "PluginProcessor.hpp"
#include "PluginEditor.hpp"

AdaptiveEchoAudioProcessor::AdaptiveEchoAudioProcessor()
    : AudioProcessor(BusesProperties().withOutput(
          "Output", juce::AudioChannelSet::stereo(), true)),
      apvts(*this, nullptr, "PARAMS", createParameterLayout()) {
    volumeSmoothed.reset(currentSampleRate, 0.02); // 20ms smoothing
    noteAmpSmoothed.reset(currentSampleRate, 0.02);
}
juce::AudioProcessorValueTreeState::ParameterLayout
AdaptiveEchoAudioProcessor::createParameterLayout() {
    std::vector<std::unique_ptr<juce::RangedAudioParameter>> params;

    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        "volume", "Volume",
        juce::NormalisableRange<float>(0.0f, 1.0f, 0.0f, 1.0f), 0.5f));

    return {params.begin(), params.end()};
}

void AdaptiveEchoAudioProcessor::prepareToPlay(double sampleRate,
                                               int /*samplesPerBlock*/) {
    currentSampleRate = sampleRate;

    phase = {0.0, 0.0};

    volumeSmoothed.reset(currentSampleRate, 0.02);
    noteAmpSmoothed.reset(currentSampleRate, 0.02);

    auto *volParam = apvts.getRawParameterValue("volume");
    volumeSmoothed.setCurrentAndTargetValue(
        volParam != nullptr ? volParam->load() : 0.5f);
    noteAmpSmoothed.setCurrentAndTargetValue(0.0f); // Start silent
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
    juce::ScopedNoDenormals noDenormals;
    const int numSamples = buffer.getNumSamples();
    const int numChans =
        juce::jmin(2, buffer.getNumChannels()); // we track phase for 2 chans

    auto *volParam = apvts.getRawParameterValue("volume");
    const float baseVolume = volParam != nullptr ? volParam->load() : 0.5f;
    volumeSmoothed.setTargetValue(baseVolume);

    // Inject on-screen keyboard events into MIDI IN buffer
    midiState.processNextMidiBuffer(midi, 0, numSamples, true);

    // Parse MIDI for note on/off events
    for (const auto metadata : midi) {
        const auto msg = metadata.getMessage();
        if (msg.isNoteOn()) {
            activeNote = msg.getNoteNumber();
            uint8_t vel = (uint8_t)msg.getVelocity();

            // Compute frequency
            const double frequency =
                juce::MidiMessage::getMidiNoteInHertz(activeNote);
            phaseInc = juce::MathConstants<double>::twoPi * frequency /
                       currentSampleRate;

            // Set amplitude target (baseVolume * velocity)
            const float targetAmp = baseVolume * (float(vel) / 127.0f);
            noteAmpSmoothed.setTargetValue(targetAmp);
        } else {
            noteAmpSmoothed.setTargetValue(0.0f);
        }
    }

    for (int ch = 0; ch < numChans; ++ch) {
        float *out = buffer.getWritePointer(ch);
        double ph = phase[(size_t)ch];

        for (int n = 0; n < numSamples; ++n) {
            const float amp = noteAmpSmoothed.getNextValue();
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
