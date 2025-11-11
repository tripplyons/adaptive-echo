#pragma once

#include <JuceHeader.h>
#include <array>
#include <vector>

#include "Note.hpp"
#include "PluginEnvelope.hpp"
#include <memory>

class AdaptiveEchoAudioProcessor : public juce::AudioProcessor {
  public:
    AdaptiveEchoAudioProcessor();
    ~AdaptiveEchoAudioProcessor() override = default;

    //==============================================================================
    void prepareToPlay(double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;
#ifndef JucePlugin_PreferredChannelConfigurations
    bool isBusesLayoutSupported(const BusesLayout &layouts) const override;
#endif
    void processBlock(juce::AudioBuffer<float> &, juce::MidiBuffer &) override;

    //==============================================================================
    juce::AudioProcessorEditor *createEditor() override;
    bool hasEditor() const override { return true; }

    //==============================================================================
    const juce::String getName() const override { return JucePlugin_Name; }
    bool acceptsMidi() const override { return true; }
    bool producesMidi() const override { return false; }
    bool isMidiEffect() const override { return false; }
    double getTailLengthSeconds() const override { return 0.0; }

    //==============================================================================
    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram(int) override {}
    const juce::String getProgramName(int) override { return {}; }
    void changeProgramName(int, const juce::String &) override {}

    //==============================================================================
    void getStateInformation(juce::MemoryBlock &destData) override;
    void setStateInformation(const void *data, int sizeInBytes) override;

    // Parameter/state
    static juce::AudioProcessorValueTreeState::ParameterLayout
    createParameterLayout();
    juce::AudioProcessorValueTreeState apvts;

    // Expose internal MidiKeyboardState
    juce::MidiKeyboardState &getMidiKeyboardState() noexcept {
        return midiState;
    }

  private:
    // Simple sine generator state per channel
    std::array<double, 2> phase{0.0, 0.0}; // support up to stereo
    double phaseInc = 0.0;                 // radians per sample
    double currentSampleRate = 48000.0;
    Note activeNote;
    ADSREnvelope env;
    std::shared_ptr<ADSREnvelope> env_ptr;

    // Smoothed volume to avoid zipper noise
    juce::SmoothedValue<float, juce::ValueSmoothingTypes::Linear>
        volumeSmoothed;
    juce::SmoothedValue<float, juce::ValueSmoothingTypes::Linear>
        noteAmpSmoothed;
    juce::MidiKeyboardState midiState;

    float a, d, s, r, ac, dc, rc;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(AdaptiveEchoAudioProcessor)
};
