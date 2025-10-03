#pragma once

#include "PluginProcessor.hpp"
#include <JuceHeader.h>

class AdaptiveEchoAudioProcessorEditor : public juce::AudioProcessorEditor {
  public:
    explicit AdaptiveEchoAudioProcessorEditor(AdaptiveEchoAudioProcessor &);
    ~AdaptiveEchoAudioProcessorEditor() override = default;

    void paint(juce::Graphics &g) override;
    void resized() override;

  private:
    AdaptiveEchoAudioProcessor &processor;

    juce::Slider volumeSlider;
    juce::Label volumeLabel;

    juce::MidiKeyboardComponent midiKeyboard{
        processor.getMidiKeyboardState(),
        juce::MidiKeyboardComponent::horizontalKeyboard};

    using SliderAttachment =
        juce::AudioProcessorValueTreeState::SliderAttachment;
    std::unique_ptr<SliderAttachment> volumeAttachment;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(
        AdaptiveEchoAudioProcessorEditor)
};