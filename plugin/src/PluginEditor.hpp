#pragma once

#include "PluginProcessor.hpp"
#include <juce_gui_extra/juce_gui_extra.h>

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
    juce::Slider freqSlider;
    juce::Label freqLabel;

    using SliderAttachment =
        juce::AudioProcessorValueTreeState::SliderAttachment;
    std::unique_ptr<SliderAttachment> volumeAttachment;
    std::unique_ptr<SliderAttachment> freqAttachment;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(
        AdaptiveEchoAudioProcessorEditor)
};
