#pragma once

#include "PluginProcessor.hpp"
#include "EnvelopeViewer.hpp"
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

    // ADSR
    juce::Slider attackSlider;
    juce::Slider decaySlider;
    juce::Slider sustainSlider;
    juce::Slider releaseSlider;

    juce::Slider attackCurveSlider;
    juce::Slider decayCurveSlider;
    juce::Slider sustainCurveSlider;
    juce::Slider releaseCurveSlider;

    juce::Label attackLabel;
    juce::Label decayLabel;
    juce::Label sustainLabel;
    juce::Label releaseLabel;

    juce::Label attackCurveLabel;
    juce::Label decayCurveLabel;
    juce::Label releaseCurveLabel;

    EnvelopeViewer envelopeViewer;

    juce::MidiKeyboardComponent midiKeyboard{
        processor.getMidiKeyboardState(),
        juce::MidiKeyboardComponent::horizontalKeyboard};

    using SliderAttachment =
        juce::AudioProcessorValueTreeState::SliderAttachment;
    std::unique_ptr<SliderAttachment> volumeAttachment;
    std::unique_ptr<SliderAttachment> attackAttachment;
    std::unique_ptr<SliderAttachment> decayAttachment;
    std::unique_ptr<SliderAttachment> sustainAttachment;
    std::unique_ptr<SliderAttachment> releaseAttachment;

    std::unique_ptr<SliderAttachment> attackCurveAttachment;
    std::unique_ptr<SliderAttachment> decayCurveAttachment;
    std::unique_ptr<SliderAttachment> releaseCurveAttachment;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(
        AdaptiveEchoAudioProcessorEditor)
};