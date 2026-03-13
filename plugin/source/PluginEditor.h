#pragma once

#include <memory>

#include <juce_gui_extra/juce_gui_extra.h>

#include "PluginProcessor.h"

class AdaptiveEchoAudioProcessorEditor final : public juce::AudioProcessorEditor,
                                               private juce::Button::Listener,
                                               private juce::ChangeListener {
public:
    explicit AdaptiveEchoAudioProcessorEditor(AdaptiveEchoAudioProcessor&);
    ~AdaptiveEchoAudioProcessorEditor() override;

    void paint(juce::Graphics&) override;
    void resized() override;

private:
    AdaptiveEchoAudioProcessor& audioProcessor;
    juce::TextButton loadSampleButton { "Load Sample" };
    juce::TextButton trainButton { "Train" };
    juce::Label samplePathLabel;
    juce::Label statusLabel;
    juce::Label frequencyLabel;
    juce::Slider referenceFrequencySlider;
    juce::MidiKeyboardComponent keyboardComponent;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> frequencyAttachment;
    std::unique_ptr<juce::FileChooser> fileChooser;

    void buttonClicked(juce::Button* button) override;
    void changeListenerCallback(juce::ChangeBroadcaster* source) override;
    void refreshFromProcessor();

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(AdaptiveEchoAudioProcessorEditor)
};
