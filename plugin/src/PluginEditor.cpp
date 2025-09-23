#include "PluginEditor.hpp"

AdaptiveEchoAudioProcessorEditor::AdaptiveEchoAudioProcessorEditor(
    AdaptiveEchoAudioProcessor &p)
    : AudioProcessorEditor(&p), processor(p),
      midiKeyboard(processor.getMidiKeyboardState(),
                   juce::MidiKeyboardComponent::horizontalKeyboard) {
    setResizable(true, true);
    setSize(500, 220);

    // Volume slider
    volumeSlider.setSliderStyle(juce::Slider::RotaryHorizontalVerticalDrag);
    volumeSlider.setTextBoxStyle(juce::Slider::TextBoxBelow, false, 60, 20);
    volumeSlider.setRange(0.0, 1.0, 0.0);
    addAndMakeVisible(volumeSlider);

    volumeLabel.setText("Volume", juce::dontSendNotification);
    volumeLabel.setJustificationType(juce::Justification::centred);
    addAndMakeVisible(volumeLabel);

    volumeAttachment = std::make_unique<SliderAttachment>(
        processor.apvts, "volume", volumeSlider);

    addAndMakeVisible(midiKeyboard);
    midiKeyboard.setAvailableRange(24, 108);
}

void AdaptiveEchoAudioProcessorEditor::paint(juce::Graphics &g) {
    g.fillAll(
        getLookAndFeel().findColour(juce::ResizableWindow::backgroundColourId));
    g.setColour(juce::Colours::white);
    g.setFont(16.0f);
    g.drawFittedText("Adaptive Echo - Sine Generator Example (w/ MIDI)",
                     getLocalBounds().reduced(10, 6),
                     juce::Justification::centredTop, 1);
}

void AdaptiveEchoAudioProcessorEditor::resized() {
    auto bounds = getLocalBounds().reduced(12);
    midiKeyboard.setBounds(bounds.removeFromBottom(100).reduced(4));

    auto header = bounds.removeFromTop(34);
    auto row = bounds.withSizeKeepingCentre(bounds.getWidth(), 120);

    auto controlArea = row.removeFromTop(100).withSizeKeepingCentre(200, 100);
    volumeSlider.setBounds(controlArea.removeFromLeft(160).reduced(4));
    volumeLabel.setBounds(volumeSlider.getX(), volumeSlider.getBottom(),
                          volumeSlider.getWidth(), 20);
}