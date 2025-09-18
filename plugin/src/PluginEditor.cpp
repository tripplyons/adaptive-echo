#include "PluginEditor.hpp"

AdaptiveEchoAudioProcessorEditor::AdaptiveEchoAudioProcessorEditor(
    AdaptiveEchoAudioProcessor &p)
    : AudioProcessorEditor(&p), processor(p) {
  setResizable(true, true);
  setSize(500, 160);

  // Volume slider
  volumeSlider.setSliderStyle(juce::Slider::RotaryHorizontalVerticalDrag);
  volumeSlider.setTextBoxStyle(juce::Slider::TextBoxBelow, false, 60, 20);
  volumeSlider.setRange(0.0, 1.0, 0.0);
  addAndMakeVisible(volumeSlider);

  volumeLabel.setText("Volume", juce::dontSendNotification);
  volumeLabel.setJustificationType(juce::Justification::centred);
  addAndMakeVisible(volumeLabel);

  volumeAttachment = std::make_unique<SliderAttachment>(processor.apvts,
                                                        "volume", volumeSlider);

  // Frequency slider
  freqSlider.setSliderStyle(juce::Slider::RotaryHorizontalVerticalDrag);
  freqSlider.setTextBoxStyle(juce::Slider::TextBoxBelow, false, 60, 20);
  freqSlider.setRange(20.0, 2000.0, 0.01);
  addAndMakeVisible(freqSlider);

  freqLabel.setText("Frequency", juce::dontSendNotification);
  freqLabel.setJustificationType(juce::Justification::centred);
  addAndMakeVisible(freqLabel);

  freqAttachment = std::make_unique<SliderAttachment>(processor.apvts,
                                                      "freq", freqSlider);
}

void AdaptiveEchoAudioProcessorEditor::paint(juce::Graphics &g) {
  g.fillAll(
      getLookAndFeel().findColour(juce::ResizableWindow::backgroundColourId));
  g.setColour(juce::Colours::white);
  g.setFont(16.0f);
  g.drawFittedText("Adaptive Echo - Sine Generator Example",
                   getLocalBounds().reduced(10, 6),
                   juce::Justification::centredTop, 1);
}

void AdaptiveEchoAudioProcessorEditor::resized() {
  auto bounds = getLocalBounds().reduced(20);
  auto top = bounds.removeFromTop(30);

  juce::ignoreUnused(top);

  auto row = bounds.withSizeKeepingCentre(400, 100);
  volumeSlider.setBounds(row.removeFromLeft(200));
  volumeLabel.setBounds(volumeSlider.getX(), volumeSlider.getBottom(),
                        volumeSlider.getWidth(), 20);

  freqSlider.setBounds(row.removeFromLeft(200));
  freqLabel.setBounds(freqSlider.getX(), freqSlider.getBottom(),
                      freqSlider.getWidth(), 20);
}
