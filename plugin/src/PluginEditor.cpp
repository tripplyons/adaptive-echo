#include "PluginEditor.hpp"

AdaptiveEchoAudioProcessorEditor::AdaptiveEchoAudioProcessorEditor(
    AdaptiveEchoAudioProcessor &p)
    : AudioProcessorEditor(&p), processor(p),
      midiKeyboard(processor.getMidiKeyboardState(),
                   juce::MidiKeyboardComponent::horizontalKeyboard) {
    setResizable(true, true);
    setSize(500, 300);

    // Volume slider
    volumeSlider.setSliderStyle(juce::Slider::RotaryHorizontalVerticalDrag);
    volumeSlider.setTextBoxStyle(juce::Slider::TextBoxAbove, false, 60, 20);
    volumeSlider.setRange(0.0, 1.0, 0.0);
    addAndMakeVisible(volumeSlider);

    volumeLabel.setText("Volume", juce::dontSendNotification);
    volumeLabel.setJustificationType(juce::Justification::centred);
    addAndMakeVisible(volumeLabel);

    // ADSR
    attackSlider.setSliderStyle(juce::Slider::RotaryHorizontalVerticalDrag);
    attackSlider.setTextBoxStyle(juce::Slider::TextBoxAbove, false, 60, 20);
    attackSlider.setRange(0.01, 5.0, 0.0);

    attackLabel.setJustificationType(juce::Justification::centred);
    attackLabel.setText("attack", juce::dontSendNotification);
    
    addAndMakeVisible(attackSlider);
    addAndMakeVisible(attackLabel);

    decaySlider.setSliderStyle(juce::Slider::RotaryHorizontalVerticalDrag);
    decaySlider.setTextBoxStyle(juce::Slider::TextBoxAbove, false, 60, 20);
    decaySlider.setRange(0.01, 5.0, 0.0);

    decayLabel.setJustificationType(juce::Justification::centred);
    decayLabel.setText("decay", juce::dontSendNotification);

    addAndMakeVisible(decaySlider);
    addAndMakeVisible(decayLabel);

    sustainSlider.setSliderStyle(juce::Slider::RotaryHorizontalVerticalDrag);
    sustainSlider.setTextBoxStyle(juce::Slider::TextBoxAbove, false, 60, 20);
    sustainSlider.setRange(0.0, 1.0, 0.0);

    sustainLabel.setJustificationType(juce::Justification::centred);
    sustainLabel.setText("sustain", juce::dontSendNotification);

    addAndMakeVisible(sustainSlider);
    addAndMakeVisible(sustainLabel);

    releaseSlider.setSliderStyle(juce::Slider::RotaryHorizontalVerticalDrag);
    releaseSlider.setTextBoxStyle(juce::Slider::TextBoxAbove, false, 60, 20);
    releaseSlider.setRange(0.01, 5.0, 0.0);

    releaseLabel.setJustificationType(juce::Justification::centred);
    releaseLabel.setText("release", juce::dontSendNotification);

    addAndMakeVisible(releaseSlider);
    addAndMakeVisible(releaseLabel);

    // Oscillator parameters
    harshnessSlider.setSliderStyle(juce::Slider::RotaryHorizontalVerticalDrag);
    harshnessSlider.setTextBoxStyle(juce::Slider::TextBoxAbove, false, 60, 20);
    harshnessSlider.setRange(0.1, 50.0, 0.0);

    harshnessLabel.setJustificationType(juce::Justification::centred);
    harshnessLabel.setText("Harshness", juce::dontSendNotification);

    addAndMakeVisible(harshnessSlider);
    addAndMakeVisible(harshnessLabel);

    warmthSlider.setSliderStyle(juce::Slider::RotaryHorizontalVerticalDrag);
    warmthSlider.setTextBoxStyle(juce::Slider::TextBoxAbove, false, 60, 20);
    warmthSlider.setRange(0.0, 20.0, 0.0);

    warmthLabel.setJustificationType(juce::Justification::centred);
    warmthLabel.setText("Warmth", juce::dontSendNotification);

    addAndMakeVisible(warmthSlider);
    addAndMakeVisible(warmthLabel);

    volumeAttachment = std::make_unique<SliderAttachment>(
        processor.apvts, "volume", volumeSlider);
    attackAttachment = std::make_unique<SliderAttachment>(
        processor.apvts, "attack", attackSlider);
    decayAttachment = std::make_unique<SliderAttachment>(processor.apvts,
                                                         "decay", decaySlider);
    sustainAttachment = std::make_unique<SliderAttachment>(
        processor.apvts, "sustain", sustainSlider);
    releaseAttachment = std::make_unique<SliderAttachment>(
        processor.apvts, "release", releaseSlider);
    warmthAttachment = std::make_unique<SliderAttachment>(
        processor.apvts, "warmth", warmthSlider);
    harshnessAttachment = std::make_unique<SliderAttachment>(
        processor.apvts, "harshness", harshnessSlider);

    addAndMakeVisible(midiKeyboard);
    midiKeyboard.setAvailableRange(24, 108);

    oscView = std::make_unique<OscillatorVisualizer>(processor.osc);
    if (oscView) {
        addAndMakeVisible(*oscView);
        oscVisible = true;
    } else {
        oscVisible = false;
    }
}

void AdaptiveEchoAudioProcessorEditor::paint(juce::Graphics &g) {
    g.fillAll(
        getLookAndFeel().findColour(juce::ResizableWindow::backgroundColourId));
    g.setColour(juce::Colours::white);
    g.setFont(16.0f);
    g.drawFittedText("Adaptive Echo - Sine Generator Example (w/ MIDI)",
                     getLocalBounds().reduced(10, 6),
                     juce::Justification::centredTop, 1);

    if (oscView && !oscVisible) {
        oscView->update();
        oscView->repaint();
    }
}

void AdaptiveEchoAudioProcessorEditor::resized() {
    auto bounds = getLocalBounds().reduced(12);

    midiKeyboard.setBounds(bounds.removeFromBottom(100).reduced(4));

    bounds.removeFromTop(34);

    auto row = bounds.withSizeKeepingCentre(bounds.getWidth(), 240);
    if (oscView)
        oscView->setBounds(row.removeFromTop(120).reduced(4));
    auto controlArea = row.removeFromTop(120);

    int sliderWidth = controlArea.getWidth() / 7;

    volumeSlider.setBounds(controlArea.removeFromLeft(sliderWidth).reduced(4));
    volumeLabel.setBounds(volumeSlider.getX(), volumeSlider.getBottom(),
                          volumeSlider.getWidth(), 20);

    attackSlider.setBounds(controlArea.removeFromLeft(sliderWidth).reduced(4));
    attackLabel.setBounds(attackSlider.getX(), attackSlider.getBottom(),
                          attackSlider.getWidth(), 20);

    decaySlider.setBounds(controlArea.removeFromLeft(sliderWidth).reduced(4));
    decayLabel.setBounds(decaySlider.getX(), decaySlider.getBottom(),
                         decaySlider.getWidth(), 20);

    sustainSlider.setBounds(controlArea.removeFromLeft(sliderWidth).reduced(4));
    sustainLabel.setBounds(sustainSlider.getX(), sustainSlider.getBottom(),
                           sustainSlider.getWidth(), 20);

    releaseSlider.setBounds(controlArea.removeFromLeft(sliderWidth).reduced(4));
    releaseLabel.setBounds(releaseSlider.getX(), releaseSlider.getBottom(),
                           releaseSlider.getWidth(), 20);

    warmthSlider.setBounds(controlArea.removeFromLeft(sliderWidth).reduced(4));
    warmthLabel.setBounds(warmthSlider.getX(), warmthSlider.getBottom(),
                          warmthSlider.getWidth(), 20);

    harshnessSlider.setBounds(
        controlArea.removeFromLeft(sliderWidth).reduced(4));
    harshnessLabel.setBounds(harshnessSlider.getX(),
                             harshnessSlider.getBottom(),
                             harshnessSlider.getWidth(), 20);
}
