#include "PluginEditor.hpp"
#include "Constants.hpp"

AdaptiveEchoAudioProcessorEditor::AdaptiveEchoAudioProcessorEditor(
    AdaptiveEchoAudioProcessor &p)
    : AudioProcessorEditor(&p), processor(p),
      midiKeyboard(processor.getMidiKeyboardState(),
                   juce::MidiKeyboardComponent::horizontalKeyboard),
      envelopeViewer(processor.apvts) {
    setResizable(true, true);
    setSize(1000, 440);

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
    attackSlider.setTextBoxStyle(juce::Slider::NoTextBox, false, 60, 20);
    attackSlider.setRange(ADSR_MIN, ADSR_MAX, 0.0);

    attackCurveSlider.setSliderStyle(
        juce::Slider::RotaryHorizontalVerticalDrag);
    attackCurveSlider.setRange(ADSR_MIN, ADSR_MAX, 0.0);
    attackCurveSlider.setTextBoxStyle(juce::Slider::NoTextBox, false, 60, 20);

    addAndMakeVisible(attackSlider);
    addAndMakeVisible(attackCurveSlider);

    decaySlider.setSliderStyle(juce::Slider::RotaryHorizontalVerticalDrag);
    decaySlider.setTextBoxStyle(juce::Slider::NoTextBox, false, 60, 20);
    decaySlider.setRange(ADSR_MIN, ADSR_MAX, 0.0);

    decayCurveSlider.setSliderStyle(juce::Slider::RotaryHorizontalVerticalDrag);
    decayCurveSlider.setTextBoxStyle(juce::Slider::NoTextBox, false, 60, 20);
    decayCurveSlider.setRange(ADSR_MIN, ADSR_MAX, 0.0);

    addAndMakeVisible(decaySlider);
    addAndMakeVisible(decayCurveSlider);

    sustainSlider.setSliderStyle(juce::Slider::RotaryHorizontalVerticalDrag);
    sustainSlider.setTextBoxStyle(juce::Slider::NoTextBox, false, 60, 20);
    sustainSlider.setRange(0.0, 1.0, 0.0);

    addAndMakeVisible(sustainSlider);

    releaseSlider.setSliderStyle(juce::Slider::RotaryHorizontalVerticalDrag);
    releaseSlider.setTextBoxStyle(juce::Slider::NoTextBox, false, 60, 20);
    releaseSlider.setRange(ADSR_MIN, ADSR_MAX, 0.0);

    releaseCurveSlider.setSliderStyle(juce::Slider::RotaryHorizontalVerticalDrag);
    releaseCurveSlider.setTextBoxStyle(juce::Slider::NoTextBox, false, 60, 20);
    releaseCurveSlider.setRange(ADSR_MIN, ADSR_MAX, 0.0);

    addAndMakeVisible(releaseSlider);
    addAndMakeVisible(releaseCurveSlider);

    attackLabel.setJustificationType(juce::Justification::centred);
    attackLabel.setText("Attack", juce::dontSendNotification);

    decayLabel.setJustificationType(juce::Justification::centred);
    decayLabel.setText("Decay", juce::dontSendNotification);

    sustainLabel.setJustificationType(juce::Justification::centred);
    sustainLabel.setText("Sustain", juce::dontSendNotification);

    releaseLabel.setJustificationType(juce::Justification::centred);
    releaseLabel.setText("Release", juce::dontSendNotification);

    attackCurveLabel.setJustificationType(juce::Justification::centred);
    attackCurveLabel.setText("Attack Curve", juce::dontSendNotification);

    decayCurveLabel.setJustificationType(juce::Justification::centred);
    decayCurveLabel.setText("Decay Curve", juce::dontSendNotification);

    releaseCurveLabel.setJustificationType(juce::Justification::centred);
    releaseCurveLabel.setText("Release Curve", juce::dontSendNotification);

    addAndMakeVisible(attackLabel);
    addAndMakeVisible(attackCurveLabel);
    addAndMakeVisible(decayLabel);
    addAndMakeVisible(decayCurveLabel);
    addAndMakeVisible(sustainLabel);
    addAndMakeVisible(releaseLabel);
    addAndMakeVisible(releaseCurveLabel);

    addAndMakeVisible(envelopeViewer);

    volumeAttachment = std::make_unique<SliderAttachment>(
        processor.apvts, "volume", volumeSlider);
    attackAttachment = std::make_unique<SliderAttachment>(
        processor.apvts, "attack", attackSlider);
    decayAttachment = std::make_unique<SliderAttachment>(
        processor.apvts, "decay", decaySlider);
    sustainAttachment = std::make_unique<SliderAttachment>(
        processor.apvts, "sustain", sustainSlider);
    releaseAttachment = std::make_unique<SliderAttachment>(
        processor.apvts, "release", releaseSlider);

    attackCurveAttachment = std::make_unique<SliderAttachment>(
        processor.apvts, "attackCurve", attackCurveSlider);
    decayCurveAttachment = std::make_unique<SliderAttachment>(
        processor.apvts, "decayCurve", decayCurveSlider);
    releaseCurveAttachment = std::make_unique<SliderAttachment>(
        processor.apvts, "releaseCurve", releaseCurveSlider);

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

    // auto header = bounds.removeFromTop(34);
    bounds.removeFromTop(34);
    auto row = bounds.withSizeKeepingCentre(bounds.getWidth(), 120);

    int totalSliderWidth = bounds.getWidth() / 2;
    int viewerWidth = bounds.getWidth() / 2;
    int sliderWidth = totalSliderWidth / 5;

    auto controlArea = row.removeFromTop(120);
    envelopeViewer.setBounds(
        controlArea.removeFromLeft(viewerWidth).reduced(4));

    auto volumeArea = controlArea.removeFromLeft(sliderWidth);
    auto attackArea = controlArea.removeFromLeft(sliderWidth);
    auto decayArea = controlArea.removeFromLeft(sliderWidth);
    auto sustainArea = controlArea.removeFromLeft(sliderWidth);
    auto releaseArea = controlArea.removeFromLeft(sliderWidth);

    volumeSlider.setBounds(volumeArea);
    volumeLabel.setBounds(volumeSlider.getX(), volumeSlider.getBottom(),
                          volumeSlider.getWidth(), 20);

    attackSlider.setBounds(
        attackArea.removeFromTop(attackArea.getHeight() / 2));
    attackCurveSlider.setBounds(
        attackArea.removeFromBottom(attackArea.getHeight()));

    decaySlider.setBounds(decayArea.removeFromTop(decayArea.getHeight() / 2));
    decayCurveSlider.setBounds(
        decayArea.removeFromBottom(decayArea.getHeight()));

    sustainSlider.setBounds(sustainArea);

    releaseSlider.setBounds(
        releaseArea.removeFromTop(releaseArea.getHeight() / 2));
    releaseCurveSlider.setBounds(
        releaseArea.removeFromBottom(releaseArea.getHeight()));

    attackLabel.setBounds(attackSlider.getX(), attackSlider.getBottom(),
                          attackSlider.getWidth(), 20);
    attackCurveLabel.setBounds(attackCurveSlider.getX(),
                               attackCurveSlider.getBottom(),
                               attackCurveSlider.getWidth(), 20);
    decayLabel.setBounds(decaySlider.getX(), decaySlider.getBottom(),
                         decaySlider.getWidth(), 20);
    decayCurveLabel.setBounds(decayCurveSlider.getX(),
                              decayCurveSlider.getBottom(),
                              decayCurveSlider.getWidth(), 20);
    sustainLabel.setBounds(sustainSlider.getX(), sustainSlider.getBottom(),
                           sustainSlider.getWidth(), 20);
    releaseLabel.setBounds(releaseSlider.getX(), releaseSlider.getBottom(),
                           releaseSlider.getWidth(), 20);
    releaseCurveLabel.setBounds(releaseCurveSlider.getX(),
                                releaseCurveSlider.getBottom(),
                                releaseCurveSlider.getWidth(), 20);
}
