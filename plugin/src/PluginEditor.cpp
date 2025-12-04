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

    // Global background colour
    getLookAndFeel().setColour(juce::ResizableWindow::backgroundColourId,
                               UI::background);

    // Slider styling
    auto stylize = [&](juce::Slider& s, juce::Label& l)
    {
        s.setColour(juce::Slider::rotarySliderFillColourId, UI::sliderFill);
        s.setColour(juce::Slider::rotarySliderOutlineColourId, UI::sliderOutline);
        s.setColour(juce::Slider::thumbColourId, UI::thumb);
        s.setColour(juce::Slider::textBoxTextColourId, UI::text);
        l.setColour(juce::Label::textColourId, UI::labelText);
    };

    stylize(volumeSlider,  volumeLabel);
    stylize(attackSlider,  attackLabel);
    stylize(decaySlider,   decayLabel);
    stylize(sustainSlider, sustainLabel);
    stylize(releaseSlider, releaseLabel);

    // Volume slider
    volumeSlider.setSliderStyle(juce::Slider::RotaryHorizontalVerticalDrag);
    volumeSlider.setTextBoxStyle(juce::Slider::NoTextBox, false, 60, 20);
    volumeSlider.setRange(0.0, 1.0, 0.0);
    addAndMakeVisible(volumeSlider);

    volumeLabel.setText("Volume", juce::dontSendNotification);
    volumeLabel.setJustificationType(juce::Justification::centred);
    addAndMakeVisible(volumeLabel);

    // ADSR
    attackSlider.setSliderStyle(juce::Slider::RotaryHorizontalVerticalDrag);
    attackSlider.setTextBoxStyle(juce::Slider::NoTextBox, false, 60, 20);
    attackSlider.setRange(ADSR_MIN, ADSR_MAX, 0.0);

    addAndMakeVisible(attackSlider);

    decaySlider.setSliderStyle(juce::Slider::RotaryHorizontalVerticalDrag);
    decaySlider.setTextBoxStyle(juce::Slider::NoTextBox, false, 60, 20);
    decaySlider.setRange(ADSR_MIN, ADSR_MAX, 0.0);

    addAndMakeVisible(decaySlider);

    sustainSlider.setSliderStyle(juce::Slider::RotaryHorizontalVerticalDrag);
    sustainSlider.setTextBoxStyle(juce::Slider::NoTextBox, false, 60, 20);
    sustainSlider.setRange(0.0, 1.0, 0.0);

    addAndMakeVisible(sustainSlider);

    releaseSlider.setSliderStyle(juce::Slider::RotaryHorizontalVerticalDrag);
    releaseSlider.setTextBoxStyle(juce::Slider::NoTextBox, false, 60, 20);
    releaseSlider.setRange(ADSR_MIN, ADSR_MAX, 0.0);

    addAndMakeVisible(releaseSlider);

    attackLabel.setJustificationType(juce::Justification::centred);
    attackLabel.setText("Attack", juce::dontSendNotification);

    decayLabel.setJustificationType(juce::Justification::centred);
    decayLabel.setText("Decay", juce::dontSendNotification);

    sustainLabel.setJustificationType(juce::Justification::centred);
    sustainLabel.setText("Sustain", juce::dontSendNotification);

    releaseLabel.setJustificationType(juce::Justification::centred);
    releaseLabel.setText("Release", juce::dontSendNotification);

    addAndMakeVisible(attackLabel);
    addAndMakeVisible(decayLabel);
    addAndMakeVisible(sustainLabel);
    addAndMakeVisible(releaseLabel);

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

    addAndMakeVisible(midiKeyboard);
    midiKeyboard.setAvailableRange(24, 108);
}

void AdaptiveEchoAudioProcessorEditor::paint(juce::Graphics& g)
{
    g.fillAll(UI::background);

    g.setColour(UI::text);
    g.setFont(48.0f);

    g.drawFittedText("Adaptive Echo",
                     getLocalBounds().reduced(10, 6),
                     juce::Justification::topLeft,
                     0);
}

void AdaptiveEchoAudioProcessorEditor::resized() {
    auto bounds = getLocalBounds().reduced(12);
    midiKeyboard.setBounds(bounds.removeFromBottom(100).reduced(4));

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

    attackSlider.setBounds(
        attackArea.removeFromTop(attackArea.getHeight()));

    decaySlider.setBounds(decayArea.removeFromTop(decayArea.getHeight()));

    sustainSlider.setBounds(sustainArea);

    releaseSlider.setBounds(
        releaseArea.removeFromTop(releaseArea.getHeight()));

    attackLabel.setBounds(attackSlider.getX(), attackSlider.getBottom(),
                          attackSlider.getWidth(), 20);
    decayLabel.setBounds(decaySlider.getX(), decaySlider.getBottom(),
                         decaySlider.getWidth(), 20);
    sustainLabel.setBounds(sustainSlider.getX(), sustainSlider.getBottom(),
                           sustainSlider.getWidth(), 20);
    releaseLabel.setBounds(releaseSlider.getX(), releaseSlider.getBottom(),
                           releaseSlider.getWidth(), 20);
    volumeLabel.setBounds(volumeSlider.getX(), volumeSlider.getBottom(),
                          volumeSlider.getWidth(), 20);
}
