#include "PluginEditor.hpp"
#include "AudioFileLoader.hpp"

AdaptiveEchoAudioProcessorEditor::AdaptiveEchoAudioProcessorEditor(
    AdaptiveEchoAudioProcessor &p)
    : AudioProcessorEditor(&p), processor(p),
      midiKeyboard(processor.getMidiKeyboardState(),
                   juce::MidiKeyboardComponent::horizontalKeyboard) {
    setResizable(true, true);
    setSize(500, 220);

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

    addAndMakeVisible (openFileButton);

    openFileButton.onClick = [this]
    {
	auto chooser = std::make_shared<juce::FileChooser>(
            "Select an audio file",
            juce::File{},
            "*.wav;*.aiff;*.mp3"
        );

        chooser->launchAsync (
            juce::FileBrowserComponent::openMode
            | juce::FileBrowserComponent::canSelectFiles,
	    [this, chooser] (const juce::FileChooser& fc)
            {
		auto file = fc.getResult();
	        if (! file.existsAsFile())
                    return;
	        processor.loadFile (file);
		repaint(); // forces paint() to update with new sample text
            }
        );
    };

    addAndMakeVisible(midiKeyboard);
    midiKeyboard.setAvailableRange(24, 108);
}

void AdaptiveEchoAudioProcessorEditor::paint(juce::Graphics &g) {
    g.fillAll(getLookAndFeel().findColour(juce::ResizableWindow::backgroundColourId));
    g.setColour(juce::Colours::white);
    g.setFont(16.0f);
    g.drawFittedText("Adaptive Echo - Sine Generator Example (w/ MIDI)",
                     getLocalBounds().reduced(10, 6),
                     juce::Justification::centredTop, 1);

    // Only display first 10 samples
    g.setFont(14.0f);
    auto y = 50;
    g.drawText("First 10 samples:", 10, y, 400, 20, juce::Justification::left);
    y += 20;

    juce::String s;
    int numToShow = std::min(10, (int)processor.loadedSamples.size());
    for (int i = 0; i < numToShow; ++i)
        s += juce::String(processor.loadedSamples[i], 5) + "  ";
    g.drawText(s, 10, y, 400, 20, juce::Justification::left);

    // Debug: print a few samples to console only, NOT all
    int numToPrint = std::min(1000, (int)processor.loadedSamples.size());
    for (int i = 0; i < numToPrint; ++i)
        DBG("Sample " << i << ": " << processor.loadedSamples[i]);
}

void AdaptiveEchoAudioProcessorEditor::resized() {
    auto bounds = getLocalBounds().reduced(12);
    midiKeyboard.setBounds(bounds.removeFromBottom(100).reduced(4));

    // auto header = bounds.removeFromTop(34);
    bounds.removeFromTop(34);
    auto row = bounds.withSizeKeepingCentre(bounds.getWidth(), 120);

    int sliderWidth = bounds.getWidth() / 5; // one for volume + 4 ADSR

    auto controlArea = row.removeFromTop(120);
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

    openFileButton.setBounds (10, 10, 120, 24);
}
