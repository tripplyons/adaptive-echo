#include "PluginEditor.h"

AdaptiveEchoAudioProcessorEditor::AdaptiveEchoAudioProcessorEditor(
    AdaptiveEchoAudioProcessor& processorToUse)
    : AudioProcessorEditor(&processorToUse),
      audioProcessor(processorToUse),
      keyboardComponent(audioProcessor.getKeyboardState(),
                        juce::MidiKeyboardComponent::horizontalKeyboard) {
    setOpaque(true);
    setSize(720, 420);

    loadSampleButton.addListener(this);
    trainButton.addListener(this);

    addAndMakeVisible(loadSampleButton);
    addAndMakeVisible(trainButton);

    samplePathLabel.setText("No sample loaded", juce::dontSendNotification);
    samplePathLabel.setJustificationType(juce::Justification::centredLeft);
    samplePathLabel.setColour(juce::Label::backgroundColourId, juce::Colours::black.withAlpha(0.08f));
    samplePathLabel.setColour(juce::Label::outlineColourId, juce::Colours::transparentBlack);
    addAndMakeVisible(samplePathLabel);

    statusLabel.setJustificationType(juce::Justification::centredLeft);
    addAndMakeVisible(statusLabel);

    trainingProgressLabel.setJustificationType(juce::Justification::centredLeft);
    trainingProgressLabel.setColour(juce::Label::textColourId, juce::Colour::fromRGB(91, 74, 58));
    addAndMakeVisible(trainingProgressLabel);

    trainingProgressBar.setTextToDisplay({});
    addAndMakeVisible(trainingProgressBar);

    frequencyLabel.setText("Reference Frequency (Hz)", juce::dontSendNotification);
    addAndMakeVisible(frequencyLabel);

    referenceFrequencySlider.setSliderStyle(juce::Slider::RotaryHorizontalVerticalDrag);
    referenceFrequencySlider.setTextBoxStyle(juce::Slider::TextBoxBelow, false, 120, 24);
    referenceFrequencySlider.setSkewFactorFromMidPoint(440.0);
    addAndMakeVisible(referenceFrequencySlider);

    addAndMakeVisible(keyboardComponent);

    frequencyAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        audioProcessor.getParameters(), "referenceFrequencyHz", referenceFrequencySlider);

    audioProcessor.addChangeListener(this);
    startTimerHz(10);
    refreshFromProcessor();
}

AdaptiveEchoAudioProcessorEditor::~AdaptiveEchoAudioProcessorEditor() {
    stopTimer();
    audioProcessor.removeChangeListener(this);
    loadSampleButton.removeListener(this);
    trainButton.removeListener(this);
}

void AdaptiveEchoAudioProcessorEditor::paint(juce::Graphics& g) {
    g.fillAll(juce::Colour::fromRGB(241, 236, 229));

    auto bounds = getLocalBounds().reduced(20);
    auto titleArea = bounds.removeFromTop(60);

    g.setColour(juce::Colour::fromRGB(37, 31, 26));
    g.setFont(juce::Font(juce::FontOptions(28.0f, juce::Font::bold)));
    g.drawText("Adaptive Echo", titleArea.removeFromTop(34), juce::Justification::centredLeft, false);

    g.setFont(14.0f);
    g.setColour(juce::Colour::fromRGB(91, 74, 58));
    g.drawText("Train a sampled sound and replay it as a tuned one-shot synth.",
               titleArea, juce::Justification::centredLeft, false);
}

void AdaptiveEchoAudioProcessorEditor::resized() {
    auto bounds = getLocalBounds().reduced(20);
    bounds.removeFromTop(72);

    auto topRow = bounds.removeFromTop(40);
    loadSampleButton.setBounds(topRow.removeFromLeft(150));
    topRow.removeFromLeft(12);
    trainButton.setBounds(topRow.removeFromLeft(140));

    bounds.removeFromTop(12);
    samplePathLabel.setBounds(bounds.removeFromTop(28));
    bounds.removeFromTop(8);
    statusLabel.setBounds(bounds.removeFromTop(24));
    bounds.removeFromTop(8);
    trainingProgressLabel.setBounds(bounds.removeFromTop(24));
    bounds.removeFromTop(8);
    trainingProgressBar.setBounds(bounds.removeFromTop(22));
    bounds.removeFromTop(18);

    auto controlsRow = bounds.removeFromTop(140);
    auto frequencyArea = controlsRow.removeFromLeft(220);
    frequencyLabel.setBounds(frequencyArea.removeFromTop(24));
    referenceFrequencySlider.setBounds(frequencyArea);

    bounds.removeFromTop(12);
    keyboardComponent.setBounds(bounds.removeFromBottom(120));
}

void AdaptiveEchoAudioProcessorEditor::buttonClicked(juce::Button* button) {
    if (button == &loadSampleButton) {
        fileChooser = std::make_unique<juce::FileChooser>("Choose a sample to train");
        auto flags = juce::FileBrowserComponent::openMode |
                     juce::FileBrowserComponent::canSelectFiles;
        fileChooser->launchAsync(flags, [this](const juce::FileChooser& chooser) {
            const auto file = chooser.getResult();
            if (file.existsAsFile()) {
                audioProcessor.loadSampleFromPath(file.getFullPathName());
            }
        });
        return;
    }

    if (button == &trainButton) {
        audioProcessor.beginTraining();
        refreshFromProcessor();
    }
}

void AdaptiveEchoAudioProcessorEditor::changeListenerCallback(juce::ChangeBroadcaster* source) {
    if (source == &audioProcessor) {
        refreshFromProcessor();
    }
}

void AdaptiveEchoAudioProcessorEditor::timerCallback() {
    refreshFromProcessor();
}

void AdaptiveEchoAudioProcessorEditor::refreshFromProcessor() {
    const auto samplePath = audioProcessor.getSamplePath();
    samplePathLabel.setText(samplePath.isNotEmpty() ? samplePath : "No sample loaded",
                            juce::dontSendNotification);
    statusLabel.setText(audioProcessor.getStatusText(), juce::dontSendNotification);
    trainingProgressLabel.setText(audioProcessor.getTrainingProgressText(),
                                  juce::dontSendNotification);
    trainingProgressValue = audioProcessor.getTrainingProgress();
    const auto shouldShowProgress =
        audioProcessor.isTraining() || trainingProgressValue > 0.0 ||
        trainingProgressLabel.getText().isNotEmpty();
    trainingProgressLabel.setVisible(shouldShowProgress);
    trainingProgressBar.setVisible(shouldShowProgress);
    trainButton.setEnabled(audioProcessor.canTrain());
    repaint();
}
