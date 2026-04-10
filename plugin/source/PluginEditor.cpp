#include "PluginEditor.h"

namespace {
const auto kPrimaryText = juce::Colour::fromRGB(37, 31, 26);
const auto kSecondaryText = juce::Colour::fromRGB(91, 74, 58);
}

AdaptiveEchoAudioProcessorEditor::AdaptiveEchoAudioProcessorEditor(
    AdaptiveEchoAudioProcessor& processorToUse)
    : AudioProcessorEditor(&processorToUse),
      audioProcessor(processorToUse),
      keyboardComponent(audioProcessor.getKeyboardState(),
                        juce::MidiKeyboardComponent::horizontalKeyboard) {
    setOpaque(true);
    setSize(1080, 660);
    setWantsKeyboardFocus(true);

    loadSampleButton.addListener(this);
    trainButton.addListener(this);
    loadSampleButton.setColour(juce::TextButton::textColourOffId, kPrimaryText);
    trainButton.setColour(juce::TextButton::textColourOffId, kPrimaryText);

    addAndMakeVisible(loadSampleButton);
    addAndMakeVisible(trainButton);

    samplePathLabel.setText("No sample loaded", juce::dontSendNotification);
    samplePathLabel.setJustificationType(juce::Justification::centredLeft);
    samplePathLabel.setColour(juce::Label::textColourId, kPrimaryText);
    samplePathLabel.setColour(juce::Label::backgroundColourId, juce::Colours::black.withAlpha(0.08f));
    samplePathLabel.setColour(juce::Label::outlineColourId, juce::Colours::transparentBlack);
    addAndMakeVisible(samplePathLabel);

    statusLabel.setJustificationType(juce::Justification::centredLeft);
    statusLabel.setColour(juce::Label::textColourId, kPrimaryText);
    addAndMakeVisible(statusLabel);

    trainingProgressLabel.setJustificationType(juce::Justification::centredLeft);
    trainingProgressLabel.setColour(juce::Label::textColourId, kSecondaryText);
    addAndMakeVisible(trainingProgressLabel);

    trainingProgressBar.setTextToDisplay({});
    addAndMakeVisible(trainingProgressBar);

    frequencyLabel.setText("Reference Frequency (Hz)", juce::dontSendNotification);
    frequencyLabel.setColour(juce::Label::textColourId, kPrimaryText);
    addAndMakeVisible(frequencyLabel);

    referenceFrequencySlider.setSliderStyle(juce::Slider::RotaryHorizontalVerticalDrag);
    referenceFrequencySlider.setTextBoxStyle(juce::Slider::TextBoxBelow, false, 120, 24);
    referenceFrequencySlider.setSkewFactorFromMidPoint(440.0);
    referenceFrequencySlider.setColour(juce::Slider::textBoxTextColourId, kPrimaryText);
    referenceFrequencySlider.setColour(juce::Slider::textBoxOutlineColourId,
                                       juce::Colours::transparentBlack);
    addAndMakeVisible(referenceFrequencySlider);
    oscAPitchTrackToggle.setColour(juce::ToggleButton::textColourId, kPrimaryText);
    oscBPitchTrackToggle.setColour(juce::ToggleButton::textColourId, kPrimaryText);
    addAndMakeVisible(oscAPitchTrackToggle);
    addAndMakeVisible(oscBPitchTrackToggle);

    configureEffectSlider(preHighPassCutoffSlider, preHighPassCutoffLabel, "Pre HP Cutoff");
    configureEffectSlider(preHighPassSlopeSlider, preHighPassSlopeLabel, "Pre HP Slope");
    configureEffectSlider(preLowPassCutoffSlider, preLowPassCutoffLabel, "Pre LP Cutoff");
    configureEffectSlider(preLowPassSlopeSlider, preLowPassSlopeLabel, "Pre LP Slope");
    configureEffectSlider(highPassCutoffSlider, highPassCutoffLabel, "High-Pass Cutoff");
    configureEffectSlider(highPassSlopeSlider, highPassSlopeLabel, "High-Pass Slope");
    configureEffectSlider(lowPassCutoffSlider, lowPassCutoffLabel, "Low-Pass Cutoff");
    configureEffectSlider(lowPassSlopeSlider, lowPassSlopeLabel, "Low-Pass Slope");
    configureEffectSlider(distortionSlider, distortionLabel, "Distortion");

    keyboardLabel.setText("MIDI Keyboard", juce::dontSendNotification);
    keyboardLabel.setColour(juce::Label::textColourId, kPrimaryText);
    addAndMakeVisible(keyboardLabel);

    keyboardComponent.setAvailableRange(24, 96);
    keyboardComponent.setLowestVisibleKey(48);
    keyboardComponent.setKeyWidth(22.0f);
    keyboardComponent.setWantsKeyboardFocus(true);
    addAndMakeVisible(keyboardComponent);

    frequencyAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        audioProcessor.getParameters(), adaptive_echo::plugin_parameters::kReferenceFrequencyId,
        referenceFrequencySlider);
    oscAPitchTrackAttachment =
        std::make_unique<juce::AudioProcessorValueTreeState::ButtonAttachment>(
            audioProcessor.getParameters(), adaptive_echo::plugin_parameters::kOscAPitchTrackId,
            oscAPitchTrackToggle);
    oscBPitchTrackAttachment =
        std::make_unique<juce::AudioProcessorValueTreeState::ButtonAttachment>(
            audioProcessor.getParameters(), adaptive_echo::plugin_parameters::kOscBPitchTrackId,
            oscBPitchTrackToggle);
    preHighPassCutoffAttachment =
        std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
            audioProcessor.getParameters(), adaptive_echo::plugin_parameters::kPreHighPassCutoffId,
            preHighPassCutoffSlider);
    preHighPassSlopeAttachment =
        std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
            audioProcessor.getParameters(), adaptive_echo::plugin_parameters::kPreHighPassSlopeId,
            preHighPassSlopeSlider);
    preLowPassCutoffAttachment =
        std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
            audioProcessor.getParameters(), adaptive_echo::plugin_parameters::kPreLowPassCutoffId,
            preLowPassCutoffSlider);
    preLowPassSlopeAttachment =
        std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
            audioProcessor.getParameters(), adaptive_echo::plugin_parameters::kPreLowPassSlopeId,
            preLowPassSlopeSlider);
    highPassCutoffAttachment =
        std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
            audioProcessor.getParameters(), adaptive_echo::plugin_parameters::kHighPassCutoffId,
            highPassCutoffSlider);
    highPassSlopeAttachment =
        std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
            audioProcessor.getParameters(), adaptive_echo::plugin_parameters::kHighPassSlopeId,
            highPassSlopeSlider);
    lowPassCutoffAttachment =
        std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
            audioProcessor.getParameters(), adaptive_echo::plugin_parameters::kLowPassCutoffId,
            lowPassCutoffSlider);
    lowPassSlopeAttachment =
        std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
            audioProcessor.getParameters(), adaptive_echo::plugin_parameters::kLowPassSlopeId,
            lowPassSlopeSlider);
    distortionAttachment =
        std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
            audioProcessor.getParameters(), adaptive_echo::plugin_parameters::kDistortionAmountId,
            distortionSlider);

    audioProcessor.addChangeListener(this);
    startTimerHz(10);
    refreshFromProcessor();
    keyboardComponent.grabKeyboardFocus();
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

    g.setColour(kPrimaryText);
    g.setFont(juce::Font(juce::FontOptions(28.0f, juce::Font::bold)));
    g.drawText("Adaptive Echo", titleArea.removeFromTop(34), juce::Justification::centredLeft, false);

    g.setFont(14.0f);
    g.setColour(kSecondaryText);
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

    auto pitchTrackArea = bounds.removeFromTop(32);
    oscAPitchTrackToggle.setBounds(pitchTrackArea.removeFromLeft(180));
    pitchTrackArea.removeFromLeft(12);
    oscBPitchTrackToggle.setBounds(pitchTrackArea.removeFromLeft(180));
    bounds.removeFromTop(12);

    auto controlsArea = bounds.removeFromTop(240);
    auto topControls = controlsArea.removeFromTop(120);
    auto bottomControls = controlsArea.removeFromTop(120);

    auto layoutControl = [](juce::Rectangle<int> area, juce::Label& label, juce::Slider& slider) {
        label.setBounds(area.removeFromTop(24));
        slider.setBounds(area);
    };

    auto rowGap = 12;
    auto layoutRow = [layoutControl, rowGap](juce::Rectangle<int> row,
                                             std::initializer_list<std::pair<juce::Label*, juce::Slider*>>
                                                 controls) mutable {
        auto count = static_cast<int>(controls.size());
        auto width = (row.getWidth() - rowGap * (count - 1)) / count;
        for (auto [label, slider] : controls) {
            auto area = row.removeFromLeft(width);
            layoutControl(area, *label, *slider);
            row.removeFromLeft(rowGap);
        }
    };

    layoutRow(topControls,
              {{&frequencyLabel, &referenceFrequencySlider},
               {&preHighPassCutoffLabel, &preHighPassCutoffSlider},
               {&preHighPassSlopeLabel, &preHighPassSlopeSlider},
               {&preLowPassCutoffLabel, &preLowPassCutoffSlider}});
    layoutRow(bottomControls,
              {{&preLowPassSlopeLabel, &preLowPassSlopeSlider},
               {&highPassCutoffLabel, &highPassCutoffSlider},
               {&highPassSlopeLabel, &highPassSlopeSlider},
               {&lowPassCutoffLabel, &lowPassCutoffSlider},
               {&lowPassSlopeLabel, &lowPassSlopeSlider},
               {&distortionLabel, &distortionSlider}});

    bounds.removeFromTop(16);
    keyboardLabel.setBounds(bounds.removeFromTop(24));
    bounds.removeFromTop(8);
    keyboardComponent.setBounds(bounds.removeFromBottom(150));
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

void AdaptiveEchoAudioProcessorEditor::configureEffectSlider(
    juce::Slider& slider, juce::Label& label, const juce::String& text) {
    label.setText(text, juce::dontSendNotification);
    label.setColour(juce::Label::textColourId, kPrimaryText);
    addAndMakeVisible(label);

    slider.setSliderStyle(juce::Slider::RotaryHorizontalVerticalDrag);
    slider.setTextBoxStyle(juce::Slider::TextBoxBelow, false, 100, 24);
    slider.setRange(0.0, 1.0, 0.001);
    slider.setColour(juce::Slider::textBoxTextColourId, kPrimaryText);
    slider.setColour(juce::Slider::textBoxOutlineColourId, juce::Colours::transparentBlack);
    addAndMakeVisible(slider);
}
