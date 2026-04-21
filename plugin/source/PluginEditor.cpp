#include "PluginEditor.h"

namespace {
const auto kBackground = juce::Colours::white;
const auto kPanelBackground = juce::Colour::fromRGB(246, 249, 255);
const auto kPrimaryText = juce::Colours::black;
const auto kSecondaryText = juce::Colour::fromRGB(78, 86, 99);
const auto kAccentBlue = juce::Colour::fromRGB(34, 110, 255);
const auto kAccentBlueDark = juce::Colour::fromRGB(21, 79, 201);
const auto kAccentBlueSoft = juce::Colour::fromRGB(228, 238, 255);

void drawPanel(juce::Graphics& g, juce::Rectangle<float> area) {
    g.setColour(kPanelBackground);
    g.fillRoundedRectangle(area, 18.0f);
}

void drawPanelHeader(juce::Graphics& g, juce::Rectangle<int> area, const juce::String& title) {
    g.setColour(kPrimaryText);
    g.setFont(juce::Font(juce::FontOptions(18.0f, juce::Font::bold)));
    g.drawText(title, area.removeFromTop(24), juce::Justification::centredLeft, false);
}
}  // namespace

AdaptiveEchoAudioProcessorEditor::AdaptiveEchoAudioProcessorEditor(
    AdaptiveEchoAudioProcessor& processorToUse)
    : AudioProcessorEditor(&processorToUse),
      audioProcessor(processorToUse),
      keyboardComponent(audioProcessor.getKeyboardState(),
                        juce::MidiKeyboardComponent::horizontalKeyboard) {
    setOpaque(true);
    setSize(1180, 860);
    setWantsKeyboardFocus(true);

    loadSampleButton.addListener(this);
    trainButton.addListener(this);
    loadSampleButton.setColour(juce::TextButton::buttonColourId, kAccentBlueSoft);
    loadSampleButton.setColour(juce::TextButton::buttonOnColourId, kAccentBlue);
    loadSampleButton.setColour(juce::TextButton::textColourOffId, kPrimaryText);
    trainButton.setColour(juce::TextButton::buttonColourId, kAccentBlue);
    trainButton.setColour(juce::TextButton::buttonOnColourId, kAccentBlueDark);
    trainButton.setColour(juce::TextButton::textColourOffId, juce::Colours::white);
    trainButton.setColour(juce::TextButton::textColourOnId, juce::Colours::white);
    addAndMakeVisible(loadSampleButton);
    addAndMakeVisible(trainButton);

    samplePathLabel.setText("No sample loaded", juce::dontSendNotification);
    samplePathLabel.setJustificationType(juce::Justification::centredLeft);
    samplePathLabel.setColour(juce::Label::textColourId, kPrimaryText);
    samplePathLabel.setColour(juce::Label::backgroundColourId, kAccentBlueSoft);
    samplePathLabel.setColour(juce::Label::outlineColourId, juce::Colours::transparentBlack);
    samplePathLabel.setFont(juce::FontOptions(14.0f));
    addAndMakeVisible(samplePathLabel);

    statusLabel.setJustificationType(juce::Justification::centredLeft);
    statusLabel.setColour(juce::Label::textColourId, kPrimaryText);
    statusLabel.setFont(juce::FontOptions(15.0f, juce::Font::bold));
    addAndMakeVisible(statusLabel);

    trainingProgressLabel.setJustificationType(juce::Justification::centredLeft);
    trainingProgressLabel.setColour(juce::Label::textColourId, kPrimaryText);
    trainingProgressLabel.setFont(juce::FontOptions(14.0f, juce::Font::bold));
    addAndMakeVisible(trainingProgressLabel);

    trainingProgressBar.setTextToDisplay({});
    trainingProgressBar.setColour(juce::ProgressBar::foregroundColourId, kAccentBlue);
    trainingProgressBar.setColour(juce::ProgressBar::backgroundColourId,
                                  kAccentBlue.withAlpha(0.12f));
    addAndMakeVisible(trainingProgressBar);

    trainingTimeLabel.setText("Training Time", juce::dontSendNotification);
    trainingTimeLabel.setColour(juce::Label::textColourId, kPrimaryText);
    trainingTimeLabel.setFont(juce::FontOptions(14.0f));
    addAndMakeVisible(trainingTimeLabel);

    trainingTimeSlider.setSliderStyle(juce::Slider::LinearHorizontal);
    trainingTimeSlider.setTextBoxStyle(juce::Slider::TextBoxRight, false, 64, 24);
    trainingTimeSlider.setNumDecimalPlacesToDisplay(0);
    trainingTimeSlider.setTextValueSuffix(" s");
    trainingTimeSlider.setColour(juce::Slider::textBoxTextColourId, kPrimaryText);
    trainingTimeSlider.setColour(juce::Slider::textBoxOutlineColourId,
                                 juce::Colours::transparentBlack);
    trainingTimeSlider.setColour(juce::Slider::trackColourId, kAccentBlue.withAlpha(0.22f));
    trainingTimeSlider.setColour(juce::Slider::thumbColourId, kAccentBlueDark);
    addAndMakeVisible(trainingTimeSlider);

    frequencyLabel.setText("Reference Frequency", juce::dontSendNotification);
    frequencyLabel.setColour(juce::Label::textColourId, kPrimaryText);
    addAndMakeVisible(frequencyLabel);

    referenceFrequencySlider.setSliderStyle(juce::Slider::RotaryHorizontalVerticalDrag);
    referenceFrequencySlider.setTextBoxStyle(juce::Slider::TextBoxBelow, false, 120, 24);
    referenceFrequencySlider.setSkewFactorFromMidPoint(440.0);
    referenceFrequencySlider.setColour(juce::Slider::textBoxTextColourId, kPrimaryText);
    referenceFrequencySlider.setColour(juce::Slider::textBoxOutlineColourId,
                                       juce::Colours::transparentBlack);
    referenceFrequencySlider.setColour(juce::Slider::rotarySliderFillColourId, kAccentBlue);
    referenceFrequencySlider.setColour(juce::Slider::rotarySliderOutlineColourId,
                                       kAccentBlue.withAlpha(0.22f));
    referenceFrequencySlider.setColour(juce::Slider::thumbColourId, kAccentBlueDark);
    addAndMakeVisible(referenceFrequencySlider);

    oscAPitchTrackToggle.setColour(juce::ToggleButton::textColourId, kPrimaryText);
    oscBPitchTrackToggle.setColour(juce::ToggleButton::textColourId, kPrimaryText);
    addAndMakeVisible(oscAPitchTrackToggle);
    addAndMakeVisible(oscBPitchTrackToggle);

    configureEffectSlider(preHighPassCutoffSlider, preHighPassCutoffLabel, "Pre HP Cutoff");
    configureEffectSlider(preHighPassSlopeSlider, preHighPassSlopeLabel, "Pre HP Slope");
    configureEffectSlider(preLowPassCutoffSlider, preLowPassCutoffLabel, "Pre LP Cutoff");
    configureEffectSlider(preLowPassSlopeSlider, preLowPassSlopeLabel, "Pre LP Slope");
    configureEffectSlider(highPassCutoffSlider, highPassCutoffLabel, "HP Cutoff");
    configureEffectSlider(highPassSlopeSlider, highPassSlopeLabel, "HP Slope");
    configureEffectSlider(lowPassCutoffSlider, lowPassCutoffLabel, "LP Cutoff");
    configureEffectSlider(lowPassSlopeSlider, lowPassSlopeLabel, "LP Slope");
    configureEffectSlider(distortionSlider, distortionLabel, "Distortion");

    keyboardLabel.setText("Pitch Tracking", juce::dontSendNotification);
    keyboardLabel.setColour(juce::Label::textColourId, kPrimaryText);
    keyboardLabel.setFont(juce::FontOptions(15.0f, juce::Font::bold));
    addAndMakeVisible(keyboardLabel);

    keyboardComponent.setAvailableRange(24, 96);
    keyboardComponent.setLowestVisibleKey(48);
    keyboardComponent.setKeyWidth(26.0f);
    keyboardComponent.setWantsKeyboardFocus(true);
    addAndMakeVisible(keyboardComponent);

    trainingTimeAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        audioProcessor.getParameters(), adaptive_echo::plugin_parameters::kTrainingTimeSecondsId,
        trainingTimeSlider);
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
    lowPassSlopeAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        audioProcessor.getParameters(), adaptive_echo::plugin_parameters::kLowPassSlopeId,
        lowPassSlopeSlider);
    distortionAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
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
    g.fillAll(kBackground);

    auto bounds = getLocalBounds().reduced(20);
    auto hero = bounds.removeFromTop(58);
    auto trainingArea = bounds.removeFromTop(214);
    bounds.removeFromTop(10);
    auto controlsArea = bounds.removeFromTop(210);
    bounds.removeFromTop(10);
    auto keyboardArea = bounds;

    auto leftControls = controlsArea.removeFromLeft((controlsArea.getWidth() - 12) / 2);
    controlsArea.removeFromLeft(12);
    auto rightControls = controlsArea;

    drawPanel(g, trainingArea.toFloat());
    drawPanel(g, leftControls.toFloat());
    drawPanel(g, rightControls.toFloat());
    drawPanel(g, keyboardArea.toFloat());

    g.setColour(kPrimaryText);
    g.setFont(juce::Font(juce::FontOptions(34.0f, juce::Font::bold)));
    g.drawText("Adaptive Echo", hero.removeFromTop(38), juce::Justification::centredLeft, false);

    drawPanelHeader(g, trainingArea.reduced(20).removeFromTop(24), "Training");
    drawPanelHeader(g, leftControls.reduced(20).removeFromTop(24), "Source Controls");
    drawPanelHeader(g, rightControls.reduced(20).removeFromTop(24), "Output Controls");
    drawPanelHeader(g, keyboardArea.reduced(20).removeFromTop(24), "Keyboard");
}

void AdaptiveEchoAudioProcessorEditor::resized() {
    auto bounds = getLocalBounds().reduced(20);
    bounds.removeFromTop(58);

    auto trainingArea = bounds.removeFromTop(214).reduced(20);
    bounds.removeFromTop(10);
    auto controlsArea = bounds.removeFromTop(210);
    bounds.removeFromTop(10);
    auto keyboardArea = bounds.reduced(20);

    auto leftControls = controlsArea.removeFromLeft((controlsArea.getWidth() - 12) / 2).reduced(20);
    controlsArea.removeFromLeft(12);
    auto rightControls = controlsArea.reduced(20);

    trainingArea.removeFromTop(26);

    auto actionRow = trainingArea.removeFromTop(40);
    auto actionControls = actionRow.removeFromRight(600);
    trainingTimeLabel.setBounds(actionControls.removeFromLeft(96));
    actionControls.removeFromLeft(8);
    trainingTimeSlider.setBounds(actionControls.removeFromLeft(180));
    actionControls.removeFromLeft(14);
    loadSampleButton.setBounds(actionControls.removeFromLeft(148));
    actionControls.removeFromLeft(14);
    trainButton.setBounds(actionControls.removeFromLeft(140));

    trainingArea.removeFromTop(8);
    samplePathLabel.setBounds(trainingArea.removeFromTop(28));
    trainingArea.removeFromTop(8);
    statusLabel.setBounds(trainingArea.removeFromTop(24));
    trainingArea.removeFromTop(6);
    trainingProgressLabel.setBounds(trainingArea.removeFromTop(20));
    trainingArea.removeFromTop(8);
    trainingProgressBar.setBounds(trainingArea.removeFromTop(24));

    leftControls.removeFromTop(26);
    rightControls.removeFromTop(26);

    auto layoutKnob = [](juce::Rectangle<int> area, juce::Label& label, juce::Slider& slider) {
        label.setBounds(area.removeFromTop(22));
        area.removeFromTop(6);
        slider.setBounds(area);
    };

    auto rowGap = 12;
    auto layoutRow =
        [layoutKnob, rowGap](
            juce::Rectangle<int> row,
            std::initializer_list<std::pair<juce::Label*, juce::Slider*>> controls) mutable {
            const auto count = static_cast<int>(controls.size());
            const auto width = (row.getWidth() - rowGap * (count - 1)) / count;
            for (auto [label, slider] : controls) {
                auto area = row.removeFromLeft(width);
                layoutKnob(area, *label, *slider);
                row.removeFromLeft(rowGap);
            }
        };

    auto leftRow = leftControls.removeFromTop(116);
    layoutRow(leftRow, {{&frequencyLabel, &referenceFrequencySlider},
                        {&preHighPassCutoffLabel, &preHighPassCutoffSlider},
                        {&preHighPassSlopeLabel, &preHighPassSlopeSlider},
                        {&preLowPassCutoffLabel, &preLowPassCutoffSlider},
                        {&preLowPassSlopeLabel, &preLowPassSlopeSlider}});

    auto rightRow = rightControls.removeFromTop(116);
    layoutRow(rightRow, {{&highPassCutoffLabel, &highPassCutoffSlider},
                         {&highPassSlopeLabel, &highPassSlopeSlider},
                         {&lowPassCutoffLabel, &lowPassCutoffSlider},
                         {&lowPassSlopeLabel, &lowPassSlopeSlider},
                         {&distortionLabel, &distortionSlider}});

    keyboardArea.removeFromTop(26);
    auto pitchTrackArea = keyboardArea.removeFromLeft(150);
    keyboardLabel.setBounds(pitchTrackArea.removeFromTop(24));
    pitchTrackArea.removeFromTop(8);
    oscAPitchTrackToggle.setBounds(pitchTrackArea.removeFromTop(24));
    pitchTrackArea.removeFromTop(8);
    oscBPitchTrackToggle.setBounds(pitchTrackArea.removeFromTop(24));
    keyboardArea.removeFromLeft(16);
    keyboardComponent.setBounds(keyboardArea.removeFromTop(222));
}

void AdaptiveEchoAudioProcessorEditor::buttonClicked(juce::Button* button) {
    if (button == &loadSampleButton) {
        fileChooser = std::make_unique<juce::FileChooser>("Choose a sample to train");
        auto flags =
            juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectFiles;
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

void AdaptiveEchoAudioProcessorEditor::timerCallback() { refreshFromProcessor(); }

void AdaptiveEchoAudioProcessorEditor::refreshFromProcessor() {
    const auto samplePath = audioProcessor.getSamplePath();
    samplePathLabel.setText(samplePath.isNotEmpty() ? samplePath : "No sample loaded",
                            juce::dontSendNotification);
    statusLabel.setText(audioProcessor.getStatusText(), juce::dontSendNotification);
    trainingProgressLabel.setText(audioProcessor.getTrainingProgressText(),
                                  juce::dontSendNotification);
    trainingProgressValue = audioProcessor.getTrainingProgress();
    const auto shouldShowProgress = audioProcessor.isTraining() || trainingProgressValue > 0.0 ||
                                    trainingProgressLabel.getText().isNotEmpty();
    trainingProgressLabel.setVisible(shouldShowProgress);
    trainingProgressBar.setVisible(shouldShowProgress);
    trainButton.setEnabled(audioProcessor.canTrain());
    repaint();
}

void AdaptiveEchoAudioProcessorEditor::configureEffectSlider(juce::Slider& slider,
                                                             juce::Label& label,
                                                             const juce::String& text) {
    label.setText(text, juce::dontSendNotification);
    label.setColour(juce::Label::textColourId, kPrimaryText);
    label.setFont(juce::FontOptions(14.0f));
    addAndMakeVisible(label);

    slider.setSliderStyle(juce::Slider::RotaryHorizontalVerticalDrag);
    slider.setTextBoxStyle(juce::Slider::TextBoxBelow, false, 100, 24);
    slider.setRange(0.0, 1.0, 0.001);
    slider.setColour(juce::Slider::textBoxTextColourId, kPrimaryText);
    slider.setColour(juce::Slider::textBoxOutlineColourId, juce::Colours::transparentBlack);
    slider.setColour(juce::Slider::rotarySliderFillColourId, kAccentBlue);
    slider.setColour(juce::Slider::rotarySliderOutlineColourId, kAccentBlue.withAlpha(0.22f));
    slider.setColour(juce::Slider::thumbColourId, kAccentBlueDark);
    addAndMakeVisible(slider);
}
