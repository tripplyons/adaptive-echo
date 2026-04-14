#include "PluginEditor.h"

namespace {
const auto kBackground = juce::Colours::white;
const auto kPanelBackground = juce::Colour::fromRGB(246, 249, 255);
const auto kPanelOutline = juce::Colour::fromRGB(214, 222, 235);
const auto kPrimaryText = juce::Colours::black;
const auto kSecondaryText = juce::Colour::fromRGB(78, 86, 99);
const auto kAccentBlue = juce::Colour::fromRGB(34, 110, 255);
const auto kAccentBlueDark = juce::Colour::fromRGB(21, 79, 201);
const auto kAccentBlueSoft = juce::Colour::fromRGB(228, 238, 255);

void drawPanel(juce::Graphics& g, juce::Rectangle<float> area) {
    g.setColour(kPanelBackground);
    g.fillRoundedRectangle(area, 18.0f);
    g.setColour(kPanelOutline);
    g.drawRoundedRectangle(area.reduced(0.5f), 18.0f, 1.0f);
}

void drawPanelHeader(juce::Graphics& g, juce::Rectangle<int> area, const juce::String& title,
                     const juce::String& subtitle) {
    g.setColour(kPrimaryText);
    g.setFont(juce::Font(juce::FontOptions(18.0f, juce::Font::bold)));
    g.drawText(title, area.removeFromTop(24), juce::Justification::centredLeft, false);

    g.setColour(kSecondaryText);
    g.setFont(13.0f);
    g.drawText(subtitle, area.removeFromTop(18), juce::Justification::centredLeft, false);
}
}  // namespace

AdaptiveEchoAudioProcessorEditor::AdaptiveEchoAudioProcessorEditor(
    AdaptiveEchoAudioProcessor& processorToUse)
    : AudioProcessorEditor(&processorToUse),
      audioProcessor(processorToUse),
      keyboardComponent(audioProcessor.getKeyboardState(),
                        juce::MidiKeyboardComponent::horizontalKeyboard) {
    setOpaque(true);
    setSize(1180, 780);
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
    trainingProgressLabel.setColour(juce::Label::textColourId, kSecondaryText);
    trainingProgressLabel.setFont(juce::FontOptions(13.0f));
    addAndMakeVisible(trainingProgressLabel);

    trainingProgressBar.setTextToDisplay({});
    trainingProgressBar.setColour(juce::ProgressBar::foregroundColourId, kAccentBlue);
    trainingProgressBar.setColour(juce::ProgressBar::backgroundColourId,
                                  kAccentBlue.withAlpha(0.12f));
    addAndMakeVisible(trainingProgressBar);

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

    keyboardLabel.setText("On-Screen MIDI Keyboard", juce::dontSendNotification);
    keyboardLabel.setColour(juce::Label::textColourId, kPrimaryText);
    keyboardLabel.setFont(juce::FontOptions(15.0f, juce::Font::bold));
    addAndMakeVisible(keyboardLabel);

    keyboardComponent.setAvailableRange(24, 96);
    keyboardComponent.setLowestVisibleKey(48);
    keyboardComponent.setKeyWidth(26.0f);
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
    g.fillAll(kBackground);

    auto bounds = getLocalBounds().reduced(22);
    auto hero = bounds.removeFromTop(72);
    auto trainingArea = bounds.removeFromTop(162);
    bounds.removeFromTop(16);
    auto controlsArea = bounds.removeFromTop(308);
    bounds.removeFromTop(16);
    auto keyboardArea = bounds;

    auto leftControls = controlsArea.removeFromLeft((controlsArea.getWidth() - 16) / 2);
    controlsArea.removeFromLeft(16);
    auto rightControls = controlsArea;

    drawPanel(g, trainingArea.toFloat());
    drawPanel(g, leftControls.toFloat());
    drawPanel(g, rightControls.toFloat());
    drawPanel(g, keyboardArea.toFloat());

    g.setColour(kPrimaryText);
    g.setFont(juce::Font(juce::FontOptions(34.0f, juce::Font::bold)));
    g.drawText("Adaptive Echo", hero.removeFromTop(40), juce::Justification::centredLeft, false);

    g.setFont(15.0f);
    g.setColour(kSecondaryText);
    g.drawText("White canvas, clearer grouping, and a larger keyboard for faster auditioning.",
               hero, juce::Justification::centredLeft, false);

    drawPanelHeader(g, trainingArea.reduced(22).removeFromTop(42), "Training",
                    "Load a sample, fit the synth, and monitor progress without hunting for controls.");
    drawPanelHeader(g, leftControls.reduced(22).removeFromTop(42), "Source Controls",
                    "Reference pitch and pre-filter shaping.");
    drawPanelHeader(g, rightControls.reduced(22).removeFromTop(42), "Output Controls",
                    "Final tonal shaping and distortion.");
    drawPanelHeader(g, keyboardArea.reduced(22).removeFromTop(42), "Keyboard",
                    "Large performance area for auditioning the trained patch.");
}

void AdaptiveEchoAudioProcessorEditor::resized() {
    auto bounds = getLocalBounds().reduced(22);
    bounds.removeFromTop(72);

    auto trainingArea = bounds.removeFromTop(162).reduced(22);
    bounds.removeFromTop(16);
    auto controlsArea = bounds.removeFromTop(308);
    bounds.removeFromTop(16);
    auto keyboardArea = bounds.reduced(22);

    auto leftControls = controlsArea.removeFromLeft((controlsArea.getWidth() - 16) / 2).reduced(22);
    controlsArea.removeFromLeft(16);
    auto rightControls = controlsArea.reduced(22);

    auto actionRow = trainingArea.removeFromTop(40);
    auto actionButtons = actionRow.removeFromRight(310);
    loadSampleButton.setBounds(actionButtons.removeFromLeft(148));
    actionButtons.removeFromLeft(14);
    trainButton.setBounds(actionButtons.removeFromLeft(148));

    trainingArea.removeFromTop(12);
    samplePathLabel.setBounds(trainingArea.removeFromTop(30));
    trainingArea.removeFromTop(10);
    statusLabel.setBounds(trainingArea.removeFromTop(26));
    trainingArea.removeFromTop(8);
    trainingProgressLabel.setBounds(trainingArea.removeFromTop(22));
    trainingArea.removeFromTop(10);
    trainingProgressBar.setBounds(trainingArea.removeFromTop(20));
    trainingArea.removeFromTop(14);
    auto toggleRow = trainingArea.removeFromTop(28);
    oscAPitchTrackToggle.setBounds(toggleRow.removeFromLeft(190));
    toggleRow.removeFromLeft(18);
    oscBPitchTrackToggle.setBounds(toggleRow.removeFromLeft(190));

    leftControls.removeFromTop(54);
    rightControls.removeFromTop(54);

    auto layoutKnob = [](juce::Rectangle<int> area, juce::Label& label, juce::Slider& slider) {
        label.setBounds(area.removeFromTop(22));
        area.removeFromTop(6);
        slider.setBounds(area);
    };

    auto rowGap = 16;
    auto layoutRow = [layoutKnob, rowGap](juce::Rectangle<int> row,
                                          std::initializer_list<std::pair<juce::Label*, juce::Slider*>>
                                              controls) mutable {
        const auto count = static_cast<int>(controls.size());
        const auto width = (row.getWidth() - rowGap * (count - 1)) / count;
        for (auto [label, slider] : controls) {
            auto area = row.removeFromLeft(width);
            layoutKnob(area, *label, *slider);
            row.removeFromLeft(rowGap);
        }
    };

    auto leftTop = leftControls.removeFromTop(112);
    auto leftBottom = leftControls.removeFromTop(112);
    layoutRow(leftTop,
              {{&frequencyLabel, &referenceFrequencySlider},
               {&preHighPassCutoffLabel, &preHighPassCutoffSlider},
               {&preHighPassSlopeLabel, &preHighPassSlopeSlider}});
    leftControls.removeFromTop(12);
    layoutRow(leftBottom,
              {{&preLowPassCutoffLabel, &preLowPassCutoffSlider},
               {&preLowPassSlopeLabel, &preLowPassSlopeSlider}});

    auto rightTop = rightControls.removeFromTop(112);
    auto rightBottom = rightControls.removeFromTop(112);
    layoutRow(rightTop,
              {{&highPassCutoffLabel, &highPassCutoffSlider},
               {&highPassSlopeLabel, &highPassSlopeSlider},
               {&lowPassCutoffLabel, &lowPassCutoffSlider}});
    rightControls.removeFromTop(12);
    layoutRow(rightBottom,
              {{&lowPassSlopeLabel, &lowPassSlopeSlider},
               {&distortionLabel, &distortionSlider}});

    keyboardArea.removeFromTop(54);
    keyboardLabel.setBounds(keyboardArea.removeFromTop(24));
    keyboardArea.removeFromTop(12);
    keyboardComponent.setBounds(keyboardArea.removeFromTop(186));
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
