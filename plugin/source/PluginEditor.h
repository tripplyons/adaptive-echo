#pragma once

#include <juce_gui_extra/juce_gui_extra.h>

#include <memory>

#include "PluginProcessor.h"

class AdaptiveEchoAudioProcessorEditor final
    : public juce::AudioProcessorEditor,
      private juce::Button::Listener,
      private juce::ChangeListener,
      private juce::Timer {
public:
  explicit AdaptiveEchoAudioProcessorEditor(AdaptiveEchoAudioProcessor &);
  ~AdaptiveEchoAudioProcessorEditor() override;

  void paint(juce::Graphics &) override;
  void resized() override;

private:
  AdaptiveEchoAudioProcessor &audioProcessor;
  juce::TextButton loadSampleButton{"Load Sample"};
  juce::TextButton loadPresetButton{"Load Preset"};
  juce::TextButton savePresetButton{"Save Preset"};
  juce::TextButton trainButton{"Train"};
  juce::Label samplePathLabel;
  juce::Label presetPathLabel;
  juce::Label statusLabel;
  juce::Label trainingProgressLabel;
  juce::Label trainingTimeLabel;
  juce::Label frequencyLabel;
  juce::Label preHighPassCutoffLabel;
  juce::Label preHighPassSlopeLabel;
  juce::Label preLowPassCutoffLabel;
  juce::Label preLowPassSlopeLabel;
  juce::Label highPassCutoffLabel;
  juce::Label highPassSlopeLabel;
  juce::Label lowPassCutoffLabel;
  juce::Label lowPassSlopeLabel;
  juce::Label distortionLabel;
  juce::Label keyboardLabel;
  juce::Label behaviorLabel;
  double trainingProgressValue = 0.0;
  juce::ProgressBar trainingProgressBar{trainingProgressValue};
  juce::Slider trainingTimeSlider;
  juce::Slider referenceFrequencySlider;
  juce::ToggleButton oscAPitchTrackToggle{"Track A"};
  juce::ToggleButton oscBPitchTrackToggle{"Track B"};
  juce::ToggleButton singleVoiceToggle{"Single Voice"};
  juce::ToggleButton constantVelocityToggle{"Constant Velocity"};
  juce::Slider preHighPassCutoffSlider;
  juce::Slider preHighPassSlopeSlider;
  juce::Slider preLowPassCutoffSlider;
  juce::Slider preLowPassSlopeSlider;
  juce::Slider highPassCutoffSlider;
  juce::Slider highPassSlopeSlider;
  juce::Slider lowPassCutoffSlider;
  juce::Slider lowPassSlopeSlider;
  juce::Slider distortionSlider;
  juce::MidiKeyboardComponent keyboardComponent;
  std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment>
      trainingTimeAttachment;
  std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment>
      frequencyAttachment;
  std::unique_ptr<juce::AudioProcessorValueTreeState::ButtonAttachment>
      oscAPitchTrackAttachment;
  std::unique_ptr<juce::AudioProcessorValueTreeState::ButtonAttachment>
      oscBPitchTrackAttachment;
  std::unique_ptr<juce::AudioProcessorValueTreeState::ButtonAttachment>
      singleVoiceAttachment;
  std::unique_ptr<juce::AudioProcessorValueTreeState::ButtonAttachment>
      constantVelocityAttachment;
  std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment>
      preHighPassCutoffAttachment;
  std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment>
      preHighPassSlopeAttachment;
  std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment>
      preLowPassCutoffAttachment;
  std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment>
      preLowPassSlopeAttachment;
  std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment>
      highPassCutoffAttachment;
  std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment>
      highPassSlopeAttachment;
  std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment>
      lowPassCutoffAttachment;
  std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment>
      lowPassSlopeAttachment;
  std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment>
      distortionAttachment;
  std::unique_ptr<juce::FileChooser> fileChooser;

  void buttonClicked(juce::Button *button) override;
  void changeListenerCallback(juce::ChangeBroadcaster *source) override;
  void timerCallback() override;
  void refreshFromProcessor();
  void configureEffectSlider(juce::Slider &slider, juce::Label &label,
                             const juce::String &text);
  juce::File getDefaultPresetFile() const;

  JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(AdaptiveEchoAudioProcessorEditor)
};
