#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_audio_utils/juce_audio_utils.h>

#include "adaptive_echo/engine.hpp"

namespace adaptive_echo::plugin_parameters {
inline constexpr auto kReferenceFrequencyId = "referenceFrequencyHz";
inline constexpr auto kOscAPitchTrackId = "oscAPitchTrack";
inline constexpr auto kOscBPitchTrackId = "oscBPitchTrack";
inline constexpr auto kPreHighPassCutoffId = "preHighPassCutoff";
inline constexpr auto kPreHighPassSlopeId = "preHighPassSlope";
inline constexpr auto kPreLowPassCutoffId = "preLowPassCutoff";
inline constexpr auto kPreLowPassSlopeId = "preLowPassSlope";
inline constexpr auto kHighPassCutoffId = "highPassCutoff";
inline constexpr auto kHighPassSlopeId = "highPassSlope";
inline constexpr auto kLowPassCutoffId = "lowPassCutoff";
inline constexpr auto kLowPassSlopeId = "lowPassSlope";
inline constexpr auto kDistortionAmountId = "distortionAmount";
}

class AdaptiveEchoAudioProcessor final : public juce::AudioProcessor,
                                         public juce::ChangeBroadcaster {
public:
    AdaptiveEchoAudioProcessor();
    ~AdaptiveEchoAudioProcessor() override;

    void prepareToPlay(double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;
    bool isBusesLayoutSupported(const BusesLayout& layouts) const override;
    void processBlock(juce::AudioBuffer<float>&, juce::MidiBuffer&) override;

    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override;

    const juce::String getName() const override;
    bool acceptsMidi() const override;
    bool producesMidi() const override;
    bool isMidiEffect() const override;
    double getTailLengthSeconds() const override;

    int getNumPrograms() override;
    int getCurrentProgram() override;
    void setCurrentProgram(int index) override;
    const juce::String getProgramName(int index) override;
    void changeProgramName(int index, const juce::String& newName) override;

    void getStateInformation(juce::MemoryBlock& destData) override;
    void setStateInformation(const void* data, int sizeInBytes) override;

    bool loadSampleFromPath(const juce::String& path);
    void beginTraining();
    bool canTrain() const;
    bool isTraining() const;

    juce::String getSamplePath() const;
    juce::String getStatusText() const;
    double getTrainingProgress() const;
    juce::String getTrainingProgressText() const;
    juce::MidiKeyboardState& getKeyboardState();
    juce::AudioProcessorValueTreeState& getParameters();

private:
    struct ActiveVoice {
        std::vector<float> samples;
        size_t position = 0;
        float velocity = 1.0f;
        uint64_t order = 0;
    };

    juce::AudioProcessorValueTreeState parameters;
    juce::MidiKeyboardState keyboardState;

    mutable std::mutex stateMutex;
    std::vector<float> trainedSettings;
    std::vector<float> loadedSample;
    juce::String samplePath;
    juce::String statusText;

    std::vector<ActiveVoice> activeVoices;
    std::atomic<bool> trainingActive { false };
    std::atomic<int> trainingGeneration { 0 };
    std::atomic<int> trainingEvalCount { 0 };
    std::atomic<float> trainingBestLoss { 0.0f };
    std::atomic<float> trainingSigma { 0.0f };
    std::atomic<float> trainingElapsedSeconds { 0.0f };
    std::atomic<float> trainingProgress { 0.0f };
    std::thread trainingThread;
    double currentSampleRate = 48000.0;
    uint64_t voiceCounter = 0;

    static constexpr int maxPolyphony = 16;
    static constexpr int trainingPopulationSize = adaptive_echo::kDefaultCRFMNESPopulationSize;
    static constexpr float trainingInitialSigma = adaptive_echo::kDefaultCRFMNESInitialSigma;
    static constexpr float trainingTimeLimitSeconds = 60.0f;

    static juce::AudioProcessorValueTreeState::ParameterLayout createParameterLayout();
    void stopTrainingThread();
    void syncEffectParametersFromSettings(const std::vector<float>& settings);
    bool decodeAudioFile(const juce::String& path, std::vector<float>& monoSamples,
                         double& sampleRate, juce::String& errorText) const;
    void setStatusText(const juce::String& newStatus);
    void resetTrainingProgress();
    std::vector<float> getCurrentSettingsSnapshot() const;
    float getReferenceFrequency() const;
    void startVoice(int midiNote, float velocity);
    void mixActiveVoices(juce::AudioBuffer<float>& buffer);
    void restoreStateFromTree(const juce::ValueTree& stateTree);
    juce::ValueTree createStateTree();

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(AdaptiveEchoAudioProcessor)
};

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter();
