#include "PluginProcessor.h"

#include <algorithm>
#include <cmath>
#include <utility>

#include "PluginEditor.h"
#include "adaptive_echo/constants.hpp"
#include "adaptive_echo/engine.hpp"
#include "adaptive_echo/filter.hpp"

namespace {
constexpr auto kSamplePathProperty = "samplePath";
constexpr auto kTrainedSettingsProperty = "trainedSettings";
constexpr auto kDefaultReferenceFrequencyHz = 440.0f;
constexpr bool kDefaultOscAPitchTrack = true;
constexpr bool kDefaultOscBPitchTrack = true;
constexpr float kDefaultBypassHighPassCutoff = 0.0f;
constexpr float kDefaultBypassHighPassSlope = 0.0f;
constexpr float kDefaultBypassLowPassCutoff = 1.0f;
constexpr float kDefaultBypassLowPassSlope = 0.0f;
constexpr float kDefaultBypassDistortion = 0.0f;

float getNormalizedParameterValue(juce::AudioProcessorValueTreeState& parameters,
                                  const juce::StringRef parameterId, float fallbackValue) {
    if (auto* parameter = parameters.getRawParameterValue(parameterId)) {
        return parameter->load();
    }
    return fallbackValue;
}

bool getBoolParameterValue(juce::AudioProcessorValueTreeState& parameters,
                           const juce::StringRef parameterId, bool fallbackValue) {
    if (auto* parameter = parameters.getRawParameterValue(parameterId)) {
        return parameter->load() >= 0.5f;
    }
    return fallbackValue;
}

void setNormalizedParameterValue(juce::AudioProcessorValueTreeState& parameters,
                                 const juce::StringRef parameterId, float value) {
    if (auto* parameter = parameters.getParameter(parameterId)) {
        parameter->setValueNotifyingHost(std::clamp(value, 0.0f, 1.0f));
    }
}

void disableSynthEffects(std::vector<float>& settings) {
    if (settings.size() < adaptive_echo::constants::NUM_SETTINGS) {
        settings.resize(adaptive_echo::constants::NUM_SETTINGS, 0.5f);
    }

    settings[adaptive_echo::constants::HIGH_PASS_CUTOFF_INDEX] = kDefaultBypassHighPassCutoff;
    settings[adaptive_echo::constants::HIGH_PASS_SLOPE_INDEX] = kDefaultBypassHighPassSlope;
    settings[adaptive_echo::constants::LOW_PASS_CUTOFF_INDEX] = kDefaultBypassLowPassCutoff;
    settings[adaptive_echo::constants::LOW_PASS_SLOPE_INDEX] = kDefaultBypassLowPassSlope;
    settings[adaptive_echo::constants::DISTORTION_INDEX] = kDefaultBypassDistortion;
}

adaptive_echo::FilterParameters<float> makeFilterParameters(juce::AudioProcessorValueTreeState& parameters,
                                                            const juce::StringRef highPassCutoffId,
                                                            const juce::StringRef highPassSlopeId,
                                                            const juce::StringRef lowPassCutoffId,
                                                            const juce::StringRef lowPassSlopeId) {
    std::vector<float> normalized(4, 0.0f);
    normalized[0] = getNormalizedParameterValue(parameters, highPassCutoffId,
                                                kDefaultBypassHighPassCutoff);
    normalized[1] =
        getNormalizedParameterValue(parameters, highPassSlopeId, kDefaultBypassHighPassSlope);
    normalized[2] = getNormalizedParameterValue(parameters, lowPassCutoffId,
                                                kDefaultBypassLowPassCutoff);
    normalized[3] =
        getNormalizedParameterValue(parameters, lowPassSlopeId, kDefaultBypassLowPassSlope);
    return adaptive_echo::mapFilterParameters(normalized, 0);
}

float getDistortionAmount(juce::AudioProcessorValueTreeState& parameters) {
    return getNormalizedParameterValue(parameters, adaptive_echo::plugin_parameters::kDistortionAmountId,
                                       kDefaultBypassDistortion);
}
}  // namespace

AdaptiveEchoAudioProcessor::AdaptiveEchoAudioProcessor()
    : AudioProcessor(BusesProperties().withOutput("Output", juce::AudioChannelSet::stereo(), true)),
      parameters(*this, nullptr, "PARAMETERS", createParameterLayout()),
      trainedSettings(adaptive_echo::default_settings()),
      statusText("Load a sample to train the synthesizer.") {}

AdaptiveEchoAudioProcessor::~AdaptiveEchoAudioProcessor() {
    stopTrainingThread();
}

juce::AudioProcessorValueTreeState::ParameterLayout
AdaptiveEchoAudioProcessor::createParameterLayout() {
    std::vector<std::unique_ptr<juce::RangedAudioParameter>> params;
    auto frequencyRange = juce::NormalisableRange<float>(20.0f, 20000.0f, 0.01f, 0.25f);
    auto normalizedRange = juce::NormalisableRange<float>(0.0f, 1.0f, 0.001f);

    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        adaptive_echo::plugin_parameters::kReferenceFrequencyId, "Reference Frequency",
        frequencyRange, kDefaultReferenceFrequencyHz));
    params.push_back(std::make_unique<juce::AudioParameterBool>(
        adaptive_echo::plugin_parameters::kOscAPitchTrackId, "Osc A Pitch Track",
        kDefaultOscAPitchTrack));
    params.push_back(std::make_unique<juce::AudioParameterBool>(
        adaptive_echo::plugin_parameters::kOscBPitchTrackId, "Osc B Pitch Track",
        kDefaultOscBPitchTrack));
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        adaptive_echo::plugin_parameters::kPreHighPassCutoffId, "Pre High-Pass Cutoff",
        normalizedRange, kDefaultBypassHighPassCutoff));
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        adaptive_echo::plugin_parameters::kPreHighPassSlopeId, "Pre High-Pass Slope",
        normalizedRange, kDefaultBypassHighPassSlope));
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        adaptive_echo::plugin_parameters::kPreLowPassCutoffId, "Pre Low-Pass Cutoff",
        normalizedRange, kDefaultBypassLowPassCutoff));
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        adaptive_echo::plugin_parameters::kPreLowPassSlopeId, "Pre Low-Pass Slope",
        normalizedRange, kDefaultBypassLowPassSlope));
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        adaptive_echo::plugin_parameters::kHighPassCutoffId, "High-Pass Cutoff", normalizedRange,
        kDefaultBypassHighPassCutoff));
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        adaptive_echo::plugin_parameters::kHighPassSlopeId, "High-Pass Slope", normalizedRange,
        kDefaultBypassHighPassSlope));
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        adaptive_echo::plugin_parameters::kLowPassCutoffId, "Low-Pass Cutoff", normalizedRange,
        kDefaultBypassLowPassCutoff));
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        adaptive_echo::plugin_parameters::kLowPassSlopeId, "Low-Pass Slope", normalizedRange,
        kDefaultBypassLowPassSlope));
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        adaptive_echo::plugin_parameters::kDistortionAmountId, "Distortion", normalizedRange,
        kDefaultBypassDistortion));
    return {params.begin(), params.end()};
}

const juce::String AdaptiveEchoAudioProcessor::getName() const {
    return JucePlugin_Name;
}

bool AdaptiveEchoAudioProcessor::acceptsMidi() const {
    return true;
}

bool AdaptiveEchoAudioProcessor::producesMidi() const {
    return false;
}

bool AdaptiveEchoAudioProcessor::isMidiEffect() const {
    return false;
}

double AdaptiveEchoAudioProcessor::getTailLengthSeconds() const {
    return adaptive_echo::constants::NUM_SECONDS;
}

int AdaptiveEchoAudioProcessor::getNumPrograms() {
    return 1;
}

int AdaptiveEchoAudioProcessor::getCurrentProgram() {
    return 0;
}

void AdaptiveEchoAudioProcessor::setCurrentProgram(int) {}

const juce::String AdaptiveEchoAudioProcessor::getProgramName(int) {
    return {};
}

void AdaptiveEchoAudioProcessor::changeProgramName(int, const juce::String&) {}

void AdaptiveEchoAudioProcessor::prepareToPlay(double sampleRate, int) {
    currentSampleRate = sampleRate > 0.0 ? sampleRate : 48000.0;
}

void AdaptiveEchoAudioProcessor::releaseResources() {}

bool AdaptiveEchoAudioProcessor::isBusesLayoutSupported(const BusesLayout& layouts) const {
    const auto& outputLayout = layouts.getMainOutputChannelSet();
    return outputLayout == juce::AudioChannelSet::mono() ||
           outputLayout == juce::AudioChannelSet::stereo();
}

void AdaptiveEchoAudioProcessor::processBlock(juce::AudioBuffer<float>& buffer,
                                              juce::MidiBuffer& midiMessages) {
    juce::ScopedNoDenormals noDenormals;
    keyboardState.processNextMidiBuffer(midiMessages, 0, buffer.getNumSamples(), true);

    buffer.clear();

    for (const auto metadata : midiMessages) {
        const auto message = metadata.getMessage();
        if (message.isNoteOn()) {
            startVoice(message.getNoteNumber(), message.getFloatVelocity());
        }
    }

    midiMessages.clear();
    mixActiveVoices(buffer);
}

juce::AudioProcessorEditor* AdaptiveEchoAudioProcessor::createEditor() {
    return new AdaptiveEchoAudioProcessorEditor(*this);
}

bool AdaptiveEchoAudioProcessor::hasEditor() const {
    return true;
}

void AdaptiveEchoAudioProcessor::getStateInformation(juce::MemoryBlock& destData) {
    const auto state = createStateTree();
    if (auto xml = state.createXml()) {
        copyXmlToBinary(*xml, destData);
    }
}

void AdaptiveEchoAudioProcessor::setStateInformation(const void* data, int sizeInBytes) {
    const auto xmlState = getXmlFromBinary(data, sizeInBytes);
    if (xmlState == nullptr) {
        return;
    }

    restoreStateFromTree(juce::ValueTree::fromXml(*xmlState));
}

bool AdaptiveEchoAudioProcessor::decodeAudioFile(const juce::String& path,
                                                 std::vector<float>& monoSamples,
                                                 double& sampleRate,
                                                 juce::String& errorText) const {
    juce::AudioFormatManager formatManager;
    formatManager.registerBasicFormats();

    const juce::File file(path);
    if (!file.existsAsFile()) {
        errorText = "Sample file not found.";
        return false;
    }

    std::unique_ptr<juce::AudioFormatReader> reader(formatManager.createReaderFor(file));
    if (reader == nullptr) {
        errorText = "Unable to decode the selected audio file.";
        return false;
    }

    juce::AudioBuffer<float> buffer(static_cast<int>(reader->numChannels),
                                    static_cast<int>(reader->lengthInSamples));
    if (!reader->read(&buffer, 0, static_cast<int>(reader->lengthInSamples), 0, true, true)) {
        errorText = "Unable to read audio samples from the selected file.";
        return false;
    }

    monoSamples.assign(static_cast<size_t>(reader->lengthInSamples), 0.0f);
    for (int sample = 0; sample < buffer.getNumSamples(); ++sample) {
        float mixed = 0.0f;
        for (int channel = 0; channel < buffer.getNumChannels(); ++channel) {
            mixed += buffer.getSample(channel, sample);
        }
        monoSamples[static_cast<size_t>(sample)] = mixed / static_cast<float>(buffer.getNumChannels());
    }

    sampleRate = reader->sampleRate;
    return true;
}

bool AdaptiveEchoAudioProcessor::loadSampleFromPath(const juce::String& path) {
    std::vector<float> monoSamples;
    double sourceSampleRate = 0.0;
    juce::String errorText;
    if (!decodeAudioFile(path, monoSamples, sourceSampleRate, errorText)) {
        setStatusText(errorText);
        sendChangeMessage();
        return false;
    }

    auto processed = adaptive_echo::preprocess_target_audio(
        monoSamples, sourceSampleRate, adaptive_echo::constants::NUM_SAMPLES);

    {
        const std::lock_guard<std::mutex> lock(stateMutex);
        loadedSample = std::move(processed);
        samplePath = path;
        statusText = "Sample loaded. Ready to train.";
    }
    resetTrainingProgress();

    sendChangeMessage();
    updateHostDisplay();
    return true;
}

void AdaptiveEchoAudioProcessor::beginTraining() {
    if (!canTrain() || trainingActive.exchange(true)) {
        return;
    }

    stopTrainingThread();

    std::vector<float> sampleSnapshot;
    {
        const std::lock_guard<std::mutex> lock(stateMutex);
        sampleSnapshot = loadedSample;
        statusText = "Training in progress...";
    }
    resetTrainingProgress();
    sendChangeMessage();

    trainingThread = std::thread([this, sample = std::move(sampleSnapshot)]() mutable {
        auto result = adaptive_echo::train_synth(
            sample, trainingPopulationSize, trainingInitialSigma, trainingTimeLimitSeconds, false,
            [this](const adaptive_echo::TrainingProgress& progress) {
                trainingGeneration = progress.generation;
                trainingEvalCount = progress.eval_count;
                trainingBestLoss = progress.best_loss;
                trainingSigma = progress.sigma;
                trainingElapsedSeconds = progress.elapsed_seconds;
                const auto normalizedProgress =
                    trainingTimeLimitSeconds > 0.0f
                        ? std::clamp(progress.elapsed_seconds / trainingTimeLimitSeconds, 0.0f,
                                     1.0f)
                        : 0.0f;
                trainingProgress = normalizedProgress;
            });
        const auto bestSettings = result.best_settings;
        {
            const std::lock_guard<std::mutex> lock(stateMutex);
            trainedSettings = bestSettings;
            statusText = juce::String::formatted(
                "Training complete. Best loss %.4f after %d generations.", result.best_loss,
                result.iterations_completed);
        }
        trainingGeneration = result.iterations_completed;
        trainingEvalCount = result.final_eval_count;
        trainingBestLoss = result.best_loss;
        trainingSigma = result.final_sigma;
        trainingProgress = 1.0f;
        trainingActive = false;

        const juce::MessageManagerLock messageManagerLock;
        if (messageManagerLock.lockWasGained()) {
            syncEffectParametersFromSettings(bestSettings);
            sendChangeMessage();
            updateHostDisplay();
        }
    });
}

void AdaptiveEchoAudioProcessor::syncEffectParametersFromSettings(
    const std::vector<float>& settings) {
    if (settings.size() < adaptive_echo::constants::NUM_SETTINGS) {
        return;
    }

    setNormalizedParameterValue(parameters, adaptive_echo::plugin_parameters::kPreHighPassCutoffId,
                                kDefaultBypassHighPassCutoff);
    setNormalizedParameterValue(parameters, adaptive_echo::plugin_parameters::kPreHighPassSlopeId,
                                kDefaultBypassHighPassSlope);
    setNormalizedParameterValue(parameters, adaptive_echo::plugin_parameters::kPreLowPassCutoffId,
                                kDefaultBypassLowPassCutoff);
    setNormalizedParameterValue(parameters, adaptive_echo::plugin_parameters::kPreLowPassSlopeId,
                                kDefaultBypassLowPassSlope);
    setNormalizedParameterValue(parameters, adaptive_echo::plugin_parameters::kHighPassCutoffId,
                                settings[adaptive_echo::constants::HIGH_PASS_CUTOFF_INDEX]);
    setNormalizedParameterValue(parameters, adaptive_echo::plugin_parameters::kHighPassSlopeId,
                                settings[adaptive_echo::constants::HIGH_PASS_SLOPE_INDEX]);
    setNormalizedParameterValue(parameters, adaptive_echo::plugin_parameters::kLowPassCutoffId,
                                settings[adaptive_echo::constants::LOW_PASS_CUTOFF_INDEX]);
    setNormalizedParameterValue(parameters, adaptive_echo::plugin_parameters::kLowPassSlopeId,
                                settings[adaptive_echo::constants::LOW_PASS_SLOPE_INDEX]);
    setNormalizedParameterValue(parameters, adaptive_echo::plugin_parameters::kDistortionAmountId,
                                settings[adaptive_echo::constants::DISTORTION_INDEX]);
}

bool AdaptiveEchoAudioProcessor::canTrain() const {
    const std::lock_guard<std::mutex> lock(stateMutex);
    return !trainingActive.load() && !loadedSample.empty();
}

bool AdaptiveEchoAudioProcessor::isTraining() const {
    return trainingActive.load();
}

juce::String AdaptiveEchoAudioProcessor::getSamplePath() const {
    const std::lock_guard<std::mutex> lock(stateMutex);
    return samplePath;
}

juce::String AdaptiveEchoAudioProcessor::getStatusText() const {
    const std::lock_guard<std::mutex> lock(stateMutex);
    return statusText;
}

double AdaptiveEchoAudioProcessor::getTrainingProgress() const {
    return static_cast<double>(trainingProgress.load());
}

juce::String AdaptiveEchoAudioProcessor::getTrainingProgressText() const {
    const auto generation = trainingGeneration.load();
    const auto evalCount = trainingEvalCount.load();
    const auto bestLoss = trainingBestLoss.load();
    const auto sigma = trainingSigma.load();
    const auto elapsedSeconds = trainingElapsedSeconds.load();
    const auto trainingNow = isTraining();

    if (generation <= 0 && evalCount <= 0 && !trainingNow) {
        return {};
    }

    if (trainingNow) {
        return juce::String::formatted(
            "Generation %d  |  Loss %.4f  |  Sigma %.3f  |  Evals %d  |  %.1fs / %.0fs",
            generation, bestLoss, sigma, evalCount, elapsedSeconds, trainingTimeLimitSeconds);
    }

    return juce::String::formatted(
        "Completed in %.1fs  |  Generations %d  |  Loss %.4f  |  Sigma %.3f  |  Evals %d",
        elapsedSeconds, generation, bestLoss, sigma, evalCount);
}

juce::MidiKeyboardState& AdaptiveEchoAudioProcessor::getKeyboardState() {
    return keyboardState;
}

juce::AudioProcessorValueTreeState& AdaptiveEchoAudioProcessor::getParameters() {
    return parameters;
}

void AdaptiveEchoAudioProcessor::stopTrainingThread() {
    if (trainingThread.joinable()) {
        trainingThread.join();
    }
}

void AdaptiveEchoAudioProcessor::setStatusText(const juce::String& newStatus) {
    const std::lock_guard<std::mutex> lock(stateMutex);
    statusText = newStatus;
}

void AdaptiveEchoAudioProcessor::resetTrainingProgress() {
    trainingGeneration = 0;
    trainingEvalCount = 0;
    trainingBestLoss = 0.0f;
    trainingSigma = trainingInitialSigma;
    trainingElapsedSeconds = 0.0f;
    trainingProgress = 0.0f;
}

std::vector<float> AdaptiveEchoAudioProcessor::getCurrentSettingsSnapshot() const {
    const std::lock_guard<std::mutex> lock(stateMutex);
    return trainedSettings;
}

float AdaptiveEchoAudioProcessor::getReferenceFrequency() const {
    if (auto* parameter = parameters.getRawParameterValue(
            adaptive_echo::plugin_parameters::kReferenceFrequencyId)) {
        return parameter->load();
    }
    return kDefaultReferenceFrequencyHz;
}

void AdaptiveEchoAudioProcessor::startVoice(int midiNote, float velocity) {
    auto settings = getCurrentSettingsSnapshot();
    disableSynthEffects(settings);
    const bool pitchTrackOscA = getBoolParameterValue(
        parameters, adaptive_echo::plugin_parameters::kOscAPitchTrackId, kDefaultOscAPitchTrack);
    const bool pitchTrackOscB = getBoolParameterValue(
        parameters, adaptive_echo::plugin_parameters::kOscBPitchTrackId, kDefaultOscBPitchTrack);

    auto rendered =
        adaptive_echo::render_note_audio(settings, getReferenceFrequency(), midiNote,
                                         currentSampleRate, pitchTrackOscA, pitchTrackOscB);
    auto preDistortionFilters = makeFilterParameters(
        parameters, adaptive_echo::plugin_parameters::kPreHighPassCutoffId,
        adaptive_echo::plugin_parameters::kPreHighPassSlopeId,
        adaptive_echo::plugin_parameters::kPreLowPassCutoffId,
        adaptive_echo::plugin_parameters::kPreLowPassSlopeId);
    auto postDistortionFilters = makeFilterParameters(
        parameters, adaptive_echo::plugin_parameters::kHighPassCutoffId,
        adaptive_echo::plugin_parameters::kHighPassSlopeId,
        adaptive_echo::plugin_parameters::kLowPassCutoffId,
        adaptive_echo::plugin_parameters::kLowPassSlopeId);

    adaptive_echo::applyFilters(preDistortionFilters, rendered, static_cast<float>(currentSampleRate));
    adaptive_echo::applyDistortion(getDistortionAmount(parameters), rendered);
    adaptive_echo::applyFilters(postDistortionFilters, rendered, static_cast<float>(currentSampleRate));

    ActiveVoice voice;
    voice.samples = std::move(rendered);
    voice.velocity = velocity;
    voice.order = ++voiceCounter;

    if (activeVoices.size() >= static_cast<size_t>(maxPolyphony)) {
        auto oldest = std::min_element(activeVoices.begin(), activeVoices.end(),
                                       [](const ActiveVoice& left, const ActiveVoice& right) {
                                           return left.order < right.order;
                                       });
        if (oldest != activeVoices.end()) {
            *oldest = std::move(voice);
            return;
        }
    }

    activeVoices.push_back(std::move(voice));
}

void AdaptiveEchoAudioProcessor::mixActiveVoices(juce::AudioBuffer<float>& buffer) {
    auto* left = buffer.getWritePointer(0);
    auto* right = buffer.getNumChannels() > 1 ? buffer.getWritePointer(1) : nullptr;
    const auto numSamples = static_cast<size_t>(buffer.getNumSamples());

    for (auto voice = activeVoices.begin(); voice != activeVoices.end();) {
        const auto remaining = voice->samples.size() - voice->position;
        const auto blockSamples = std::min(numSamples, remaining);

        for (size_t i = 0; i < blockSamples; ++i) {
            const auto sample = voice->samples[voice->position + i] * voice->velocity;
            left[i] += sample;
            if (right != nullptr) {
                right[i] += sample;
            }
        }

        voice->position += blockSamples;
        if (voice->position >= voice->samples.size()) {
            voice = activeVoices.erase(voice);
        } else {
            ++voice;
        }
    }
}

juce::ValueTree AdaptiveEchoAudioProcessor::createStateTree() {
    auto state = parameters.copyState();

    const std::lock_guard<std::mutex> lock(stateMutex);
    state.setProperty(kSamplePathProperty, samplePath, nullptr);
    state.setProperty(kTrainedSettingsProperty,
                      juce::String(adaptive_echo::serialize_settings(trainedSettings)), nullptr);
    return state;
}

void AdaptiveEchoAudioProcessor::restoreStateFromTree(const juce::ValueTree& stateTree) {
    if (!stateTree.isValid()) {
        return;
    }

    parameters.replaceState(stateTree);

    const auto restoredPath = stateTree.getProperty(kSamplePathProperty).toString();
    const auto restoredSettings =
        adaptive_echo::deserialize_settings(stateTree.getProperty(kTrainedSettingsProperty).toString().toStdString());

    {
        const std::lock_guard<std::mutex> lock(stateMutex);
        trainedSettings = restoredSettings;
    }

    if (restoredPath.isNotEmpty()) {
        if (!loadSampleFromPath(restoredPath)) {
            setStatusText("Saved sample path could not be reloaded. Trained settings were restored.");
        }
    }

    sendChangeMessage();
}

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter() {
    return new AdaptiveEchoAudioProcessor();
}
