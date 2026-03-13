#include "PluginProcessor.h"

#include <algorithm>
#include <utility>

#include "PluginEditor.h"
#include "adaptive_echo/constants.hpp"
#include "adaptive_echo/engine.hpp"

namespace {
constexpr auto kReferenceFrequencyId = "referenceFrequencyHz";
constexpr auto kSamplePathProperty = "samplePath";
constexpr auto kTrainedSettingsProperty = "trainedSettings";
constexpr auto kDefaultReferenceFrequencyHz = 440.0f;
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
    auto frequencyRange =
        juce::NormalisableRange<float>(20.0f, 20000.0f, 0.01f, 0.25f);
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        kReferenceFrequencyId, "Reference Frequency", frequencyRange, kDefaultReferenceFrequencyHz));
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
    sendChangeMessage();

    trainingThread = std::thread([this, sample = std::move(sampleSnapshot)]() mutable {
        auto result = adaptive_echo::train_synth(sample, 32, 3.0f, 60.0f, false);
        {
            const std::lock_guard<std::mutex> lock(stateMutex);
            trainedSettings = std::move(result.best_settings);
            statusText = "Training complete.";
        }
        trainingActive = false;
        sendChangeMessage();
        updateHostDisplay();
    });
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

std::vector<float> AdaptiveEchoAudioProcessor::getCurrentSettingsSnapshot() const {
    const std::lock_guard<std::mutex> lock(stateMutex);
    return trainedSettings;
}

float AdaptiveEchoAudioProcessor::getReferenceFrequency() const {
    if (auto* parameter = parameters.getRawParameterValue(kReferenceFrequencyId)) {
        return parameter->load();
    }
    return kDefaultReferenceFrequencyHz;
}

void AdaptiveEchoAudioProcessor::startVoice(int midiNote, float velocity) {
    auto settings = getCurrentSettingsSnapshot();
    auto rendered =
        adaptive_echo::render_note_audio(settings, getReferenceFrequency(), midiNote, currentSampleRate);

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
