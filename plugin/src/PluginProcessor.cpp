#include "PluginProcessor.hpp"
#include "PluginEditor.hpp"

AdaptiveEchoAudioProcessor::AdaptiveEchoAudioProcessor()
    : AudioProcessor(BusesProperties().withOutput(
          "Output", juce::AudioChannelSet::stereo(), true)),
      apvts(*this, nullptr, "PARAMS", createParameterLayout()) {
    volumeSmoothed.reset(currentSampleRate, 0.002); // 2ms smoothing
    noteAmpSmoothed.reset(currentSampleRate, 0.002);
}

juce::AudioProcessorValueTreeState::ParameterLayout
AdaptiveEchoAudioProcessor::createParameterLayout() {
    std::vector<std::unique_ptr<juce::RangedAudioParameter>> params;

    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        "volume", "Volume",
        juce::NormalisableRange<float>(0.0f, 1.0f, 0.0f, 1.0f), 0.5f));

    // ADSR
    juce::NormalisableRange<float> ADSRrange =
        juce::NormalisableRange<float>(ADSR_MIN, ADSR_MAX, 0.01f, 0.3f);
    juce::NormalisableRange<float> CurveRange =
        juce::NormalisableRange<float>(0.0, 1.0, 0.01f, 1.0f);
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        "attack", "Attack", ADSRrange, 0.5f));
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        "decay", "Decay", ADSRrange, 0.5f));
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        "sustain", "Sustain",
        juce::NormalisableRange<float>(0.0f, 1.0f, 0.0f, 1.0f), 0.5f));
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        "release", "Release", ADSRrange, 0.5f));
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        "attackControlX", "AttackControlX", CurveRange, 0.5f));
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        "attackControlY", "AttackControlY", CurveRange, 0.5f));
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        "decayControlX", "DecayControlX", CurveRange, 0.5f));
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        "decayControlY", "DecayControlY", CurveRange, 0.5f));
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        "releaseControlX", "ReleaseControlX", CurveRange, 0.5f));
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        "releaseControlY", "ReleaseControlY", CurveRange, 0.5f));

    return {params.begin(), params.end()};
}

void AdaptiveEchoAudioProcessor::prepareToPlay(double sampleRate,
                                               int /*samplesPerBlock*/) {
    currentSampleRate = sampleRate;

    phase = {0.0, 0.0};

    volumeSmoothed.reset(currentSampleRate, 0.02);
    noteAmpSmoothed.reset(currentSampleRate, 0.02);

    if (auto *volParam = apvts.getRawParameterValue("volume"))
        volumeSmoothed.setCurrentAndTargetValue(volParam->load());
    else
        volumeSmoothed.setCurrentAndTargetValue(0.5f);

    noteAmpSmoothed.setCurrentAndTargetValue(0.0f); // start silent

    // ADSR parameters
    a = d = r = 0.01f;
    s = 1.0f;
    ac = dc = rc = 1.0f;

    env = ADSREnvelope(a, d, s, r, ac, dc, rc,
                       static_cast<int>(currentSampleRate));
    env_ptr = std::make_shared<ADSREnvelope>(env);
    Note activeNote = Note();
}

void AdaptiveEchoAudioProcessor::releaseResources() {}

#ifndef JucePlugin_PreferredChannelConfigurations
bool AdaptiveEchoAudioProcessor::isBusesLayoutSupported(
    const BusesLayout &layouts) const {
    // Only allow mono or stereo outputs
    const auto &mainOut = layouts.getMainOutputChannelSet();
    return mainOut == juce::AudioChannelSet::mono() ||
           mainOut == juce::AudioChannelSet::stereo();
}
#endif

void AdaptiveEchoAudioProcessor::processBlock(juce::AudioBuffer<float> &buffer,
                                              juce::MidiBuffer &midi) {
    juce::ScopedNoDenormals noDenormals;
    const int numSamples = buffer.getNumSamples();
    const int numChans = juce::jmin(2, buffer.getNumChannels());

    // Update envelope params if changed
    float new_a = a;
    float new_d = d;
    float new_s = s;
    float new_r = r;
    float new_ac = ac;
    float new_dc = dc;
    float new_rc = rc;

    if (auto *aParam = apvts.getRawParameterValue("attack"))
        new_a = aParam->load();
    if (auto *dParam = apvts.getRawParameterValue("decay"))
        new_d = dParam->load();
    if (auto *sParam = apvts.getRawParameterValue("sustain"))
        new_s = sParam->load();
    if (auto *rParam = apvts.getRawParameterValue("release"))
        new_r = rParam->load();
    if (auto *acParam = apvts.getRawParameterValue("attackCurve"))
        new_ac = acParam->load();
    if (auto *dcParam = apvts.getRawParameterValue("decayCurve"))
        new_dc = dcParam->load();
    if (auto *rcParam = apvts.getRawParameterValue("releaseCurve"))
        new_rc = rcParam->load();

    // Update ADSR parameters
    if (new_a != a || new_d != d || new_s != s || new_r != r || new_ac != ac || new_dc != dc || new_rc != rc) {
        a = new_a;
        d = new_d;
        s = new_s;
        r = new_r;
        ac = new_ac;
        dc = new_ac;
        rc = new_ac;
        env = ADSREnvelope(a, d, s, r, ac, dc, rc,
                           static_cast<int>(currentSampleRate));
        env_ptr = std::make_shared<ADSREnvelope>(env);
        activeNote.set_env(env_ptr);
    }

    // Update global volume target
    if (auto *volParam = apvts.getRawParameterValue("volume"))
        volumeSmoothed.setTargetValue(volParam->load());

    // Handle MIDI
    midiState.processNextMidiBuffer(midi, 0, numSamples, true);

    for (auto metadata : midi) {
        const auto msg = metadata.getMessage();
        if (msg.isNoteOn()) {
            if (msg.getNoteNumber() != activeNote.num)
                activeNote.num = msg.getNoteNumber();
            activeNote.reset();
            uint8_t vel = (uint8_t)msg.getVelocity();
            const double frequency =
                juce::MidiMessage::getMidiNoteInHertz(activeNote.num);
            phaseInc = juce::MathConstants<double>::twoPi * frequency /
                       currentSampleRate;
            noteAmpSmoothed.setTargetValue((float)vel / 127.0f);
        } else if (msg.isNoteOff()) {
            activeNote.start_release();
        }
    }

    for (int ch = 0; ch < numChans; ++ch) {
        float *out = buffer.getWritePointer(ch);
        double ph = phase[(size_t)ch];

        for (int n = 0; n < numSamples; ++n) {
            if (!activeNote.is_expired()) {
                float noteLevel = noteAmpSmoothed.getNextValue();
                float globalVol = volumeSmoothed.getNextValue();

                float amp = noteLevel * globalVol;
                out[n] = std::sin(ph) * amp;

                ph += phaseInc;
                if (ph >= juce::MathConstants<double>::twoPi)
                    ph -= juce::MathConstants<double>::twoPi;
            } else {
                out[n] = 0.0f;
                (void)volumeSmoothed.getNextValue();
                (void)noteAmpSmoothed.getNextValue();
            }
        }
        phase[(size_t)ch] = ph;
    }

    activeNote.applyEnvelopeToBuffer(buffer, 0, numSamples);

    // Clear any extra channels (e.g., if host created more)
    for (int ch = numChans; ch < buffer.getNumChannels(); ++ch)
        buffer.clear(ch, 0, numSamples);
}

void AdaptiveEchoAudioProcessor::getStateInformation(
    juce::MemoryBlock &destData) {
    if (auto state = apvts.copyState(); state.isValid()) {
        juce::MemoryOutputStream mos(destData, true);
        state.writeToStream(mos);
    }
}

void AdaptiveEchoAudioProcessor::setStateInformation(const void *data,
                                                     int sizeInBytes) {
    juce::ValueTree tree =
        juce::ValueTree::readFromData(data, (size_t)sizeInBytes);
    if (tree.isValid())
        apvts.replaceState(tree);

    // Nudge smoothed value to loaded state
    if (auto *volParam = apvts.getRawParameterValue("volume"))
        volumeSmoothed.setCurrentAndTargetValue(volParam->load());
}

juce::AudioProcessorEditor *AdaptiveEchoAudioProcessor::createEditor() {
    return new AdaptiveEchoAudioProcessorEditor(*this);
}

// This factory must be present in the TU with the processor class.
juce::AudioProcessor *JUCE_CALLTYPE createPluginFilter() {
    return new AdaptiveEchoAudioProcessor();
}