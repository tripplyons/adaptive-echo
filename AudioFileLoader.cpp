#include "AudioFileLoader.hpp"

std::vector<std::vector<double>>
AudioFileLoader::loadAudioFile (const juce::File& file)
{
    juce::AudioFormatManager formatManager;
    formatManager.registerBasicFormats();

    std::unique_ptr<juce::AudioFormatReader> reader (
        formatManager.createReaderFor (file)
    );

    if (! reader)
        return {}; // failed to read

    const int numChannels = reader->numChannels;
    const int64 numSamples = reader->lengthInSamples;

    juce::AudioBuffer<float> buffer ((int) numChannels, (int) numSamples);
    reader->read (&buffer, 0, (int) numSamples, 0, true, true);

    std::vector<std::vector<double>> result;
    result.resize (numChannels);

    for (int ch = 0; ch < numChannels; ++ch)
    {
        result[ch].resize (numSamples);
        const float* data = buffer.getReadPointer (ch);

        for (int i = 0; i < numSamples; ++i)
            result[ch][i] = static_cast<double> (data[i]);
    }

    return result;
}
