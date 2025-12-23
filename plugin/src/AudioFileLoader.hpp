#pragma once
#include <juce_audio_formats/juce_audio_formats.h>
#include <vector>

class AudioFileLoader
{
public:
    static std::vector<std::vector<double>>
    loadAudioFile (const juce::File& file);
};
