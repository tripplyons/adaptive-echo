#pragma once

#include <JuceHeader.h>

const float ADSR_MIN = 0.001; 
const float ADSR_MAX = 5.0;
const float PI = 3.14;
const juce::Point<float> ONE = juce::Point<float>(1.0, 1.0);

namespace UI
{
    static const juce::Colour background    = juce::Colour(0xff141723);
    static const juce::Colour text          = juce::Colour(0xffe6e6e6);
    static const juce::Colour sliderFill    = juce::Colour(0xff2ca1f9);
    static const juce::Colour sliderOutline = juce::Colour(0xff2d324f);
    static const juce::Colour labelText     = juce::Colour(0xffcccccc);
    static const juce::Colour thumb         = juce::Colour(0xff2ca1f9);
}