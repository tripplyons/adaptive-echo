#pragma once

#include <JuceHeader.h>
#include "Oscillator.hpp"

class OscillatorVisualizer : public juce::Component,
                             private juce::Timer
{
public:
    OscillatorVisualizer(WavetableOscillator& oscRef)
        : osc(oscRef)
    {
        samples.resize(numPoints);
        startTimerHz(30);
    }

    void paint(juce::Graphics& g) override
    {
        g.fillAll(juce::Colours::black);
        g.setColour(juce::Colours::white);

        auto r = getLocalBounds().reduced(6);
        auto mid = r.getCentreY();

        juce::Path p;
        p.startNewSubPath(r.getX(), mid);

        const float w = (float) r.getWidth();
        const float dx = w / (float)(numPoints - 1);

        for (size_t i = 0; i < samples.size(); ++i)
        {
            const float x = r.getX() + dx * (float)i;
            const float y = juce::jmap(samples[i], -1.0f, 1.0f,
                                       (float)r.getBottom(), (float)r.getY());
            p.lineTo(x, y);
        }

        g.strokePath(p, juce::PathStrokeType(2.0f));
    }

    void resized() override {}

    void update()
    {
        for (size_t i = 0; i < numPoints; ++i)
        {
            float ph = (float)i / (float)numPoints;
            float rad = ph * juce::MathConstants<float>::twoPi;
            samples[i] = osc.sample(rad);
        }
    }

private:
    void timerCallback() override
    {
        update();
        repaint();
    }

    WavetableOscillator& osc;

    static constexpr size_t numPoints = 512;
    std::vector<float> samples;
};
