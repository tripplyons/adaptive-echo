#pragma once
#include "Constants.hpp"
#include <JuceHeader.h>

class ControlPoint : public juce::Component {
  public:
    // X parameter is normalized by the segment dimensions
    ControlPoint(juce::AudioProcessorValueTreeState &vts,
                 const juce::String &xParam, const juce::String &yParam)
        : apvts(vts), xParamID(xParam), yParamID(yParam) {
        setSize(6, 6);
        segStart = 0.0f;
        segEnd = 1.0f;
    }

    const juce::String &getXParamID() const { return xParamID; }
    const juce::String &getYParamID() const { return yParamID; }

    // Returns the stored segment parameter values
    juce::Point<float> getPos() const {
        float x = 0.0f;
        float y = 0.0f;

        if (auto px = apvts.getRawParameterValue(xParamID))
            x = px->load();

        if (auto py = apvts.getRawParameterValue(yParamID))
            y = py->load();

        return {x, y};
    }

    void setVisualizerSpaceBounds(juce::Rectangle<float> b) {
        visualizerSpace = b;
    }

    // segStart / segEnd are floats from 0.0-1.0 specifying the allowed absolute
    // x-range as a fraction of the full envelope width for this control point's
    // segment
    void setXRange(float startFrac, float endFrac) {
        segStart = juce::jlimit(0.0f, 1.0f, startFrac);
        segEnd = juce::jlimit(0.0f, 1.0f, endFrac);

        // Avoid division-by-zero
        if (segEnd <= segStart + 1e-6f)
            segEnd = segStart + 1e-6f;
    }

    float getSegStart() const { return segStart; }
    float getSegEnd() const { return segEnd; }

  private:
    void mouseDown(const juce::MouseEvent &e) override {
        dragger.startDraggingComponent(this, e);
    }

    void mouseDrag(const juce::MouseEvent &e) override {
        dragger.dragComponent(this, e, nullptr);

        if (visualizerSpace.isEmpty())
            return;

        // Compute normalized coordinates
        auto centre = getBounds().getCentre().toFloat();

        float normX =
            (centre.x - visualizerSpace.getX()) / visualizerSpace.getWidth();
        float normY = 1.0f - ((centre.y - visualizerSpace.getY()) /
                              visualizerSpace.getHeight());

        normX = juce::jlimit(0.0f, 1.0f, normX);
        normY = juce::jlimit(0.0f, 1.0f, normY);

        // Convert the global X position into a normalized value local to the
        // segment
        float segNormX = (normX - segStart) / (segEnd - segStart);
        segNormX = juce::jlimit(0.0f, 1.0f, segNormX);

        if (auto p = apvts.getParameter(xParamID))
            p->setValueNotifyingHost(segNormX);

        if (auto p = apvts.getParameter(yParamID))
            p->setValueNotifyingHost(normY);
    }

    void paint(juce::Graphics &g) override {
        g.setColour(juce::Colours::white);
        g.fillEllipse(0.0f, 0.0f, (float)getWidth(), (float)getHeight());

        g.setColour(juce::Colours::black.withAlpha(0.6f));
        g.drawEllipse(0.0f, 0.0f, (float)getWidth(), (float)getHeight(), 1.0f);
    }

    juce::AudioProcessorValueTreeState &apvts;
    juce::String xParamID, yParamID;
    juce::ComponentDragger dragger;
    juce::Rectangle<float> visualizerSpace;

    float segStart, segEnd;
};

class EnvelopeViewer : public juce::Component, private juce::Timer {
  public:
    EnvelopeViewer(juce::AudioProcessorValueTreeState &state) : apvts(state) {
        attackCtrl = std::make_unique<ControlPoint>(apvts, "attackControlX",
                                                    "attackControlY");
        decayCtrl = std::make_unique<ControlPoint>(apvts, "decayControlX",
                                                   "decayControlY");
        releaseCtrl = std::make_unique<ControlPoint>(apvts, "releaseControlX",
                                                     "releaseControlY");

        addAndMakeVisible(*attackCtrl);
        addAndMakeVisible(*decayCtrl);
        addAndMakeVisible(*releaseCtrl);

        startTimerHz(30);
    }

    void resized() override {
        const float M = 10.0f;
        envelopeBounds = getLocalBounds().toFloat().reduced(M);

        attackCtrl->setVisualizerSpaceBounds(envelopeBounds);
        decayCtrl->setVisualizerSpaceBounds(envelopeBounds);
        releaseCtrl->setVisualizerSpaceBounds(envelopeBounds);

        updateSegmentRanges();
        updatePositions();
    }

    void paint(juce::Graphics &g) override {
        g.fillAll(juce::Colours::black);
        g.setColour(juce::Colours::white);
        g.drawRect(getLocalBounds(), 1);

        float attack = loadParam("attack");
        float decay = loadParam("decay");
        float sustain = loadParam("sustain");
        float release = loadParam("release");

        attack = juce::jmax(0.0f, attack);
        decay = juce::jmax(0.0f, decay);
        release = juce::jmax(0.0f, release);
        sustain = juce::jlimit(0.0f, 1.0f, sustain);

        float maxTime = ADSR_MAX * 3.0f;
        float w = envelopeBounds.getWidth();
        float h = envelopeBounds.getHeight();

        auto t2x = [&](float t) {
            return envelopeBounds.getX() + (t / maxTime) * w;
        };

        auto lv2y = [&](float v) { return envelopeBounds.getBottom() - v * h; };

        // Original parameters for control points are stored as normalized
        // values for each segment
        float ax = loadParam("attackControlX");
        float ay = loadParam("attackControlY");
        float dx = loadParam("decayControlX");
        float dy = loadParam("decayControlY");
        float rx = loadParam("releaseControlX");
        float ry = loadParam("releaseControlY");

        // Compute key times in seconds relative to 0..maxTime.
        float t0 = 0.0f;
        float t1 = attack;
        float t2 = attack + decay;
        float t4 = maxTime;
        float t3 = juce::jmax(
            t2, maxTime - release); // Clamp to keep it a valid function

        // Endpoints
        juce::Point<float> startPt(t2x(t0), envelopeBounds.getBottom());
        juce::Point<float> attackPt(t2x(t1), lv2y(1.0f));
        juce::Point<float> decayPt(t2x(t2), lv2y(sustain));
        juce::Point<float> sustainEndPt(t2x(t3), lv2y(sustain));
        juce::Point<float> releaseEndPt(t2x(t4), envelopeBounds.getBottom());

        // Compute absolute times for control points
        float attackCtrlTime = t0 + ax * (t1 - t0);
        float decayCtrlTime = t1 + dx * juce::jmax(0.0f, (t2 - t1));
        float releaseCtrlTime = t3 + rx * juce::jmax(0.0f, (t4 - t3));

        // Compute control point positions
        juce::Point<float> attackCtrlPt(t2x(attackCtrlTime), lv2y(ay));
        juce::Point<float> decayCtrlPt(t2x(decayCtrlTime), lv2y(dy));
        juce::Point<float> releaseCtrlPt(t2x(releaseCtrlTime), lv2y(ry));

        // Clamp control points to prevent invalid envelopes
        attackCtrlPt.x = juce::jlimit(startPt.x, attackPt.x, attackCtrlPt.x);
        decayCtrlPt.x = juce::jlimit(attackPt.x, decayPt.x, decayCtrlPt.x);
        releaseCtrlPt.x =
            juce::jlimit(sustainEndPt.x, releaseEndPt.x, releaseCtrlPt.x);

        decayCtrlPt.y = juce::jlimit(attackPt.y, sustainEndPt.y, decayCtrlPt.y);
        releaseCtrlPt.y =
            juce::jlimit(sustainEndPt.y, startPt.y, releaseCtrlPt.y);

        // Build actual taper of envelope
        juce::Path p;
        p.startNewSubPath(startPt);
        p.quadraticTo(attackCtrlPt, attackPt);
        p.quadraticTo(decayCtrlPt, decayPt);
        p.lineTo(sustainEndPt);
        p.quadraticTo(releaseCtrlPt, releaseEndPt);

        g.setColour(juce::Colours::cyan);
        g.strokePath(p, juce::PathStrokeType(2.0f));

        g.setColour(juce::Colours::grey.withAlpha(0.4f));
        g.drawVerticalLine((int)attackPt.x, envelopeBounds.getY(),
                           envelopeBounds.getBottom());
        g.drawVerticalLine((int)decayPt.x, envelopeBounds.getY(),
                           envelopeBounds.getBottom());
        g.drawVerticalLine((int)sustainEndPt.x, envelopeBounds.getY(),
                           envelopeBounds.getBottom());

        g.setColour(juce::Colours::white.withAlpha(0.6f));
        g.drawLine(startPt.x, startPt.y, attackCtrlPt.x, attackCtrlPt.y, 1.0f);
        g.drawLine(attackCtrlPt.x, attackCtrlPt.y, attackPt.x, attackPt.y,
                   1.0f);
        g.drawLine(attackPt.x, attackPt.y, decayCtrlPt.x, decayCtrlPt.y, 1.0f);
        g.drawLine(decayCtrlPt.x, decayCtrlPt.y, decayPt.x, decayPt.y, 1.0f);
        g.drawLine(sustainEndPt.x, sustainEndPt.y, releaseCtrlPt.x,
                   releaseCtrlPt.y, 1.0f);
        g.drawLine(releaseCtrlPt.x, releaseCtrlPt.y, releaseEndPt.x,
                   releaseEndPt.y, 1.0f);
    }

  private:
    float loadParam(const juce::String &id) const {
        if (auto p = apvts.getRawParameterValue(id))
            return p->load();

        return 0.0f;
    }

    void timerCallback() override {
        updateSegmentRanges();
        updatePositions();
        repaint();
    }

    void updatePositions() {
        if (envelopeBounds.isEmpty())
            return;

        positionControlPoint(*attackCtrl);
        positionControlPoint(*decayCtrl);
        positionControlPoint(*releaseCtrl);
    }

    // Position component according to its stored parameters and segment range
    void positionControlPoint(ControlPoint &c) {
        float segLocalX = loadParam(c.getXParamID());
        float localY = loadParam(c.getYParamID());

        float sStart = c.getSegStart();
        float sEnd = c.getSegEnd();

        float frac = sStart + segLocalX * (sEnd - sStart);
        frac = juce::jlimit(0.0f, 1.0f, frac);

        float cx = envelopeBounds.getX() + frac * envelopeBounds.getWidth();
        float cy =
            envelopeBounds.getBottom() - localY * envelopeBounds.getHeight();

        int x = juce::roundToInt(cx - c.getWidth() * 0.5f);
        int y = juce::roundToInt(cy - c.getHeight() * 0.5f);

        c.setTopLeftPosition(x, y);
    }

    // Compute and set the segment ranges to constrain them appropriately
    void updateSegmentRanges() {
        float attack = loadParam("attack");
        float decay = loadParam("decay");
        float release = loadParam("release");

        attack = juce::jmax(0.0f, attack);
        decay = juce::jmax(0.0f, decay);
        release = juce::jmax(0.0f, release);

        float maxTime = ADSR_MAX * 3.0f;

        // Compute timeline points
        float t0 = 0.0f;
        float t1 = attack;
        float t2 = attack + decay;
        float t4 = maxTime;
        float t3 = juce::jmax(t2, maxTime - release);

        // Convert to fractions of full envelope
        float f0 = t0 / maxTime;
        float f1 = t1 / maxTime;
        float f2 = t2 / maxTime;
        float f3 = t3 / maxTime;
        float f4 = t4 / maxTime; // should be 1.0f

        // Attack control allowed region: (f0, f1]
        attackCtrl->setXRange(f0, juce::jmax(f0 + 1e-6f, f1));

        // Decay control allowed region: (f1, f2]
        decayCtrl->setXRange(f1, juce::jmax(f1 + 1e-6f, f2));

        // Release control allowed region: (f3, f4]
        releaseCtrl->setXRange(f3, juce::jmax(f3 + 1e-6f, f4));
    }

    juce::AudioProcessorValueTreeState &apvts;

    std::unique_ptr<ControlPoint> attackCtrl;
    std::unique_ptr<ControlPoint> decayCtrl;
    std::unique_ptr<ControlPoint> releaseCtrl;

    juce::Rectangle<float> envelopeBounds;
};
