#include "MainComponent.h"

//==============================================================================
MainComponent::MainComponent()
{
    // Make sure you set the size of the component after
    // you add any child components.
    
    // Configure Chord Label
    addAndMakeVisible(chordLabel);
    chordLabel.setText("...", juce::dontSendNotification);
    chordLabel.setJustificationType(juce::Justification::centred);
    chordLabel.setColour(juce::Label::textColourId, juce::Colours::white);
    
    // Configure Confidence Label
    addAndMakeVisible(confidenceLabel);
    confidenceLabel.setText("Waiting for audio...", juce::dontSendNotification);
    confidenceLabel.setJustificationType(juce::Justification::centred);
    confidenceLabel.setColour(juce::Label::textColourId, juce::Colours::lightgrey);
    
    // Configure Audio Settings Button
    addAndMakeVisible(audioSettingsButton);
    audioSettingsButton.setButtonText("Audio Settings");
    audioSettingsButton.onClick = [this] {
        showAudioSettings = !showAudioSettings;
        if (showAudioSettings)
        {
            audioSettingsComp.reset(new juce::AudioDeviceSelectorComponent(deviceManager,
                                                                         0, 256,  // min/max input channels
                                                                         0, 256,  // min/max output channels
                                                                         false,   // show midi input options
                                                                         false,   // show midi output options
                                                                         false,   // show channels as stereo pairs
                                                                         false)); // hide advanced options
            addAndMakeVisible(audioSettingsComp.get());
            audioSettingsComp->setSize(getWidth() - 40, getHeight() - 100);
            resized();
        }
        else
        {
            audioSettingsComp.reset();
        }
    };

    setSize (800, 600);

    // Some platforms require permissions to open input channels so request that here
    if (juce::RuntimePermissions::isRequired (juce::RuntimePermissions::recordAudio)
        && ! juce::RuntimePermissions::isGranted (juce::RuntimePermissions::recordAudio))
    {
        juce::RuntimePermissions::request (juce::RuntimePermissions::recordAudio,
                                           [&] (bool granted) { setAudioChannels (granted ? 2 : 0, 2); });
    }
    else
    {
        // Specify the number of input and output channels that we want to open
        setAudioChannels (2, 2);
    }
}

MainComponent::~MainComponent()
{
    // This shuts down the audio device and clears the audio source.
    shutdownAudio();
}

//==============================================================================
void MainComponent::prepareToPlay (int samplesPerBlockExpected, double sampleRate)
{
    // This function will be called when the audio device is started, or when
    // its settings (i.e. sample rate, block size, etc) are changed.

    // You can use this function to initialise any resources you might need,
    // but be careful - it will be called on the audio thread, not the GUI thread.

    // For more details, see the help for AudioProcessor::prepareToPlay()
}

void MainComponent::getNextAudioBlock (const juce::AudioSourceChannelInfo& bufferToFill)
{
    // Your audio-processing code goes here!

    // For more details, see the help for AudioProcessor::getNextAudioBlock()

    // Right now we are not producing any data, in which case we need to clear the buffer
    // (to prevent the output of random noise)
    bufferToFill.clearActiveBufferRegion();
}

void MainComponent::releaseResources()
{
    // This will be called when the audio device stops, or when it is being
    // restarted due to a setting change.

    // For more details, see the help for AudioProcessor::releaseResources()
}

//==============================================================================
void MainComponent::paint (juce::Graphics& g)
{
    // Modern Dark Theme Background
    juce::Colour gradientStart = juce::Colour::fromString("FF050A14"); // Very dark blue/black
    juce::Colour gradientEnd = juce::Colour::fromString("FF001530");   // Deep blue
    
    juce::ColourGradient backgroundGradient(gradientStart, 0, 0, gradientEnd, 
                                          (float)getWidth(), (float)getHeight(), false);
    g.setGradientFill(backgroundGradient);
    g.fillAll();

    // Add a subtle glow/vignette center
    g.setGradientFill(juce::ColourGradient(juce::Colours::blue.withAlpha(0.05f), 
                                         (float)getWidth() / 2.0f, (float)getHeight() / 2.0f,
                                         juce::Colours::transparentBlack, 
                                         (float)getWidth() / 1.5f, (float)getHeight() / 1.5f, true));
    g.fillAll();
    
    if (showAudioSettings)
    {
        g.fillAll(juce::Colours::black.withAlpha(0.8f));
    }
}

void MainComponent::resized()
{
    // Layout
    auto area = getLocalBounds();
    
    // Center area for chord display
    auto centerArea = area.removeFromTop(area.getHeight() * 0.7f).reduced(20);
    
    // Determine font sizes based on window height
    float chordFontSize = (float)juce::jmin(getWidth(), getHeight()) * 0.25f;
    float labelFontSize = (float)juce::jmin(getWidth(), getHeight()) * 0.05f;

    chordLabel.setFont(juce::Font(chordFontSize, juce::Font::bold));
    chordLabel.setBounds(centerArea);
    
    confidenceLabel.setFont(juce::Font(labelFontSize, juce::Font::plain));
    confidenceLabel.setBounds(area.reduced(20));
    
    audioSettingsButton.setBounds(getWidth() - 150, 10, 140, 30);
    
    if (audioSettingsComp)
    {
        audioSettingsComp->setBounds(20, 50, getWidth() - 40, getHeight() - 60);
    }
}
