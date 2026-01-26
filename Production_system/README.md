
## PipeWire scripts 

# List available audio devices
python pipewire_capture.py --list-devices

# Stream audio with level meter (Ctrl+C to stop)
python pipewire_capture.py --stream

# Record 10 seconds to a file
python pipewire_capture.py --record output.wav --duration 10

# Use a specific device
python pipewire_capture.py --device 2 --stream