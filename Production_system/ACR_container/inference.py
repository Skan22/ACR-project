import numpy as np

# CONFIGURATION
SAMPLE_RATE = 22050
MODEL_INPUT_DURATION = 2.0  # How many seconds your model needs to "see"
MODEL_INPUT_SIZE = int(SAMPLE_RATE * MODEL_INPUT_DURATION) # 44100 samples

# The Buffer (Initialize with zeros)
# This acts as the model's "short-term memory"
rolling_buffer = np.zeros(MODEL_INPUT_SIZE, dtype=np.float32)

def process_audio_chunk(new_frames):
    """
    new_frames: A small chunk from PipeWire (e.g., 1024 samples)
    """
    global rolling_buffer
    
    # 1. Shift the old data to the left
    # If new_frames has 1024 samples, we shift everything left by 1024
    chunk_size = len(new_frames)
    rolling_buffer = np.roll(rolling_buffer, -chunk_size)
    
    # 2. Insert the new data at the end
    rolling_buffer[-chunk_size:] = new_frames
    
    # 3. Now you have a full second of audio mixed with the latest data
    # THIS is what you send to your model
    return rolling_buffer

# --- INSIDE YOUR MAIN LOOP ---
# audio_block = audio_queue.get()  <-- This is only ~46ms (1024 frames)
# full_input = process_audio_chunk(audio_block) <-- This is 1s (22050 frames)
# prediction = model.predict(full_input)