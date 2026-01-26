"""
Simple PipeWire Audio Capture Script

Uses sounddevice library which automatically integrates with PipeWire
on modern Linux systems (via the PipeWire-ALSA or PipeWire-PulseAudio bridge).
"""

import sounddevice as sd
import numpy as np
import argparse
import wave
import queue
import sys
from datetime import datetime


# ==========================
# AUDIO CONFIG
# ==========================
SAMPLE_RATE = 22050
CHANNELS = 1
FRAME_SIZE = 1024
DTYPE = "float32"


class PipeWireCapture:
    """Simple audio capture class using PipeWire via sounddevice."""

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        channels: int = CHANNELS,
        frame_size: int = FRAME_SIZE,
        device: int | str | None = None,
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.frame_size = frame_size
        self.device = device
        self.audio_queue = queue.Queue()
        self.stream = None
        self.is_recording = False

    def _audio_callback(self, indata, frames, time, status):
        """Callback function for audio stream."""
        if status:
            print(f"Audio status: {status}", file=sys.stderr)
        # Put a copy of the audio data into the queue
        self.audio_queue.put(indata.copy())

    def list_devices(self):
        """List available audio input devices."""
        print("\n=== Available Audio Devices ===")
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if device["max_input_channels"] > 0:
                marker = " [DEFAULT]" if i == sd.default.device[0] else ""
                print(f"  [{i}] {device['name']}{marker}")
                print(f"       Inputs: {device['max_input_channels']}, "
                      f"Sample Rate: {device['default_samplerate']}")
        print()

    def start_stream(self):
        """Start the audio capture stream."""
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            blocksize=self.frame_size,
            dtype=DTYPE,
            device=self.device,
            callback=self._audio_callback,
        )
        self.stream.start()
        self.is_recording = True
        print(f"Recording started at {self.sample_rate}Hz, {self.channels} channel(s)")

    def stop_stream(self):
        """Stop the audio capture stream."""
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self.is_recording = False
        print("Recording stopped")

    def get_audio_chunk(self, timeout: float = 1.0) -> np.ndarray | None:
        """Get the next audio chunk from the queue."""
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def record_to_file(self, filename: str, duration: float):
        """Record audio to a WAV file for a specified duration."""
        print(f"Recording {duration} seconds to {filename}...")
        
        frames = []
        total_frames = int(self.sample_rate * duration)
        recorded_frames = 0

        self.start_stream()

        try:
            while recorded_frames < total_frames:
                chunk = self.get_audio_chunk()
                if chunk is not None:
                    frames.append(chunk)
                    recorded_frames += len(chunk)
        except KeyboardInterrupt:
            print("\nRecording interrupted by user")
        finally:
            self.stop_stream()

        if frames:
            audio_data = np.concatenate(frames, axis=0)
            # Trim to exact duration
            audio_data = audio_data[:total_frames]
            self._save_wav(filename, audio_data)
            print(f"Saved {len(audio_data) / self.sample_rate:.2f} seconds to {filename}")

    def _save_wav(self, filename: str, audio_data: np.ndarray):
        """Save audio data to a WAV file."""
        # Convert float32 to int16 for WAV
        audio_int16 = (audio_data * 32767).astype(np.int16)
        
        with wave.open(filename, "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_int16.tobytes())

    def stream_audio(self, callback=None):
        """
        Stream audio continuously, optionally calling a callback for each chunk.
        
        Args:
            callback: Optional function to process each audio chunk.
                     Signature: callback(audio_chunk: np.ndarray) -> bool
                     Return False to stop streaming.
        """
        self.start_stream()

        try:
            while self.is_recording:
                chunk = self.get_audio_chunk()
                if chunk is not None:
                    if callback:
                        if not callback(chunk):
                            break
                    else:
                        # Default: print RMS level
                        rms = np.sqrt(np.mean(chunk**2))
                        bars = int(rms * 50)
                        print(f"\rLevel: {'█' * bars}{' ' * (50 - bars)} {rms:.4f}", end="")
        except KeyboardInterrupt:
            print("\nStreaming stopped by user")
        finally:
            self.stop_stream()


def main():
    parser = argparse.ArgumentParser(description="PipeWire Audio Capture")
    parser.add_argument(
        "--list-devices", "-l",
        action="store_true",
        help="List available audio input devices"
    )
    parser.add_argument(
        "--device", "-d",
        type=int,
        default=None,
        help="Audio device index to use"
    )
    parser.add_argument(
        "--record", "-r",
        type=str,
        metavar="FILENAME",
        help="Record audio to a WAV file"
    )
    parser.add_argument(
        "--duration", "-t",
        type=float,
        default=5.0,
        help="Recording duration in seconds (default: 5.0)"
    )
    parser.add_argument(
        "--stream", "-s",
        action="store_true",
        help="Stream audio and display level meter"
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=SAMPLE_RATE,
        help=f"Sample rate in Hz (default: {SAMPLE_RATE})"
    )

    args = parser.parse_args()

    capture = PipeWireCapture(
        sample_rate=args.sample_rate,
        device=args.device,
    )

    if args.list_devices:
        capture.list_devices()
    elif args.record:
        capture.record_to_file(args.record, args.duration)
    elif args.stream:
        print("Streaming audio (Ctrl+C to stop)...")
        capture.stream_audio()
    else:
        # Default: show help
        parser.print_help()
        print("\nExamples:")
        print("  python pipewire_capture.py --list-devices")
        print("  python pipewire_capture.py --stream")
        print("  python pipewire_capture.py --record output.wav --duration 10")


if __name__ == "__main__":
    main()
