import cv2
import numpy as np
import tensorflow as tf
import pyaudio
import librosa
import threading
import time
import queue

# Load the saved model
model = tf.keras.models.load_model('/kaggle/working/guitar_chord_model.h5')

# Constants for audio processing
SAMPLE_RATE = 22050
DURATION = 3  # seconds
BUFFER_SIZE = 1024  # Audio buffer size
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
CHANNELS = 1
FORMAT = pyaudio.paFloat32

# List of chord classes based on your dataset structure
CHORD_CLASSES = ['Am', 'Bb', 'Bdim', 'C', 'Dm', 'Em', 'F', 'G']

# Create a queue for communication between audio and video threads
audio_queue = queue.Queue()

# Function to process audio and predict chords
def audio_processing():
    # Initialize PyAudio
    p = pyaudio.PyAudio()
    
    # Initialize buffer to store audio data
    audio_buffer = np.zeros(SAMPLE_RATE * DURATION, dtype=np.float32)
    
    # Open audio stream
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=BUFFER_SIZE
    )
    
    print("Audio stream started, listening for chords...")
    
    try:
        while True:
            # Read audio data
            audio_data = stream.read(BUFFER_SIZE, exception_on_overflow=False)
            audio_chunk = np.frombuffer(audio_data, dtype=np.float32)
            
            # Roll the buffer and add new data
            audio_buffer = np.roll(audio_buffer, -len(audio_chunk))
            audio_buffer[-len(audio_chunk):] = audio_chunk
            
            # Process audio and extract features
            mel_spec = librosa.feature.melspectrogram(
                y=audio_buffer,
                sr=SAMPLE_RATE,
                n_mels=N_MELS,
                n_fft=N_FFT,
                hop_length=HOP_LENGTH
            )
            
            # Convert to decibels
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Reshape for model input
            feature = mel_spec_db[np.newaxis, ..., np.newaxis]
            
            # Make prediction
            prediction = model.predict(feature, verbose=0)[0]
            chord_index = np.argmax(prediction)
            confidence = prediction[chord_index]
            
            # Put the prediction in the queue for the video thread
            detected_chord = CHORD_CLASSES[chord_index]
            audio_queue.put((detected_chord, confidence))
            
            time.sleep(0.1)  # Small delay to prevent CPU overload
    
    except KeyboardInterrupt:
        print("Stopping audio processing...")
    
    finally:
        # Close audio stream
        stream.stop_stream()
        stream.close()
        p.terminate()

# Function for OpenCV display
def video_display():
    # Create window
    cv2.namedWindow("Guitar Chord Detector", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Guitar Chord Detector", 800, 600)
    
    # Create blank image for display
    display = np.ones((600, 800, 3), dtype=np.uint8) * 255
    
    # Initialize variables
    current_chord = "No chord detected"
    current_confidence = 0.0
    
    try:
        while True:
            # Clear display
            display.fill(255)
            
            # Try to get the latest prediction from the queue
            try:
                while not audio_queue.empty():
                    current_chord, current_confidence = audio_queue.get_nowait()
            except queue.Empty:
                pass
            
            # Display information
            cv2.putText(display, "Guitar Chord Detector", (200, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            
            # Draw a line
            cv2.line(display, (50, 80), (750, 80), (0, 0, 0), 2)
            
            # Display chord with larger font
            cv2.putText(display, f"Detected Chord:", (50, 200), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            
            # Only show high-confidence predictions with green color
            chord_color = (0, 180, 0) if current_confidence > 0.7 else (0, 0, 180)
            
            # Display the detected chord prominently
            cv2.putText(display, current_chord, (400, 300), 
                       cv2.FONT_HERSHEY_SIMPLEX, 5, chord_color, 5)
            
            # Show confidence
            cv2.putText(display, f"Confidence: {current_confidence:.2f}", (300, 400), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, chord_color, 2)
            
            # Add instructions
            cv2.putText(display, "Press 'q' to quit", (300, 500), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 1)
            
            # Display the image
            cv2.imshow("Guitar Chord Detector", display)
            
            # Check for quit
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("Stopping video display...")
    
    finally:
        cv2.destroyAllWindows()

# Start audio processing in a separate thread
audio_thread = threading.Thread(target=audio_processing)
audio_thread.daemon = True
audio_thread.start()

# Run the video display in the main thread
video_display()