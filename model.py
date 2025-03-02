#This is the code from kaggle that I used to train the model


# from keras import models, layers 
# import tensorflow as tf
# train_dir = "guitar-chords.v1i.folder/train"
# validation_dir = "guitar-chords.v1i.folder/valid"

# # Create training and validation datasets
# train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
#     train_dir,
#     labels='inferred',
#     label_mode='categorical',
#     batch_size=32,
#     image_size=(224, 224),
#     shuffle=True
# )

# validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
#     validation_dir,
#     labels='inferred',
#     label_mode='categorical',
#     batch_size=32,
#     image_size=(224, 224),
#     shuffle=False
# )
# #Data augmentation
# data_augmentation = tf.keras.Sequential([
#     tf.keras.layers.RandomRotation(0.2),
#     tf.keras.layers.RandomTranslation(0.1, 0.1),
#     tf.keras.layers.RandomFlip("horizontal")
# ])

# # Apply augmentation to the training dataset
# augmented_train_dataset = train_dataset.map(
#     lambda x, y: (data_augmentation(x, training=True), y)
# )

# # Determine the number of classes from the dataset
# num_classes = len(train_dataset.class_names)
# ##TRAINING THE MODEL
# model = models.Sequential([
#     #three convolutional layers with max pooling  
#     layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
#     layers.MaxPooling2D((2, 2)),
#     layers.Conv2D(64, (3, 3), activation='relu'),
#     layers.MaxPooling2D((2, 2)),
#     layers.Conv2D(128, (3, 3), activation='relu'),
#     layers.MaxPooling2D((2, 2)),
#     layers.Flatten(),
#     layers.Dense(128, activation='relu'),
#     layers.Dense(num_classes, activation='softmax')
# ])

# model.compile(optimizer='adam',
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])

# model.fit(
#     train_dataset,
#     validation_data=validation_dataset,
#     epochs=20
# )

# model.save('image_classification_model.h5')

# loss, accuracy = model.evaluate(validation_dataset)
# print(f'Validation Loss: {loss:.4f}, Accuracy: {accuracy:.4f}')
# loss, accuracy = model.evaluate(validation_dataset)
# import cv2
# import numpy as np
# import tensorflow as tf
# import librosa
# import time
# import os
# from watchdog.observers import Observer
# from watchdog.events import FileSystemEventHandler

# # Load the saved model
# model_path = 'guitar_chord_model.h5'  # Update with your model path
# model = tf.keras.models.load_model(model_path)

# # Constants for audio processing
# SAMPLE_RATE = 22050
# DURATION = 3  # seconds
# N_MELS = 128
# N_FFT = 2048
# HOP_LENGTH = 512

# # List of chord classes
# CHORD_CLASSES = ['Am', 'Bb', 'Bdim', 'C', 'Dm', 'Em', 'F', 'G']

# def predict_chord_from_file(file_path):
#     # Load audio file
#     audio, _ = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
    
#     # Pad if audio is shorter than expected duration
#     if len(audio) < SAMPLE_RATE * DURATION:
#         audio = np.pad(audio, (0, SAMPLE_RATE * DURATION - len(audio)), mode='constant')
    
#     # Extract mel spectrogram
#     mel_spec = librosa.feature.melspectrogram(
#         y=audio, 
#         sr=SAMPLE_RATE, 
#         n_mels=N_MELS,
#         n_fft=N_FFT,
#         hop_length=HOP_LENGTH
#     )
    
#     # Convert to decibels
#     mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
#     # Reshape for model input
#     feature = mel_spec_db[np.newaxis, ..., np.newaxis]
    
#     # Make prediction
#     prediction = model.predict(feature, verbose=0)[0]
#     chord_index = np.argmax(prediction)
#     confidence = prediction[chord_index]
    
#     return CHORD_CLASSES[chord_index], confidence

# # Function for displaying results with OpenCV
# def display_result(chord, confidence, file_path):
#     # Create window
#     cv2.namedWindow("Guitar Chord Detector", cv2.WINDOW_NORMAL)
#     cv2.resizeWindow("Guitar Chord Detector", 800, 600)
    
#     # Create blank image for display
#     display = np.ones((600, 800, 3), dtype=np.uint8) * 255
    
#     # Display information
#     cv2.putText(display, "Guitar Chord Detector", (200, 60), 
#                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
#     # Draw a line
#     cv2.line(display, (50, 80), (750, 80), (0, 0, 0), 2)
    
#     # Display file name
#     file_name = os.path.basename(file_path)
#     cv2.putText(display, f"File: {file_name}", (50, 150), 
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    
#     # Display chord information
#     cv2.putText(display, f"Detected Chord:", (50, 200), 
#                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
#     # Color based on confidence
#     chord_color = (0, 180, 0) if confidence > 0.7 else (0, 0, 180)
    
#     # Display the detected chord prominently
#     cv2.putText(display, chord, (400, 300), 
#                cv2.FONT_HERSHEY_SIMPLEX, 5, chord_color, 5)
    
#     # Show confidence
#     cv2.putText(display, f"Confidence: {confidence:.2f}", (300, 400), 
#                 cv2.FONT_HERSHEY_SIMPLEX, 1, chord_color, 2)
    
#     # Add instructions
#     cv2.putText(display, "Press any key to continue, 'q' to quit", (250, 500), 
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 1)
    
#     # Display the image
#     cv2.imshow("Guitar Chord Detector", display)
#     key = cv2.waitKey(0)
#     cv2.destroyAllWindows()
    
#     return key == ord('q')

# # File system event handler to watch for new audio files
# class AudioFileHandler(FileSystemEventHandler):
#     def on_created(self, event):
#         if not event.is_directory and event.src_path.endswith(('.wav', '.mp3')):
#             print(f"New audio file detected: {event.src_path}")
#             chord, confidence = predict_chord_from_file(event.src_path)
#             print(f"Detected chord: {chord} with confidence: {confidence:.2f}")
#             quit_app = display_result(chord, confidence, event.src_path)
            
#             if quit_app:
#                 observer.stop()

# # Watch directory for new audio files
# def watch_directory(directory_path):
#     event_handler = AudioFileHandler()
#     global observer
#     observer = Observer()
#     observer.schedule(event_handler, directory_path, recursive=False)
#     observer.start()
    
#     try:
#         print(f"Watching directory {directory_path} for audio files...")
#         print("Record a chord and save the audio file in this directory.")
#         print("Press Ctrl+C to stop.")
#         while observer.is_alive():
#             observer.join(1)
#     except KeyboardInterrupt:
#         observer.stop()
#     observer.join()

# # Main function
# if __name__ == "__main__":
#     # Directory to watch for new audio files
#     watch_dir = "audio_files"  # Update this to your directory
    
#     # Create directory if it doesn't exist
#     if not os.path.exists(watch_dir):
#         os.makedirs(watch_dir)
        
#     # Start watching the directory
#     watch_directory(watch_dir)