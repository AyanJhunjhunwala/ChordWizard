import cv2
import numpy as np
import tensorflow as tf
import pyaudio
import librosa
import threading
import time
import queue
import sys
import random
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout, 
                             QHBoxLayout, QWidget, QPushButton, QProgressBar, QAction)
from PyQt5.QtGui import QFont, QPainter, QColor, QPen, QBrush, QFontDatabase
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QRect, QPropertyAnimation, QEasingCurve, pyqtProperty
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
# -------------------------------
# Fretboard Widget to Display Chord Positions
# -------------------------------
class FretboardWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.chord = None
        # Define chord positions (for 6 strings: low E to high E)
        # -1 means muted, 0 means open, any positive number is the fret number.
        self.chord_positions = {
            'C':    [-1, 3, 2, 0, 1, 0],
            'Dm':   [-1, -1, 0, 2, 3, 1],
            'Em':   [0, 2, 2, 0, 0, 0],
            'F':    [1, 3, 3, 2, 1, 1],
            'G':    [3, 2, 0, 0, 0, 3],
            'Am':   [-1, 0, 2, 2, 1, 0],
            'Bb':   [-1, 1, 3, 3, 3, 1],
            'Bdim': [-1, 2, 3, 2, 3, -1]
        }
        self.current_positions = None
        self.setMinimumSize(200, 300)

    def update_chord(self, chord):
        self.chord = chord
        self.current_positions = self.chord_positions.get(chord, None)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = self.rect()
        margin = 20
        # Reserve some space at the top for markers (X/O)
        diagram_rect = QRect(margin, margin + 20, rect.width() - 2 * margin, rect.height() - 2 * margin - 20)
        
        # Define diagram grid: 6 strings (vertical) and 5 frets (horizontal: nut + 4 frets)
        num_strings = 6
        num_frets = 5  # Nut + 4 frets
        cell_width = diagram_rect.width() / (num_strings - 1)
        cell_height = diagram_rect.height() / num_frets

        # Draw nut as a thick horizontal line at the top of the diagram_rect
        nut_y = diagram_rect.top()
        painter.setPen(QPen(QColor(255, 255, 255), 4))
        painter.drawLine(diagram_rect.left(), nut_y, diagram_rect.right(), nut_y)

        # Draw fret lines (horizontal lines)
        painter.setPen(QPen(QColor(255, 255, 255), 2))
        for i in range(1, num_frets + 1):
            y = diagram_rect.top() + i * cell_height
            painter.drawLine(diagram_rect.left(), int(y), diagram_rect.right(), int(y))

        # Draw string lines (vertical lines)
        for i in range(num_strings):
            x = diagram_rect.left() + i * cell_width
            painter.setPen(QPen(QColor(255, 255, 255), 2))
            painter.drawLine(int(x), diagram_rect.top(), int(x), diagram_rect.bottom())

        # Draw markers for each string
        if self.current_positions:
            # Loop over strings (assume order: low E, A, D, G, B, high E)
            for i, pos in enumerate(self.current_positions):
                # Calculate x position for the string
                x = diagram_rect.left() + i * cell_width
                # Marker position above the nut
                marker_x = x
                marker_y = diagram_rect.top() - 10
                if pos == -1:
                    # Muted string: draw an "X" in red
                    painter.setPen(QPen(QColor(255, 0, 0), 2))
                    painter.drawText(int(marker_x - 5), int(marker_y), "X")
                elif pos == 0:
                    # Open string: draw an "O" in green
                    painter.setPen(QPen(QColor(0, 255, 0), 2))
                    painter.drawText(int(marker_x - 5), int(marker_y), "O")
                else:
                    # Fretted note: draw a filled circle in the corresponding fret cell.
                    fret_index = pos  # fret number (e.g., 1 means first fret cell)
                    y_top = diagram_rect.top() + (fret_index - 1) * cell_height
                    y_bottom = diagram_rect.top() + fret_index * cell_height
                    center_y = (y_top + y_bottom) / 2
                    center_x = x
                    radius = min(cell_width, cell_height) / 4
                    painter.setPen(QPen(QColor(255, 255, 0), 2))
                    painter.setBrush(QBrush(QColor(255, 255, 0)))
                    painter.drawEllipse(QRect(int(center_x - radius), int(center_y - radius), int(2 * radius), int(2 * radius)))
            
        # Optionally, display the chord name at the bottom
        if self.chord:
            painter.setPen(QColor(255, 255, 255))
            painter.setFont(QFont("Arial", 16, QFont.Bold))
            adjusted_rect = QRect(rect.left(), rect.top(), rect.width(), rect.height() - 5)
            painter.drawText(adjusted_rect, Qt.AlignBottom | Qt.AlignHCenter, self.chord)

# -------------------------------
# Existing Widgets & Application Code
# -------------------------------
class AudioWaveformWidget(FigureCanvas):
    def __init__(self, parent=None, width=5, height=2, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(AudioWaveformWidget, self).__init__(fig)
        self.setParent(parent)
        
        # Remove background and frame
        fig.patch.set_facecolor('black')
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
        self.axes.set_facecolor('black')
        
        # Remove all ticks, labels and grid
        self.axes.set_xticks([])
        self.axes.set_yticks([])
        self.axes.set_xticklabels([])
        self.axes.set_yticklabels([])
        self.axes.spines['top'].set_visible(False)
        self.axes.spines['right'].set_visible(False)
        self.axes.spines['bottom'].set_visible(False)
        self.axes.spines['left'].set_visible(False)
        
        # Initialize line
        self.line, = self.axes.plot([], [], lw=2, color='#00FFFF')
        self.axes.set_ylim(-1, 1)
        self.axes.set_xlim(0, 1024)
        
    def update_waveform(self, audio_data):
        plot_data = audio_data[-1024:]
        x = np.arange(len(plot_data))
        self.line.set_data(x, plot_data)
        self.draw()

# -------------------------------
# Solid Color Chord Display (instead of a gradient)
# -------------------------------
class SolidChordDisplay(QWidget):
    def __init__(self, parent=None):
        super(SolidChordDisplay, self).__init__(parent)
        self.chord = "Play"
        self.confidence = 0.0
        self.setMinimumHeight(150)
        
        self.hue = 0.0
        self.target_hue = 0.3
        self.animation_speed = 0.005
        
        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self.animate_color)
        self.animation_timer.start(50)
        
        self.pulse_size = 0
        self.pulse_growing = True
        
    def animate_color(self):
        if abs(self.hue - self.target_hue) > 0.01:
            if self.hue < self.target_hue:
                self.hue += self.animation_speed
            else:
                self.hue -= self.animation_speed
        else:
            self.target_hue = random.uniform(0.05, 0.15)
        self.update()
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = self.rect()
        
        # Clear background with black
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor("#000000")))
        painter.drawRect(rect)
        
        center = rect.center()
        # Use one solid color based on current hue.
        main_color = QColor.fromHsvF(self.hue, 0.8, 0.9)
        
        # Animate pulse size
        if self.pulse_growing:
            self.pulse_size += 0.5
            if self.pulse_size > 10:
                self.pulse_growing = False
        else:
            self.pulse_size -= 0.5
            if self.pulse_size < 0:
                self.pulse_growing = True
        
        pulse_rect = QRect(
            int(center.x() - 100 - self.pulse_size),
            int(center.y() - 100 - self.pulse_size),
            int(200 + self.pulse_size * 2),
            int(200 + self.pulse_size * 2)
        )
        painter.setBrush(QBrush(main_color))
        painter.drawEllipse(pulse_rect)
        
        painter.setPen(QColor(255, 255, 255))
        painter.setFont(QFont("Arial", 48, QFont.Bold))
        painter.drawText(rect, Qt.AlignCenter, self.chord)
        
    def update_chord(self, chord, confidence):
        self.chord = chord
        self.confidence = confidence
        
        chord_map = {
            'C': 0.1,
            'Dm': 0.2,
            'Em': 0.33,
            'F': 0.5,
            'G': 0.6,
            'Am': 0.8,
            'Bb': 0.9,
            'Bdim': 0.95
        }
        
        if chord in chord_map:
            self.target_hue = chord_map[chord]
        self.update()

# -------------------------------
# Mouse-based Progression Selector Widget
# -------------------------------
class ProgressionSelectorWidget(QWidget):
    chord_selected = pyqtSignal(str)
    def __init__(self, chords, parent=None):
        super().__init__(parent)
        self.chords = chords
        self.init_ui()
        
    def init_ui(self):
        layout = QHBoxLayout(self)
        layout.setSpacing(5)
        for chord in self.chords:
            btn = QPushButton(chord)
            btn.setFixedSize(50, 50)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #222222; 
                    color: #FFFFFF; 
                    font-size: 14px; 
                    border-radius: 5px;
                }
                QPushButton:hover {
                    background-color: #444444;
                }
            """)
            btn.clicked.connect(lambda checked, c=chord: self.chord_selected.emit(c))
            layout.addWidget(btn)

# -------------------------------
# Main Application
# -------------------------------
class ChordWizardApp(QMainWindow):
    update_signal = pyqtSignal(str, float, np.ndarray)
    suggest_signal = pyqtSignal(list)
    
    def __init__(self):
        super().__init__()
        self.load_fonts()
        
        self.model = tf.keras.models.load_model('guitar_chord_model.h5')
        self.CHORD_CLASSES = ['Am', 'Bb', 'Bdim', 'C', 'Dm', 'Em', 'F', 'G']
        
        self.SAMPLE_RATE = 22050
        self.DURATION = 3
        self.BUFFER_SIZE = 1024
        self.N_MELS = 128
        self.N_FFT = 2048
        self.HOP_LENGTH = 512
        self.CHANNELS = 1
        self.FORMAT = pyaudio.paFloat32
        
        self.audio_queue = queue.Queue()
        self.audio_buffer = np.zeros(self.SAMPLE_RATE * self.DURATION, dtype=np.float32)
        
        # Chord progression tracking
        self.current_progression = []
        self.last_chord_time = 0
        self.silence_threshold = 2.0
        self.current_chord = None
        self.last_chord_change_time = 0
        
        # Manual progression variables (for mouse-based selection)
        self.manual_progression_active = False
        self.selected_progression = []  # will be built via mouse clicks
        self.progression_index = 0
        
        self.setup_ui()
        self.create_menus()
        self.setup_audio()
        
        self.update_signal.connect(self.update_display)
        self.suggest_signal.connect(self.show_suggestions)
        
        self.audio_thread = threading.Thread(target=self.audio_processing)
        self.audio_thread.daemon = True
        self.audio_thread.start()
    
    def load_fonts(self):
        self.cursive_font = QFont("Brush Script MT")
        self.cursive_font.setItalic(True)
        if not self.is_font_available("Brush Script MT"):
            cursive_fonts = ["Comic Sans MS", "Segoe Script", "Lucida Handwriting"]
            for font in cursive_fonts:
                if self.is_font_available(font):
                    self.cursive_font = QFont(font)
                    break
    
    def is_font_available(self, font_name):
        return font_name in QFontDatabase().families()
        
    def setup_ui(self):
        self.setWindowTitle("ChordWizard")
        self.setMinimumSize(800, 600)
        self.setStyleSheet("background-color: #000000;")
        
        # Create a horizontal layout to hold left-side content and fretboard widget
        main_widget = QWidget()
        main_hlayout = QHBoxLayout(main_widget)
        
        # Left side: vertical layout with title, chord display, waveform, progression, etc.
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(20, 20, 20, 20)
        left_layout.setSpacing(15)
        
        title_label = QLabel("ChordWizard")
        title_label.setAlignment(Qt.AlignCenter)
        self.cursive_font.setPointSize(32)
        title_label.setFont(self.cursive_font)
        title_label.setStyleSheet("color: #FFFFFF; margin: 0;")
        left_layout.addWidget(title_label)
        
        self.chord_display = SolidChordDisplay()
        left_layout.addWidget(self.chord_display)
        
        self.waveform_widget = AudioWaveformWidget(left_widget, width=5, height=2)
        left_layout.addWidget(self.waveform_widget)
        
        self.progression_label = QLabel("Chord Progression: None")
        self.progression_label.setAlignment(Qt.AlignCenter)
        self.progression_label.setStyleSheet("color: #FFFFFF; font-size: 16px;")
        left_layout.addWidget(self.progression_label)
        
        # New label to show the next expected chord (highlighted in red)
        self.next_chord_label = QLabel("Next Chord: None")
        self.next_chord_label.setAlignment(Qt.AlignCenter)
        self.next_chord_label.setStyleSheet("color: #FF0000; font-size: 18px;")
        left_layout.addWidget(self.next_chord_label)
        
        # Add the mouse-based progression selector widget.
        self.progression_selector = ProgressionSelectorWidget(self.CHORD_CLASSES)
        self.progression_selector.chord_selected.connect(self.add_chord_to_manual_progression)
        left_layout.addWidget(self.progression_selector)
        
        # Button to finish manual progression.
        self.finish_progression_button = QPushButton("Finish Manual Progression")
        self.finish_progression_button.setStyleSheet("background-color: #333333; color: #FFFFFF;")
        self.finish_progression_button.clicked.connect(self.finish_manual_progression)
        left_layout.addWidget(self.finish_progression_button)
        
        # Existing suggestions container (remains for auto mode)
        self.suggestions_container = QWidget()
        self.suggestions_layout = QHBoxLayout(self.suggestions_container)
        self.suggestions_layout.setAlignment(Qt.AlignCenter)
        self.suggestions_layout.setContentsMargins(0, 0, 0, 0)
        self.suggestions_layout.setSpacing(10)
        
        suggestion_label = QLabel("Try next:")
        suggestion_label.setStyleSheet("color: #CCCCCC; font-size: 14px;")
        self.suggestions_layout.addWidget(suggestion_label)
        left_layout.addWidget(self.suggestions_container)
        
        self.confidence_bar = QProgressBar()
        self.confidence_bar.setRange(0, 100)
        self.confidence_bar.setValue(0)
        self.confidence_bar.setTextVisible(False)
        self.confidence_bar.setMaximumHeight(4)
        self.confidence_bar.setStyleSheet("""
            QProgressBar {
                border: none;
                background-color: #333333;
                border-radius: 2px;
            }
            QProgressBar::chunk {
                background-color: #FFFFFF;
                border-radius: 2px;
            }
        """)
        left_layout.addWidget(self.confidence_bar)
        
        self.status_label = QLabel("Listening for chords...")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("color: #999999; font-size: 12px;")
        left_layout.addWidget(self.status_label)
        
        main_hlayout.addWidget(left_widget)
        
        # Right side: Fretboard widget to display chord positions
        self.fretboard_widget = FretboardWidget()
        self.fretboard_widget.setFixedWidth(250)
        main_hlayout.addWidget(self.fretboard_widget)
        
        self.setCentralWidget(main_widget)
    
    def create_menus(self):
        menu_bar = self.menuBar()
        progression_menu = menu_bar.addMenu("Progression")
        
        # Existing menu actions (optional to use alongside mouse selection)
        standard_action = QAction("Standard (C-F-G-C)", self)
        standard_action.triggered.connect(lambda: self.select_progression(["C", "F", "G", "C"]))
        progression_menu.addAction(standard_action)
        
        iv_iv_action = QAction("I-IV-V (C-F-G)", self)
        iv_iv_action.triggered.connect(lambda: self.select_progression(["C", "F", "G"]))
        progression_menu.addAction(iv_iv_action)
        
        iv_v_vi_iv_action = QAction("I-V-vi-IV (C-G-Am-F)", self)
        iv_v_vi_iv_action.triggered.connect(lambda: self.select_progression(["C", "G", "Am", "F"]))
        progression_menu.addAction(iv_v_vi_iv_action)
        
        finish_manual_action = QAction("Finish Manual Progression", self)
        finish_manual_action.triggered.connect(self.finish_manual_progression)
        progression_menu.addAction(finish_manual_action)
    
    # For menu-based selection (still available)
    def select_progression(self, progression):
        self.manual_progression_active = True
        self.selected_progression = progression[:]  # for display
        self.progression_index = 0
        self.update_progression_display()
        if self.selected_progression:
            self.next_chord_label.setText("Next Chord: " + self.selected_progression[0])
        self.status_label.setText("Manual progression active.")
        self.suggest_signal.emit([])
    
    # Called when a chord button is clicked in the progression selector.
    def add_chord_to_manual_progression(self, chord):
        if not self.manual_progression_active:
            self.manual_progression_active = True
            self.selected_progression = []
            self.progression_index = 0
            self.status_label.setText("Manual progression active.")
        self.selected_progression.append(chord)
        self.update_progression_display()
        if len(self.selected_progression) > self.progression_index:
            next_chord = self.selected_progression[self.progression_index]
            self.next_chord_label.setText("Next Chord: " + next_chord)
    
    def update_progression_display(self):
        if self.manual_progression_active and self.selected_progression:
            displayed = []
            for i, chord in enumerate(self.selected_progression):
                if i == self.progression_index:
                    displayed.append(f"<b><span style='color: #FF0000'>{chord}</span></b>")
                else:
                    displayed.append(chord)
            progression_text = " → ".join(displayed)
            self.progression_label.setText("Manual Progression: " + progression_text)
        elif self.current_progression:
            progression_text = " → ".join(self.current_progression)
            self.progression_label.setText(f"Current Progression: {progression_text}")
        else:
            self.progression_label.setText("Chord Progression: None")
    
    def finish_manual_progression(self):
        self.manual_progression_active = False
        self.selected_progression = []
        self.progression_index = 0
        self.progression_label.setText("Chord Progression: None")
        self.next_chord_label.setText("Next Chord: None")
        self.status_label.setText("Manual progression finished, auto-detect resumed.")
    
    def setup_audio(self):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.SAMPLE_RATE,
            input=True,
            frames_per_buffer=self.BUFFER_SIZE
        )
        
    def audio_processing(self):
        try:
            sys.setrecursionlimit(10000)
            while True:
                try:
                    audio_data = self.stream.read(self.BUFFER_SIZE, exception_on_overflow=False)
                    audio_chunk = np.frombuffer(audio_data, dtype=np.float32)
                    
                    self.audio_buffer = np.roll(self.audio_buffer, -len(audio_chunk))
                    self.audio_buffer[-len(audio_chunk):] = audio_chunk
                    
                    if np.max(np.abs(audio_chunk)) > 0.01:
                        mel_spec = librosa.feature.melspectrogram(
                            y=self.audio_buffer,
                            sr=self.SAMPLE_RATE,
                            n_mels=self.N_MELS,
                            n_fft=self.N_FFT,
                            hop_length=self.HOP_LENGTH
                        )
                        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
                        feature = mel_spec_db[np.newaxis, ..., np.newaxis]
                        
                        prediction = self.model.predict(feature, verbose=0)[0]
                        chord_index = np.argmax(prediction)
                        confidence = prediction[chord_index]
                        detected_chord = self.CHORD_CLASSES[chord_index]
                        
                        threshold = 0.5
                        current_time = time.time()
                        
                        if confidence > threshold:
                            if self.manual_progression_active and self.selected_progression:
                                expected_chord = self.selected_progression[self.progression_index]
                                if detected_chord == expected_chord and current_time - self.last_chord_change_time > 1.0:
                                    self.current_chord = detected_chord
                                    self.last_chord_change_time = current_time
                                    self.update_signal.emit(detected_chord, confidence, audio_chunk)
                                    self.fretboard_widget.update_chord(detected_chord)
                                    
                                    self.progression_index += 1
                                    if self.progression_index < len(self.selected_progression):
                                        next_chord = self.selected_progression[self.progression_index]
                                        self.suggest_signal.emit([next_chord])
                                        self.next_chord_label.setText("Next Chord: " + next_chord)
                                    else:
                                        self.finish_manual_progression()
                                    self.update_progression_display()
                                else:
                                    self.suggest_signal.emit([expected_chord])
                            else:
                                if self.current_chord != detected_chord:
                                    self.current_chord = detected_chord
                                    self.last_chord_change_time = current_time
                                    if not self.current_progression or current_time - self.last_chord_time > self.silence_threshold:
                                        self.current_progression = [detected_chord]
                                    else:
                                        if detected_chord not in self.current_progression:
                                            self.current_progression.append(detected_chord)
                                    self.last_chord_time = current_time
                                    self.get_chord_suggestions(self.current_progression)
                                
                                self.update_signal.emit(detected_chord, confidence, audio_chunk)
                                self.fretboard_widget.update_chord(detected_chord)
                        else:
                            self.update_signal.emit("", 0, audio_chunk)
                    else:
                        self.update_signal.emit("", 0, audio_chunk)
                    
                    time.sleep(0.1)
                except Exception as inner_e:
                    print(f"Inner audio processing error: {inner_e}")
                    time.sleep(0.5)
                    continue
        except Exception as e:
            print(f"Audio processing error: {e}")
            import traceback
            traceback.print_exc()
    
    def get_chord_suggestions(self, current_progression):
        def fetch_suggestions():
            try:
                standard_progressions = {
                    'C': {
                        'C-F': ['G'],
                        'C-G': ['Am', 'F'],
                        'C-Am': ['F', 'G'],
                        'C-F-G': ['C'],
                        'C-G-Am': ['F'],
                        'C-Am-F': ['G'],
                        'C': ['F', 'G', 'Am'],
                    },
                    'F': {
                        'F-Bb': ['C'],
                        'F-C': ['Bb', 'Dm'],
                        'F-Dm': ['Bb', 'C'],
                        'F-Bb-C': ['F'],
                        'F': ['Bb', 'C', 'Dm'],
                    },
                    'G': {
                        'G-C': ['D', 'Em'],
                        'G-D': ['Em', 'C'],
                        'G-Em': ['C', 'D'],
                        'G-C-D': ['G'],
                        'G': ['C', 'D', 'Em'],
                    },
                    'Am': {
                        'Am-Dm': ['E', 'G', 'F'],
                        'Am-F': ['G', 'C', 'Dm'],
                        'Am-G': ['F', 'Dm'],
                        'Am-Dm-G': ['Am', 'C', 'F'],
                        'Am': ['Dm', 'F', 'G'],
                    },
                    'Dm': {
                        'Dm-Gm': ['A', 'Bb', 'C'],
                        'Dm-Bb': ['C', 'F', 'Gm'],
                        'Dm-C': ['Bb', 'F'],
                        'Dm': ['Gm', 'Bb', 'C'],
                    },
                    'Em': {
                        'Em-Am': ['B', 'C', 'D'],
                        'Em-C': ['D', 'G', 'Am'],
                        'Em-D': ['C', 'G'],
                        'Em': ['Am', 'C', 'D'],
                    },
                    'Bdim': {
                        'Bdim-C': ['Am', 'G'],
                        'Bdim-Em': ['Am', 'G'],
                        'Bdim-G': ['C', 'Am'],
                        'Bdim': ['C', 'Em', 'G'],
                    },
                }
                
                progression_key = '-'.join(current_progression)
                if len(current_progression) > 0:
                    last_chord = current_progression[-1]
                    if last_chord in standard_progressions:
                        if progression_key in standard_progressions[last_chord]:
                            suggestions = standard_progressions[last_chord][progression_key]
                        else:
                            suggestions = standard_progressions[last_chord].get(last_chord, [])
                    else:
                        suggestions = self.CHORD_CLASSES[:3]
                    
                    suggestions = [c for c in suggestions if c in self.CHORD_CLASSES]
                    
                    if suggestions:
                        suggestions = suggestions[:3]
                        self.suggest_signal.emit(suggestions)
                self.update_progression_display()
            except Exception as e:
                print(f"Error getting suggestions: {e}")
                import traceback
                traceback.print_exc()
        
        suggestion_thread = threading.Thread(target=fetch_suggestions)
        suggestion_thread.daemon = True
        suggestion_thread.start()
    
    def update_display(self, chord, confidence, audio_chunk):
        self.chord_display.update_chord(chord, confidence)
        self.confidence_bar.setValue(int(confidence * 100))
        self.waveform_widget.update_waveform(audio_chunk)
        if chord:
            self.status_label.setText(f"Detected: {chord} ({confidence:.2f})")
        else:
            self.status_label.setText("Listening for chords...")
    
    def show_suggestions(self, suggestions):
        # Remove previous suggestion bubbles (if any)
        for i in reversed(range(1, self.suggestions_layout.count())):
            item = self.suggestions_layout.itemAt(i)
            if item.widget() and item.widget().objectName() == "suggestion":
                widget = item.widget()
                self.suggestions_layout.removeItem(item)
                widget.deleteLater()
        for chord in suggestions:
            bubble = SuggestionBubble(chord)
            bubble.setObjectName("suggestion")
            self.suggestions_layout.addWidget(bubble)
    
    def closeEvent(self, event):
        if hasattr(self, 'stream') and self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
        if hasattr(self, 'p') and self.p is not None:
            self.p.terminate()
        event.accept()

# -------------------------------
# Suggestion Bubble Widget (unchanged)
# -------------------------------
class SuggestionBubble(QWidget):
    def __init__(self, chord, parent=None):
        super(SuggestionBubble, self).__init__(parent)
        self.chord = chord
        self._opacity = 1.0
        self.setFixedSize(60, 60)
        
        chord_map = {
            'C': 0.1,
            'Dm': 0.2,
            'Em': 0.33,
            'F': 0.5,
            'G': 0.6,
            'Am': 0.8,
            'Bb': 0.9,
            'Bdim': 0.95
        }
        
        self.hue = chord_map.get(chord, random.random())
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = self.rect()
        
        color = QColor.fromHsvF(self.hue, 0.8, 0.9, self._opacity)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(color))
        painter.drawEllipse(rect)
        
        painter.setPen(QColor(255, 255, 255, int(255 * self._opacity)))
        painter.setFont(QFont("Arial", 14, QFont.Bold))
        painter.drawText(rect, Qt.AlignCenter, self.chord)
        
    def fade_out(self):
        self.animation = QPropertyAnimation(self, b"bubble_opacity")
        self.animation.setDuration(1000)
        self.animation.setStartValue(1.0)
        self.animation.setEndValue(0.0)
        self.animation.setEasingCurve(QEasingCurve.OutQuad)
        self.animation.finished.connect(self.deleteLater)
        self.animation.start()
        
    def get_bubble_opacity(self):
        return self._opacity
        
    def set_bubble_opacity(self, opacity):
        if self._opacity != opacity:
            self._opacity = opacity
            self.update()
        
    bubble_opacity = pyqtProperty(float, get_bubble_opacity, set_bubble_opacity)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = ChordWizardApp()
    main_window.show()
    sys.exit(app.exec_())
