import os
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import librosa.display
from pydub import AudioSegment
from pydub.silence import split_on_silence
from TTS.api import TTS


###
# Getting started:
# pip install TTS librosa soundfile numpy matplotlib pydub
###

# ==============================
# CONFIGURATION
# ==============================

INPUT_FILE = "steve_jobs_2005_raw.mp3"                    # speech
INPUT_SAMPLE_FILE = "steve_jobs_2005_raw_sample.wav"      # speech snippet
CLEAN_FILE = "jobs_clean.wav"
REFERENCE_FILE = "jobs_reference.wav"
SYNTH_FILE = "jobs_synthetic.wav"

TARGET_SR = 22050
REFERENCE_DURATION_SEC = 90            # short snipped
SYNTH_TEXT = "Today, we are introducing the new iPhone 2026 made of wood!"

# ==============================
# STEP 1 – PREPROCESS AUDIO
# ==============================

def preprocess_audio():
    print("Loading audio...")
    y, sr = librosa.load(INPUT_FILE, sr=TARGET_SR, mono=True)

    print("Normalizing...")
    y = y / np.max(np.abs(y))

    print("Saving cleaned audio...")
    sf.write(CLEAN_FILE, y, TARGET_SR)

# ==============================
# STEP 2 – EXTRACT REFERENCE CLIP
# ==============================

def extract_reference():
    print("Extracting reference segment...")

    audio = AudioSegment.from_wav(CLEAN_FILE)

    chunks = split_on_silence(
        audio,
        min_silence_len=700,
        silence_thresh=-40
    )

    combined = AudioSegment.empty()
    duration_needed = REFERENCE_DURATION_SEC * 1000

    for chunk in chunks:
        combined += chunk
        if len(combined) >= duration_needed:
            break

    combined = combined[:duration_needed]
    combined.export(REFERENCE_FILE, format="wav")

    print("Reference clip saved.")

# ==============================
# STEP 3 – VOICE CLONING
# ==============================

def generate_synthetic():
    print("Loading XTTS model (first time may take a while)...")

    tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")

    print("Generating synthetic speech...")

    tts.tts_to_file(
        text=SYNTH_TEXT,
        speaker_wav=REFERENCE_FILE,
        language="en",
        file_path=SYNTH_FILE
    )

    print("Synthetic audio saved.")

# ==============================
# STEP 4 – SPECTROGRAM ANALYSIS
# ==============================

def plot_spectrogram(file, title):
    y, sr = librosa.load(file)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)

    plt.figure(figsize=(10, 6))
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.savefig('m05_03_spectro.png', dpi=300)
    plt.show()

def plot_comparison(original_path, synthetic_path):
    # Audio laden
    y_orig, sr_orig = librosa.load(original_path, sr=16000)
    y_syn, sr_syn = librosa.load(synthetic_path, sr=16000)

    # Mel-Spektrogramme berechnen
    mel_orig = librosa.feature.melspectrogram(y=y_orig, sr=sr_orig)
    mel_syn = librosa.feature.melspectrogram(y=y_syn, sr=sr_syn)

    mel_orig_db = librosa.power_to_db(mel_orig, ref=np.max)
    mel_syn_db = librosa.power_to_db(mel_syn, ref=np.max)

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    librosa.display.specshow(mel_orig_db, sr=sr_orig, x_axis='time', y_axis='mel')
    plt.title("Original - Mel Spectrogram")
    plt.colorbar(format='%+2.0f dB')
    #plt.savefig('m05_03_spectro_origi.png', dpi=300), nur original printen..

    plt.subplot(2, 1, 2)
    plt.subplots_adjust(hspace=0.35)
    librosa.display.specshow(mel_syn_db, sr=sr_syn, x_axis='time', y_axis='mel')
    plt.title("Synthetic - Mel Spectrogram")
    plt.colorbar(format='%+2.0f dB')
    plt.savefig('m05_03_spectro_compare.png', dpi=300)

    plt.tight_layout()
    plt.show()

def analyze():
    print("Generating spectrogram comparison...")
    plot_spectrogram(REFERENCE_FILE, "Original Voice (Reference)")
    plot_spectrogram(SYNTH_FILE, "Synthetic Voice")
    plot_comparison(INPUT_SAMPLE_FILE, SYNTH_FILE)

# ==============================
# MAIN PIPELINE
# ==============================

def main():
    preprocess_audio()
    extract_reference()
    generate_synthetic()
    analyze()
    print("Pipeline complete.")

if __name__ == "__main__":
    main()
