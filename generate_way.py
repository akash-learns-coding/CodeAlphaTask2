from gtts import gTTS
from pydub import AudioSegment
import os

# Dataset emotions and example sentences
emotions = {
    "angry": "I am really upset right now!",
    "happy": "I am feeling very happy today!",
    "sad": "I am so sad and disappointed.",
    "neutral": "This is just a normal day."
}

# Make dataset directories
if not os.path.exists("dataset"):
    os.makedirs("dataset")

for emotion, text in emotions.items():
    folder = os.path.join("dataset", emotion)
    os.makedirs(folder, exist_ok=True)

    # Generate TTS in mp3
    tts = gTTS(text, lang="en")
    mp3_path = os.path.join(folder, f"{emotion}_1.mp3")
    wav_path = os.path.join(folder, f"{emotion}_1.wav")

    tts.save(mp3_path)

    # Convert to WAV
    sound = AudioSegment.from_mp3(mp3_path)
    sound.export(wav_path, format="wav")

    print(f"Created {wav_path}")

print(" All WAV files generated successfully!")
