import numpy as np
import soundfile as sf
import time
from datetime import datetime
from queue import Queue
from collections import OrderedDict
import threading
from collections import deque
import sounddevice as sd
import noisereduce as nr
import librosa
from pydub import AudioSegment
import json
import os
import requests

samplerate = 22050
blocksize = 1024
vad_threshold = 0.02
min_audio_length = 4000

audio_queue = deque()
audio_lock = threading.Lock()
result_queue = Queue()

save_dir = 'files'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    with audio_lock:
        audio_queue.append(indata.copy())
    print(f"Appended {frames} frames to audio_queue")

def preprocess_audio(audio_data, original_sr=44100, target_sr=16000):
    audio_data = librosa.resample(audio_data, orig_sr=original_sr, target_sr=target_sr)
    audio_data = nr.reduce_noise(y=audio_data, sr=target_sr)
    mfccs = librosa.feature.mfcc(y=audio_data, sr=target_sr, n_mfcc=13)
    normalized_audio = librosa.util.normalize(audio_data)
    return normalized_audio

def convert_to_mp3(wav_path, mp3_path):
    audio = AudioSegment.from_wav(wav_path)
    audio.export(mp3_path, format="mp3")

def process_audio(flask1_url):
    collected_audio = []
    silence_start_time = None

    while True:
        if len(audio_queue) > 0:
            with audio_lock:
                audio_data = np.concatenate(audio_queue, axis=0).flatten()
                audio_queue.clear()
            print(f"Processing {len(audio_data)} frames")
            current_energy = np.mean(np.abs(audio_data))
            print(f"Current energy: {current_energy}")

            if current_energy >= vad_threshold:
                collected_audio.append(audio_data)
                silence_start_time = time.time()
            else:
                if silence_start_time is not None and time.time() - silence_start_time > 0.5 and len(collected_audio) > 0:
                    if len(np.concatenate(collected_audio)) > min_audio_length:
                        full_audio = np.concatenate(collected_audio)
                        timestamp = datetime.now().strftime("%y%m%d%H%M%S")
                        received_audio_path = f"./files/received_audio_{timestamp}.wav"
                        save_audio_file(received_audio_path, full_audio)

                        with open(received_audio_path, 'rb') as f:
                            files = {'audio': (f'received_audio_{timestamp}.wav', f.read(), 'audio/wav')}
                            response = requests.post(flask1_url, files=files)
                            
                            if response.status_code != 200:
                                print(f"Error from Flask(1): {response.text}")
                                continue

                        result = response.json()
                        first_key = next(iter(result))
                        first_value = result[first_key]

                        result_queue.put(first_value)
                        save_logs(result)
                        print(f"Result added to queue: {first_value}")

                    collected_audio = []
                    with audio_lock:
                        audio_queue.clear()
                        print("Silence detected for a while, stopping recording and clearing audio queue.")
        else:
            time.sleep(0.1)

def save_audio_file(filepath, audio_data):
    try:
        sf.write(filepath, audio_data, samplerate)
    except Exception as e:
        print(f"Error saving audio file: {e}")

def save_logs(result):
    if os.path.exists("logs.json"):
        with open("logs.json", "r", encoding="utf-8") as logs:
            logs = json.load(logs, object_pairs_hook=OrderedDict)
    else:
        logs = OrderedDict()

    new_logs = OrderedDict([(k, v) for k, v in result.items()] + list(logs.items()))

    with open("logs.json", "w", encoding="utf-8") as logs:
        json.dump(new_logs, logs, ensure_ascii=False, indent="\t")

def start_audio_processing(flask1_url):
    stream = sd.InputStream(callback=audio_callback, channels=1, samplerate=samplerate, blocksize=blocksize)
    stream.start()

    audio_thread = threading.Thread(target=process_audio, args=(flask1_url,))
    audio_thread.start()
