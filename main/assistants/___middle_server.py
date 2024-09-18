import os
import soundfile as sf
import numpy as np
from datetime import datetime
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx
import threading
import json
from collections import deque, OrderedDict
import sounddevice as sd
import time
import noisereduce as nr
import librosa
from pydub import AudioSegment

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

flask1_url = 'http://localhost:7000/process_audio'
save_dir = 'files'
result_queue = deque()

samplerate = 16000
blocksize = 16000
vad_threshold = 0.005
min_audio_length = 4000

audio_queue = deque()
audio_lock = threading.Lock()

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    with audio_lock:
        audio_queue.append(indata.copy())
    print(f"Appended {frames} frames to audio_queue")

def preprocess_audio(audio_data, original_sr=44100, target_sr=16000):
    # 샘플링 레이트 변환
    audio_data = librosa.resample(audio_data, orig_sr=original_sr, target_sr=target_sr)
    
    # 노이즈 제거
    audio_data = nr.reduce_noise(y=audio_data, sr=target_sr)
    
    # 정규화
    normalized_audio = librosa.util.normalize(audio_data)
    
    return normalized_audio

def save_audio_file(filepath, audio_data, samplerate):
    try:
        sf.write(filepath, audio_data, samplerate)
    except Exception as e:
        print(f"Error saving audio file: {e}")

def convert_to_mp3(audio_data, samplerate, mp3_path):
    # NumPy array를 직접 MP3로 변환
    audio_segment = AudioSegment(
        data=audio_data.tobytes(),
        sample_width=audio_data.dtype.itemsize,
        frame_rate=samplerate,
        channels=1
    )
    audio_segment.export(mp3_path, format="mp3")

def process_audio():
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
                        
                        # 전처리 및 MP3 변환 적용
                        preprocessed_audio = preprocess_audio(full_audio, original_sr=samplerate, target_sr=samplerate)
                        mp3_path = f"./files/preprocessed_audio_{timestamp}.mp3"
                        convert_to_mp3(preprocessed_audio, samplerate, mp3_path)

                        # 전처리된 MP3 파일 전송
                        with open(mp3_path, 'rb') as f:
                            files = {'audio': (f'preprocessed_audio_{timestamp}.mp3', f.read(), 'audio/mp3')}
                            response = requests.post(flask1_url, files=files)
                            
                            if response.status_code != 200:
                                print(f"Error from Flask(1): {response.text}")
                                continue

                        result = response.json()
                        first_key = next(iter(result))
                        first_value = result[first_key]

                        result_queue.append(first_value)
                        save_logs(result)
                        print(f"Result added to queue: {first_value}")

                    collected_audio = []
                    with audio_lock:
                        audio_queue.clear()
                        print("Silence detected for a while, stopping recording and clearing audio queue.")
        else:
            time.sleep(0.1)

def save_logs(result):
    log_file_path = "logs.json"

    # 로그 파일이 없으면 생성
    if not os.path.exists(log_file_path):
        with open(log_file_path, "w", encoding="utf-8") as logs:
            json.dump({}, logs)

    # 기존 로그 파일을 읽고 새로운 로그를 추가
    with open(log_file_path, "r", encoding="utf-8") as logs:
        logs_data = json.load(logs, object_pairs_hook=OrderedDict)

    new_logs = OrderedDict([(k, v) for k, v in result.items()] + list(logs_data.items()))

    # 새로운 로그를 저장
    with open(log_file_path, "w", encoding="utf-8") as logs:
        json.dump(new_logs, logs, ensure_ascii=False, indent="\t")

@app.post("/process_audio")
async def process_audio_request(file: UploadFile = File(...)):
    try:
        audio_data = await file.read()
        timestamp = datetime.now().strftime("%y%m%d%H%M%S")
        received_audio_path = f"./files/received_audio_{timestamp}.mp3"

        with open(received_audio_path, 'wb') as f:
            f.write(audio_data)

        async with httpx.AsyncClient() as client:
            files = {'audio': (f'received_audio_{timestamp}.mp3', audio_data, 'audio/mp3')}
            response = await client.post(flask1_url, files=files)
            
            if response.status_code != 200:
                return JSONResponse(status_code=500, content={"error": f"Error from Flask(1): {response.text}"})

        result = response.json()
        first_key = next(iter(result))
        first_value = result[first_key]

        result_queue.append(first_value)
        save_logs(result)
        print(f"Result added to queue: {first_value}")

        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Error processing audio: {str(e)}"})

@app.get("/get_result")
async def get_result():
    if result_queue:
        result = result_queue.popleft()
        return JSONResponse(content=result)
    return JSONResponse(content={})

if __name__ == '__main__':
    # 오디오 스트림 시작
    stream = sd.InputStream(callback=audio_callback, channels=1, samplerate=samplerate, blocksize=blocksize)
    stream.start()

    # 오디오 처리 스레드 시작
    audio_thread = threading.Thread(target=process_audio)
    audio_thread.start()

    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=7001)
