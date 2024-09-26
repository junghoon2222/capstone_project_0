import asyncio
import websockets
import whisper
import numpy as np
from scipy.signal import resample_poly
from scipy.io.wavfile import write
from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from scipy.signal import resample

model_name = "large-v3"
whisper_model = whisper.load_model(model_name, device="cuda")
print("Whisper model loaded")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def resample_audio(audio_data, original_rate, target_rate):
    number_of_samples = round(len(audio_data) * target_rate / original_rate)
    resampled_audio = resample(audio_data, number_of_samples)
    return resampled_audio


def bandpass_filter(audio, sample_rate, lowcut, highcut):
    # 푸리에 변환
    fft_audio = np.fft.rfft(audio)
    frequencies = np.fft.rfftfreq(len(audio), d=1/sample_rate)

    # 주파수 필터링
    fft_audio[(frequencies < lowcut) | (frequencies > highcut)] = 0

    # 역 푸리에 변환
    filtered_audio = np.fft.irfft(fft_audio)
    return filtered_audio.astype(np.float32)

async def transcribe(websocket):
    threshold = 2000  # 임계값 설정
    target_sample_rate = 16000  # 목표 샘플레이트
    lowcut = 300  # 저주파수 컷오프
    highcut = 3400  # 고주파수 컷오프
    data = bytearray()

    message = await websocket.recv()

    # 오디오 데이터를 정수형으로 변환
    audio = np.frombuffer(message, dtype=np.float32)

    # 절대값으로 변경
    abs_audio = np.abs(audio)*32768
    mean_value = np.mean(abs_audio)

    if mean_value > threshold:
        data.extend(message)

        while True:
            message = await websocket.recv()
            audio = np.frombuffer(message, dtype=np.float32)
            abs_audio = np.abs(audio)*32768 
            mean_value = np.mean(abs_audio)
            data.extend(message)
            
            # print(mean_value)
            
            if mean_value < threshold:
                silence_first_time = asyncio.get_event_loop().time()
                while asyncio.get_event_loop().time() - silence_first_time < 1:
                    message = await websocket.recv()
                    audio = np.frombuffer(message, dtype=np.float32)
                    abs_audio = np.abs(audio)*32768
                    mean_value = np.mean(abs_audio)

                    data.extend(message)
                    if mean_value > threshold:
                        silence_first_time = asyncio.get_event_loop().time()
                break

        # 오디오 데이터를 float32로 변환
        audio = np.frombuffer(data, dtype=np.float32)

        # 리샘플링
        resampled_audio = resample_audio(audio, original_rate=48000, target_rate=16000)

        # 현재 시각을 기반으로 파일 이름 생성
        now = datetime.now()
        filename = f"{now.hour}_{now.minute}.wav"

        # 리샘플된 오디오를 파일로 저장
        write(filename, target_sample_rate, resampled_audio)
        print(f"Resampled audio saved as {filename}")

        # 주파수 필터링
        filtered_audio = bandpass_filter(resampled_audio, target_sample_rate, lowcut, highcut)

        # Whisper 모델로 음성 인식
        result = whisper_model.transcribe(filtered_audio)
        text = result["text"].strip()
        print("Transcription result:", text)
        return text






            
async def main():
    start_server = websockets.serve(transcribe, "0.0.0.0", 50007)
    print("WebSocket server started on ws://localhost:50007")
    await start_server
    await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(main())
