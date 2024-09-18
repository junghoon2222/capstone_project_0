import re
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import socket
import numpy as np
import threading
import io
import torch
import soundfile as sf
import datetime
import os

from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from TTS.api import TTS

import deepl

from main.assistants.modules.LLM import GPT

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def get_whisper(model):
    torch_dtype = torch.float32 if model == "whisper" else torch.float32

    if model == "whisper":
        model_id = "openai/whisper-large-v3"
    if model == "d_whisper":
        model_id = "distil-whisper/distil-large-v3"

    return model_id, torch_dtype

def do_whisper(audio_buffer, dtype):
    audio, sample_rate = sf.read(audio_buffer, dtype='float64')
    inputs = processor(audio, sampling_rate=sample_rate, return_tensors="pt")
    input_features = inputs.input_features.to(device, dtype=torch.float32) 
    gen_kwargs = {
        "max_new_tokens": 50,
        "num_beams": 1,
        "do_sample": True,
        "temperature": 0.4,
        "return_timestamps": False,
        "language": "korean"
    }

    # Whisper process
    with torch.no_grad():
        pred_ids = whisper.generate(input_features, **gen_kwargs)
        user_text = processor.batch_decode(pred_ids, skip_special_tokens=True)[0].strip()

    if user_text in filtered_phrases:
        print(f"Filtered phrase detected: {user_text}")
        pass

    if user_text:
        print(f"Sending user_text: {user_text}")
    else:
        user_text = "."

    return user_text

model_id, torch_dtype = get_whisper("whisper")

torch_dtype = torch.float32

whisper = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True)
whisper.to(device)
processor = AutoProcessor.from_pretrained(model_id)
filtered_phrases = ["감사합니다.", "네.", "감사합니다", ".", "네", "아멘"]

DEEPL_API_KEY = "da238b27-7185-45d5-9724-e10ba6d58a74:fx"
deepl_translator = deepl.Translator(DEEPL_API_KEY)

# tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

@app.post("/process_audio")
async def process_audio(file: UploadFile = File(...)):
    try:
        # 클라이언트로부터 오디오 데이터 수신
        audio_data = await file.read()
        
        audio_buffer = io.BytesIO(audio_data)
        audio_buffer.seek(0)

        timestamp = datetime.datetime.now().strftime("%m%d%H%M%S")

        # 오디오 데이터를 WAV 파일로 저장
        received_wav_file_path = f"files/received_audio_{timestamp}.mp3"
        with open(received_wav_file_path, 'wb') as f:
            f.write(audio_buffer.read())
        print(f"Audio saved to {received_wav_file_path}")

        # Whisper process
        audio_buffer.seek(0)  # Whisper 처리를 위해 다시 스트림의 시작 위치로 이동

        user_text = do_whisper(audio_buffer, torch_dtype)
        if user_text in filtered_phrases:
            return JSONResponse(status_code=200, content={timestamp: {"user_text": user_text, "gpt_text": "Filtered"}})
        
        # LLM process
        gpt_text = GPT(user_text)
        print(f"LLama: {gpt_text}")

        return JSONResponse(status_code=200, content={timestamp: {"user_text": user_text, "gpt_text": gpt_text}})

    except Exception as e:
        print(f"Error Process Audio: {e}")
        raise HTTPException(status_code=500, detail=f"Error Process Audio: {str(e)}")

if __name__ == '__main__':
    if not os.path.exists('./files'):
        os.makedirs('./files')
    import uvicorn
    uvicorn.run(app, host='localhost', port=7000)
