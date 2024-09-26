import asyncio
import websockets
import whisper
import numpy as np

from fastapi import FastAPI
import requests
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

from bs4 import BeautifulSoup
import re
import json


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


async def transcribe(websocket, path):
    async for message in websocket:
        print("Received audio data")

        audio = np.frombuffer(message, dtype=np.int16).astype(np.float32) / 32768.0

        result = whisper_model.transcribe(audio)
        text = result["text"].strip()
        print("Transcription result:", text)

        await websocket.send(text)


@app.get("/get_weather")
def get_weather():
    response = requests.get(
        "https://search.naver.com/search.naver?sm=tab_hty.top&where=nexearch&ssc=tab.nx.all&query=%EA%B5%AC%EB%AF%B8+%EB%82%A0%EC%94%A8"
    )

    if response.status_code == 200:

        soup = BeautifulSoup(response.content, "lxml")

        # 'main_pack' ID를 가진 요소를 찾음
        status_wrap_text = soup.find(class_="status_wrap").get_text()
        temperature = soup.find(class_="temperature_inner").get_text()

    final = []

    result = re.split(r"\s{2,}", status_wrap_text)
    for i, w in enumerate(result):
        if i == 0 or i == 1 or i == 6 or i == 7 or i == 8 or i == 13:
            continue
        final.append(w.strip())

    final_dict = {}

    # final_dict['current_temp'] = final[0]
    # final_dict['current_temp_icon']
    for i in temperature.split("/"):
        final.append(i.strip())
    final_dict = dict(
        current_temp=final[0][5:],
        diff_temp=final[1].split(" ")[1],
        weather=final[2],
        weather_icon="https://ssl.pstatic.net/sstatic/keypage/outside/scui/weather_new_new/img/weather_svg_v2/icon_flat_wt1.svg",
        feeling_temp=final[3].split(" ")[1],
        dust=final[4].split(" ")[1],
        mini_dust=final[5].split(" ")[1],
        ultrawave=final[6].split(" ")[1],
        sun=final[7].split(" ")[1],
        min_temp=final[-2][4:],
        max_temp=final[-1][4:],
    )
    # json_final_dict = json.dumps(final_dict, ensure_ascii=False)

    return final_dict


if __name__ == "__main__":
    # uvicorn.run(app, host="0.0.0.0", port=7002)

    start_server = websockets.serve(transcribe, "0.0.0.0", 50007)

    asyncio.get_event_loop().run_until_complete(start_server)
    print("WebSocket server started on ws://localhost:50007")
    asyncio.get_event_loop().run_forever()

    # print(get_weather())
