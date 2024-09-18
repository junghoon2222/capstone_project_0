import pyaudio
import numpy as np
import websockets
import asyncio
import os
import re
import time
from openai import OpenAI
import dotenv
from copy import deepcopy
from datetime import datetime

import whisper

from fastAPI import FastAPI
import aiohttp
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

from bs4 import BeautifulSoup
import json
from pre import *

def contains_japanese(text):
    japanese_pattern = re.compile(r"[\u3040-\u30FF\u4E00-\u9FFF]")
    return bool(japanese_pattern.search(text))


def contains_english(text):
    english_pattern = re.compile(r"[A-Za-z]")
    return bool(english_pattern.search(text))


def filter_text(text):
    if not text.strip():
        return True
    if contains_japanese(text) and contains_english(text):
        return True
    if contains_japanese(text):
        return True
    if contains_english(text):
        return True
    else:
        return False


_classifier_init = [
    {
        "role": "system",
        "content": """
        Based on the conversation context, determine if the user's last message requires real-time or frequently updated information. Follow these steps to evaluate:
        
        1. Is the user's last message a follow-up question related to the previous conversation?
        2. Does the message require real-time information, periodically updated data, or information that could change based on external conditions?
        3. Did the user ask to check or search for something?

        If the answer to any of these questions is 'yes,' output '0'. If none of these apply, output '1'.

        Only output '0' or '1'.

        Example:

        Context###
        user: I need to check some stock prices
        assistant: Sure, let me know which company's stock price you are looking for.
        user: Apple and Tesla
        assistant: The current stock price for Apple is $178.50, and for Tesla, it is $730.23. These are the latest figures from the stock exchange.
        user: What about yesterday's closing prices?
        assistant: Yesterday, Apple closed at $176.80 and Tesla at $725.65.
        user: Any updates for today?
        assistant: As of now, there have been no major announcements from Apple or Tesla regarding their stock prices. Would you like to track the changes throughout the day?
        user: Actually, can you show me a chart comparing them?

        0
        """,
    },
]

_keyword_maker_init = [
    {
        "role": "system",
        "content": """
        Your role is only to make keywords or a phrase to find out user's question from one's word.
        You'd better not put on any unnessesary words.
    """,
    }
]


_html_rag_init = [
    {
        "role": "system",
        "content": """     
        Based on the context, generate an appropriate response to the user. 
        If the execution was successful, provide the output. 
        If there was an error or issue, analyze the error and suggest a correction. 
        Do not include any unnecessary explanations or comments—output only the necessary response. 
        If the execution results include any sensitive information, handle it appropriately. 
        Don't provide any links and just provide shortly up to 3 sentences
        Use friendly-tone.

     """,
    }
]

_chat_messages_init = [
    {
        "role": "system",
        "content": """    
        You are an assistant in a smart mirror.
        When a user asks about their appearance or initiates a conversation related to their appearance, respond positively but avoid being insincere.
        Limit your responses to a maximum of two sentences, and avoid using emojis or emoticons. 
        Use friendly-tone.
        Only speak in Korean.
        
    """,
    }
]



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


async def record_and_send_audio(websocket):

    # chat_messages = chat_messages_init
    dotenv.load_dotenv()

    client = OpenAI()

    classifier_init = deepcopy(_classifier_init)
    keyword_maker_init = deepcopy(_keyword_maker_init)
    html_rag_init = deepcopy(_html_rag_init)
    chat_messages = deepcopy(_chat_messages_init)

    while True:
        async for message in websocket:
            print("Received audio data")

            audio = np.frombuffer(message, dtype=np.int16).astype(np.float32) / 32768.0

            result = whisper_model.transcribe(audio)
            text = result["text"].strip()

            if not filter_text(text):

                os.system("cls" if os.name == "nt" else "clear")
                print(text)
                
        input_message = text
        
        if input_message and input_message.strip():

            chat_messages.append({"role": "user", "content": input_message})
            await websocket.send("input " + input_message)

        else:
            continue

        context = ["Context###\n\n"]

        now = datetime.now()

        formatted_time = now.strftime("%Y-%m-%d %H:%M")

        context.append("CurrentTime :" + formatted_time)

        for message in chat_messages[1:]:
            context.append(": ".join(message.values()))

        context = " ".join(context)
        
        context = {"role": "user", "content": context}

        classifier_init.append(context)

        classifier = client.chat.completions.create(
            model="gpt-4o", messages=classifier_init
        )

        query_class = classifier.choices[0].message.content

        classifier_init = deepcopy(_classifier_init)
        print(query_class)

        if query_class == "0":
            keyword_maker_init.append({"role": "user", "content": input_message})

            keyword_maker = client.chat.completions.create(
                model="gpt-4o-mini-2024-07-18", messages=keyword_maker_init
            )

            keyword = keyword_maker.choices[0].message.content

            print("keyword: ", keyword)

            _html = crawl(keyword)

            html_rag_init.append({"role": "user", "content": _html})

            html_rag = client.chat.completions.create(
                model="gpt-4o-mini-2024-07-18", messages=html_rag_init
            )

            answer = html_rag.choices[0].message.content
            print(answer)
            await websocket.send("output " + answer)

            chat_messages.append({"role": "assistant", "content": answer})

            init_time = time.time()
            keyword_maker_init = deepcopy(_keyword_maker_init)
            html_rag_init = deepcopy(_html_rag_init)

        if query_class == "1":

            chatter = client.chat.completions.create(
                model="gpt-4o-mini-2024-07-18", messages=chat_messages
            )
            answer = chatter.choices[0].message.content
            await websocket.send("output" + answer)
            print(answer)

            chat_messages.append({"role": "assistant", "content": answer})

            init_time = time.time()

        while True:
            async for message in websocket:
                print("Received audio data")

                audio = np.frombuffer(message, dtype=np.int16).astype(np.float32) / 32768.0

                result = whisper_model.transcribe(audio)
                text = result["text"].strip()

                if not filter_text(text):

                    os.system("cls" if os.name == "nt" else "clear")
                    print(text)
                    
            input_message = text
            
            if input_message and input_message.strip():
                chat_messages.append({"role": "user", "content": input_message})
                await websocket.send("input " + input_message)

            else:
                continue

            context = ["Context###\n\n"]
            now = datetime.now()

            formatted_time = now.strftime("%Y-%m-%d %H:%M")

            context.append("CurrentTime :" + formatted_time)

            for message in chat_messages[1:]:
                context.append(": ".join(message.values()))

            context = " ".join(context)

            context = {"role": "user", "content": context}
            classifier_init = deepcopy(_classifier_init)

            classifier_init.append(context)

            print(context)

            classifier = client.chat.completions.create(
                model="gpt-4o", messages=classifier_init
            )

            query_class = classifier.choices[0].message.content

            # print(query_class)

            if query_class == "0":
                keyword_maker_init.append({"role": "user", "content": input_message})

                keyword_maker = client.chat.completions.create(
                    model="gpt-4o-mini-2024-07-18", messages=keyword_maker_init
                )

                keyword = keyword_maker.choices[0].message.content

                print("keyword: ", keyword)

                _html = crawl(keyword)

                html_rag_init.append({"role": "user", "content": _html})

                html_rag = client.chat.completions.create(
                    model="gpt-4o-mini-2024-07-18", messages=html_rag_init
                )

                answer = html_rag.choices[0].message.content
                print(answer)
                await websocket.send("output " + answer)
                chat_messages.append({"role": "assistant", "content": answer})

                html_rag_init = deepcopy(_html_rag_init)
                if time.time() - init_time > 180:
                    chat_messages = deepcopy(_chat_messages_init)
                    print("3 minute has flew")
                    break

            if query_class == "1":

                chatter = client.chat.completions.create(
                    model="gpt-4o-mini-2024-07-18", messages=chat_messages
                )
                answer = chatter.choices[0].message.content
                chat_messages.append({"role": "assistant", "content": answer})
                print(answer)
                await websocket.send("output " + answer)
                if time.time() - init_time > 180:
                    chat_messages = deepcopy(_chat_messages_init)
                    print("3 minute has flew")
                    break



@app.get("/get_weather")
async def get_weather():
    async with aiohttp.ClientSession() as session:
        async with session.get(
            "https://search.naver.com/search.naver?sm=tab_hty.top&where=nexearch&ssc=tab.nx.all&query=%EA%B5%AC%EB%AF%B8+%EB%82%A0%EC%94%A8"
        ) as response:

            if response.status == 200:
                content = await response.text()

                soup = BeautifulSoup(content, "lxml")

                status_wrap_text = soup.find(class_="status_wrap").get_text()
                temperature = soup.find(class_="temperature_inner").get_text()

                final = []

                result = re.split(r"\s{2,}", status_wrap_text)
                for i, w in enumerate(result):
                    if i == 0 or i == 1 or i == 6 or i == 7 or i == 8 or i == 13:
                        continue
                    final.append(w.strip())

                final_dict = {}

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

                return final_dict
            else:
                return {"error": "Failed to fetch weather data"}



if __name__ == "__main__":
    # FastAPI 앱 실행
    uvicorn.run(app, host="0.0.0.0", port=50006)  # 날씨 엔드포인트 (프론트에서 접근)

    # Audio 전송 소켓 서버 실행
    start_server_2 = websockets.serve(record_and_send_audio, "0.0.0.0", 50008)
    print("WebSocket server started on ws://localhost:50008")

    asyncio.run(start_server_2)
