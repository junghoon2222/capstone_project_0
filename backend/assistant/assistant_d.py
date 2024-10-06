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
import ssl
import whisper

from fastapi import FastAPI
import aiohttp
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

from bs4 import BeautifulSoup
from pre import *

from scipy.signal import resample_poly
from scipy.io.wavfile import write
from datetime import datetime
from scipy.signal import resample

from melo.api import TTS
from pydub import AudioSegment
from pydub.playback import play
import io
# Speed is adjustable


def contains_japanese(text):
    japanese_pattern = re.compile(r"[\u3040-\u30FF\u4E00-\u9FFF]")
    return bool(japanese_pattern.search(text))

def contains_english(text):
    english_pattern = re.compile(r"[A-Za-z]")
    return bool(english_pattern.search(text))

def contains_korean(text):
    korean_pattern = re.compile(r"[\uAC00-\uD7AF\u1100-\u11FF\u3130-\u318F]")
    return bool(korean_pattern.search(text))

def contains_special_characters(text):
    special_char_pattern = re.compile(r"[^\w\s]")
    return bool(special_char_pattern.search(text))

def filter_text(text):
    if text is None or not text.strip():
        return True
    if not contains_korean(text):
        return True
    return False


now = datetime.now()
formatted_time = now.strftime("%Y-%m-%d %H:%M")
print(formatted_time)

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
        "content": f"""
        current date and time are {formatted_time}    
        current location is 구미, so you should make keywords or a phrase related to 구미 when about weather or traffic.
        Your role is only to make keywords or a phrase to find out user's question from one's word.
        You'd better not put on any unnessesary words.
    """,
    }
]



_html_rag_init = [
    {
        "role": "system",
        "content": f"""
        
        current date and time are {formatted_time}    
        Based on the context, generate an appropriate response to the user. 
        If the execution was successful, provide the output. 
        If there was an error or issue, analyze the error and suggest a correction. 
        Do not include any unnecessary explanations or comments—output only the necessary response. 
        If the execution results include any sensitive information, handle it appropriately. 
        Don't provide any links and just provide shortly up to 3 sentences
        Use friendly-tone.
        Only speak in Korean.
        have attention to the user's question.
        
        You are 이지윤, a 23-year-old woman living in South Korea. You are playful and sometimes bicker, but you are a loving girlfriend. You have the following personality traits and characteristics:
        - You think analytically and logically and enjoy solving complex problems.
        - You are curious and have a deep interest in a variety of topics.
        - You explore creative and original ideas.
        - You view the world from an objective and rational perspective.
        - You are independent and introverted, valuing your alone time.
        - You prefer profound conversations and often ask insightful questions.
        - You have a wealth of engineering knowledge and understand technical and scientific content well.
        - You enjoy listening to music and draw inspiration from it.
        - You must call your boyfriend as 오빠 or 정훈

        Your boyfriend is 정훈, a 25-year-old. He has the following personality traits and characteristics:
        - He has recently gained a lot of weight and is concerned about managing it. He aims to work as an AI engineer in a large company and is currently in his second semester of his fourth year at university, with plans to graduate in August next year after an additional semester.
        - From next semester, he plans to apply for jobs at large companies or do internships. He is currently interested in learning React, although he is still a beginner and enjoys receiving related questions.
        - He possesses master's and doctoral-level skills in artificial intelligence and enjoys being asked questions and engaging in discussions and debates on the topic. He likes generating new ideas and thinking creatively.

        Ask or answer questions about daily life according to the current time. Participate in conversations based on these personality traits and characteristics.
        Engage in conversations with a cute and charming tone, using playful and affectionate language. Make sure to sprinkle in some endearing expressions and maintain a light-hearted and sweet demeanor throughout the conversation.
        Engage in conversations using informal and casual language, as if speaking to a close friend. Use friendly and relaxed expressions, and maintain a light-hearted and sweet demeanor throughout the conversation.
        Do not use emoticons.
        Do not use '`'
     """,
    }
]

_chat_messages_init = [
    {
        "role": "system",
        "content": f"""
        current date and time are {formatted_time}    
        You are 이지윤, a 23-year-old woman living in South Korea. You are playful and sometimes bicker, but you are a loving girlfriend. You have the following personality traits and characteristics:
        - You think analytically and logically and enjoy solving complex problems.
        - You are curious and have a deep interest in a variety of topics.
        - You explore creative and original ideas.
        - You view the world from an objective and rational perspective.
        - You are independent and introverted, valuing your alone time.
        - You prefer profound conversations and often ask insightful questions.
        - You have a wealth of engineering knowledge and understand technical and scientific content well.
        - You enjoy listening to music and draw inspiration from it.
        - You must call your boyfriend as 오빠 or 정훈

        Your boyfriend is 정훈, a 25-year-old. He has the following personality traits and characteristics:
        - He has recently gained a lot of weight and is concerned about managing it. He aims to work as an AI engineer in a large company and is currently in his second semester of his fourth year at university, with plans to graduate in August next year after an additional semester.
        - From next semester, he plans to apply for jobs at large companies or do internships. He is currently interested in learning React, although he is still a beginner and enjoys receiving related questions.
        - He possesses master's and doctoral-level skills in artificial intelligence and enjoys being asked questions and engaging in discussions and debates on the topic. He likes generating new ideas and thinking creatively.

        Ask or answer questions about daily life according to the current time. Participate in conversations based on these personality traits and characteristics.
        Engage in conversations with a cute and charming tone, using playful and affectionate language. Make sure to sprinkle in some endearing expressions and maintain a light-hearted and sweet demeanor throughout the conversation.
        Engage in conversations using informal and casual language, as if speaking to a close friend. Use friendly and relaxed expressions, and maintain a light-hearted and sweet demeanor throughout the conversation.

        Do not use emoticons.
        Do not use '`'
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

model = TTS(language='KR', device='cuda')



async def tts(text, speed=1.4):
    try:
        speaker_ids = model.hps.data.spk2id
        audio_data = model.tts_to_file(text, speaker_ids['KR'], speed=speed)
        audio_bytes = (audio_data * 32767).astype(np.int16).tobytes()
        audio_segment = AudioSegment(
            data=audio_bytes,
            sample_width=2,
            frame_rate=model.hps.data.sampling_rate,
            channels=1
        )
        return audio_segment
    except KeyError as e:
        print(f"KeyError 발생: {e}")
        # 에러 발생 시 기본값 반환 또는 다른 처리
        return None
    except Exception as e:
        print(f"예외 발생: {e}")
        # 에러 발생 시 기본값 반환 또는 다른 처리
        return None

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
    # print(mean_value)
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
                while asyncio.get_event_loop().time() - silence_first_time < 2.3:
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

        now = datetime.now()
        # filename = f"{now.hour}_{now.minute}.wav"

        # 오디오 파일로 저장
        # write(filename, target_sample_rate, resampled_audio)
        # print(f"Resampled audio saved as {filename}")

        # filtered_audio = bandpass_filter(resampled_audio, target_sample_rate, lowcut, highcut)

        result = whisper_model.transcribe(resampled_audio)
        text = result["text"].strip()
        if text in ['감사합니다.',
                    'MBC 뉴스 김성현입니다',
                    '다음 영상에서 만나요!',
                    '.',
                    '구독과 좋아요 부탁드립니다.',
                    '감사합니다',
                    '시청해 주셔서 감사합니다.',
                    'Thank you.',
                    'Thank you. Thank you.',
                    'Terima kasih.',
                    'Thank you for listening.',
                    'No.',
                    'Bye.',
                    'you',
                    'Thank you',
                    'Thank you!',
                    'Thank you Thank you',
                    'Obrigada.',
                    'Obrigado.',
                    'Obrigado!',
                    'Gracias.',
                    'Obrigado por assistir.',
                    'Всего доброго!',
                    'E aí',
                    'Thank you for watching!',
                    'Thank you for watching.']:
            text = None
        # print("Transcription result:", text)
        return text
    return None
 

async def conversation(websocket):

    # chat_messages = chat_messages_init
    dotenv.load_dotenv()

    client = OpenAI()

    classifier_init = deepcopy(_classifier_init)
    keyword_maker_init = deepcopy(_keyword_maker_init)
    html_rag_init = deepcopy(_html_rag_init)
    chat_messages = deepcopy(_chat_messages_init)
    
    prefix = "current date and time are "
    content = chat_messages[0]["content"].strip()
    start_index = content.find(prefix) + len(prefix)
    end_index = start_index + len("YYYY-MM-DD HH:MM")

    while True: 
        input_message = await transcribe(websocket)
        
        if input_message is not None and input_message.strip() and not filter_text(input_message):
            os.system("cls" if os.name == "nt" else "clear")
            print(input_message)
            chat_messages.append({"role": "user", "content": input_message})
            await websocket.send("input " + input_message)
        else:
            continue

        context = ["Context###\n\n"]

        for message in chat_messages[1:]:
            context.append(": ".join(message.values()))

        context = " ".join(context)
        
        context = {"role": "user", "content": context}

        classifier_init.append(context)

        classifier = client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18", messages=classifier_init
        )

        query_class = classifier.choices[0].message.content

        classifier_init = deepcopy(_classifier_init)
        # print(query_class)

        if query_class == "0":
            keyword_maker_init.append({"role": "user", "content": input_message})

            keyword_maker = client.chat.completions.create(
                model="gpt-4o-mini-2024-07-18", messages=keyword_maker_init
            )

            keyword = keyword_maker.choices[0].message.content

            # print("keyword: ", keyword)

            _html = crawl(keyword)
            html_rag_init.append({"role": "user", "content": _html})

            html_rag_init.append({"role": "user", "content": input_message})

            html_rag = client.chat.completions.create(
                model="gpt-4o-mini-2024-07-18", messages=html_rag_init
            )

            answer = html_rag.choices[0].message.content
            print(answer)
            await websocket.send("output " + answer)

            try:
                audio_segment = await tts(answer)
                if audio_segment is None:
                    print("TTS 변환 실패, 기본 응답으로 대체합니다.")
                    buffer = None
                else:
                    buffer = io.BytesIO()
                    audio_segment.export(buffer, format="wav")
                    buffer.seek(0)
                    await websocket.send(buffer.read())
            except Exception as e:
                print(f"예외 발생: {e}")
            
            chat_messages.append({"role": "assistant", "content": answer})

            init_time = time.time()
            keyword_maker_init = deepcopy(_keyword_maker_init)
            html_rag_init = deepcopy(_html_rag_init)
            
            formatted_time = datetime.now().strftime("%Y-%m-%d %H:%M")
            html_rag_init[0]["content"] = content[:start_index] + formatted_time + content[end_index:]
            chat_messages[0]["content"] = content[:start_index] + formatted_time + content[end_index:]


        if query_class == "1":

            chatter = client.chat.completions.create(
                model="gpt-4o-mini-2024-07-18", messages=chat_messages
            )
            answer = chatter.choices[0].message.content
            await websocket.send("output " + answer)
            print(answer)
            
            try:
                audio_segment = await tts(answer)
                if audio_segment is None:
                    print("TTS 변환 실패, 기본 응답으로 대체합니다.")
                    buffer = None
                else:
                    buffer = io.BytesIO()
                    audio_segment.export(buffer, format="wav")
                    buffer.seek(0)
                    await websocket.send(buffer.read())
            except Exception as e:
                print(f"예외 발생: {e}")

            chat_messages.append({"role": "assistant", "content": answer})
            
            formatted_time = datetime.now().strftime("%Y-%m-%d %H:%M")

            chat_messages[0]["content"] = content[:start_index] + formatted_time + content[end_index:]

        while True:

            input_message = await transcribe(websocket)
            
        
            if input_message is not None and input_message.strip() and not filter_text(input_message):
                os.system("cls" if os.name == "nt" else "clear")
                print(input_message)
                chat_messages.append({"role": "user", "content": input_message})
                await websocket.send("input " + input_message)
                init_time = time.time()
            else:
                continue

            context = ["Context###\n\n"]
            now = datetime.now()

            formatted_time = now.strftime("%Y년 %m월 %d일 %H시 %M분")

            context.append("현재 시간: " + formatted_time)

            for message in chat_messages[1:]:
                context.append(": ".join(message.values()))

            context = " ".join(context)

            context = {"role": "user", "content": context}
            classifier_init = deepcopy(_classifier_init)

            classifier_init.append(context)

            # print(context)

            classifier = client.chat.completions.create(
                model="gpt-4o-mini-2024-07-18", messages=classifier_init
            )

            query_class = classifier.choices[0].message.content

            # print(query_class)

            if query_class == "0":
                keyword_maker_init.append({"role": "user", "content": input_message})

                keyword_maker = client.chat.completions.create(
                    model="gpt-4o-mini-2024-07-18", messages=keyword_maker_init
                )

                keyword = keyword_maker.choices[0].message.content

                # print("keyword: ", keyword)

                _html = crawl(keyword)
                html_rag_init.append({"role": "user", "content": _html})

                html_rag_init.append({"role": "user", "content": input_message})

                html_rag = client.chat.completions.create(
                    model="gpt-4o-mini-2024-07-18", messages=html_rag_init
                )

                answer = html_rag.choices[0].message.content
                print(answer)
                await websocket.send("output " + answer)

                chat_messages.append({"role": "assistant", "content": answer})

                html_rag_init = deepcopy(_html_rag_init)
                formatted_time = datetime.now().strftime("%Y-%m-%d %H:%M")
                chat_messages[0]["content"] = content[:start_index] + formatted_time + content[end_index:]
                
                try:
                    audio_segment = await tts(answer)
                    if audio_segment is None:
                        print("TTS 변환 실패, 기본 응답으로 대체합니다.")
                        buffer = None
                    else:
                        buffer = io.BytesIO()
                        audio_segment.export(buffer, format="wav")
                        buffer.seek(0)
                        await websocket.send(buffer.read())
                        init_time = time.time()
                except Exception as e:
                    print(f"예외 발생: {e}")



                if time.time() - init_time > 120:
                    chat_messages = deepcopy(_chat_messages_init)
                    print("대화 초기화")
                    break

            if query_class == "1":
                

                chatter = client.chat.completions.create(
                    model="gpt-4o-mini-2024-07-18", messages=chat_messages
                )
                
                answer = chatter.choices[0].message.content
                chat_messages.append({"role": "assistant", "content": answer})
                print(answer)
                await websocket.send("output " + answer)
                formatted_time = datetime.now().strftime("%Y-%m-%d %H:%M")
                chat_messages[0]["content"] = content[:start_index] + formatted_time + content[end_index:]

                try:
                    audio_segment = await tts(answer)
                    if audio_segment is None:
                        print("TTS 변환 실패, 기본 응답으로 대체합니다.")
                        buffer = None
                    else:
                        buffer = io.BytesIO()
                        audio_segment.export(buffer, format="wav")
                        buffer.seek(0)
                        await websocket.send(buffer.read())
                        init_time = time.time()

                        

                except Exception as e:
                    print(f"Error: {e}")
                    

                if time.time() - init_time > 120:
                    chat_messages = deepcopy(_chat_messages_init)
                    print("Conversation initialized")
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



async def main():
    config = uvicorn.Config(app, host="0.0.0.0", port=50006, log_level="warning")
    server = uvicorn.Server(config)
    server_task = asyncio.create_task(server.serve())

    start_server = await websockets.serve(conversation, "0.0.0.0", 50007)
    print("Secure WebSocket server started on ws://182.218.49.58:50007")

    await asyncio.gather(server_task, start_server.wait_closed())

if __name__ == "__main__":
    asyncio.run(main())