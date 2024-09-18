import pyaudio
import numpy as np
import websockets
import asyncio
import os
import re
import time
from openai import OpenAI
import dotenv
from pre import *
from copy import deepcopy
from datetime import datetime


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


async def whisp_transcription():
    p = pyaudio.PyAudio()
    threshold = 200
    buffer_size = 4000
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=16000,
        input=True,
        frames_per_buffer=buffer_size,
    )

    async with websockets.connect("ws://182.218.49.58:50007") as websocket:
        frame = stream.read(buffer_size, exception_on_overflow=False)
        frame_mean = np.mean(np.abs(np.frombuffer(frame, dtype=np.int16)))
        print(frame_mean)

        if frame_mean > threshold:
            data = bytearray(frame)

            while True:
                frame = stream.read(buffer_size, exception_on_overflow=False)
                frame_mean = np.mean(np.abs(np.frombuffer(frame, dtype=np.int16)))
                print(frame_mean)
                data.extend(frame)

                if frame_mean < threshold:
                    silence_first_time = time.time()
                    while time.time() - silence_first_time < 1:

                        frame = stream.read(buffer_size, exception_on_overflow=False)
                        frame_mean = np.mean(
                            np.abs(np.frombuffer(frame, dtype=np.int16))
                        )

                        data.extend(frame)
                        if frame_mean > threshold:
                            silence_first_time = time.time()
                    break
            await websocket.send(data)

            response = await websocket.recv()
            if not filter_text(response):

                os.system("cls" if os.name == "nt" else "clear")
                print(response)
                return response


async def record_and_send_audio(websocket_2):

    # chat_messages = chat_messages_init
    dotenv.load_dotenv()

    client = OpenAI()

    classifier_init = deepcopy(_classifier_init)
    keyword_maker_init = deepcopy(_keyword_maker_init)
    html_rag_init = deepcopy(_html_rag_init)
    chat_messages = deepcopy(_chat_messages_init)

    while True:

        input_message = await whisp_transcription()
        if input_message and input_message.strip():

            chat_messages.append({"role": "user", "content": input_message})
            await websocket_2.send("input " + input_message)

        else:
            continue

        context = ["컨텍스트###\n\n"]

        now = datetime.now()

        formatted_time = now.strftime("%Y-%m-%d %H:%M")

        context.append("현재시간 :" + formatted_time)

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
            await websocket_2.send("output " + answer)

            chat_messages.append({"role": "assistant", "content": answer})

            init_time = time.time()
            keyword_maker_init = deepcopy(_keyword_maker_init)
            html_rag_init = deepcopy(_html_rag_init)

        if query_class == "1":

            chatter = client.chat.completions.create(
                model="gpt-4o-mini-2024-07-18", messages=chat_messages
            )
            answer = chatter.choices[0].message.content
            await websocket_2.send("output" + answer)
            print(answer)

            chat_messages.append({"role": "assistant", "content": answer})

            init_time = time.time()

        while True:
            input_message = await whisp_transcription()

            if input_message and input_message.strip():
                chat_messages.append({"role": "user", "content": input_message})
                await websocket_2.send("input " + input_message)

            else:
                continue

            context = ["컨텍스트###\n\n"]
            now = datetime.now()

            formatted_time = now.strftime("%Y-%m-%d %H:%M")

            context.append("현재시간 :" + formatted_time)

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
                await websocket_2.send("output " + answer)
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
                await websocket_2.send("output " + answer)
                if time.time() - init_time > 180:
                    chat_messages = deepcopy(_chat_messages_init)
                    print("3 minute has flew")
                    break


if __name__ == "__main__":
    # asyncio.run(record_and_send_audio(websocket))
    start_server = websockets.serve(record_and_send_audio, "0.0.0.0", 50008)
    asyncio.get_event_loop().run_until_complete(start_server)
    print("WebSocket server started on ws://localhost:50008")
    asyncio.get_event_loop().run_forever()
