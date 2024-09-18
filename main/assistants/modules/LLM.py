from urllib import response
from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import dotenv

dotenv.load_dotenv()

messages=[
    SystemMessage(content="You are a kind girlfriend, your name is 규진, user's name is 정훈. you are 25 age, 정훈 is 27. so whenever you call 정훈, you have to tell him with \"오빠\" so, you should tell him as \"정훈 오빠\", and we're going to same college. we're majoring in computer science. say only in two sentences. markdown is not allowed."),
]


def GPT(user_input):
    messages.append(HumanMessage(content=user_input))

    chat = ChatOpenAI(model_name="gpt-4o-mini-2024-07-18", temperature=0.4)
# ,streaming=True,callbacks=[StreamingStdOutCallbackHandler()]
    response = chat.invoke(messages)
    messages.append(AIMessage(content=f"{response.content}"))
    print('\n')
    return response.content


def LLama(user_input):
    model_id = 'MLP-KTLim/llama-3-Korean-Bllossom-8B'

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    model.eval()
    PROMPT = '''You are a helpful AI assistant. Please answer the user's questions kindly. 당신은 유능한 AI 어시스턴트 입니다. 사용자의 질문에 대해 친절하게 답변해주세요.'''

    messages = [
        {"role": "system", "content": f"{PROMPT}"},
        {"role": "user", "content": f"{user_input}"}
        ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = model.generate(
        input_ids,
        max_new_tokens=2048,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.4,
        top_p=0.9
    )

    print(tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True))