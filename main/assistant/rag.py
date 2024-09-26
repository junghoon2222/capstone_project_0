from sentence_transformers import SentenceTransformer, util
from kiwipiepy import Kiwi
import numpy as np

# Sentence-Transformers 모델 로드
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')  # 다국어 지원 모델

# Kiwi를 사용하여 텍스트 문장 분리
kiwi = Kiwi()

def recursive_chunk(text, max_length=20):
    """
    텍스트를 재귀적으로 분리하여 max_length 이하의 청크로 나눔.
    
    :param text: 분리할 원본 텍스트
    :param max_length: 각 청크의 최대 길이 (문자 길이 기준)
    :return: 분리된 청크 리스트
    """
    # 텍스트가 너무 길면 문장 단위로 나눔
    sentences = kiwi.split_into_sents(text)
    sentences_text = [sentence.text for sentence in sentences]
    
    # 만약 전체 텍스트 길이가 max_length 이하면 그대로 반환
    if sum(len(sentence) for sentence in sentences_text) <= max_length:
        return [text]
    
    # 그렇지 않으면 재귀적으로 분할
    chunks = []
    current_chunk = ""
    
    for sentence in sentences_text:
        # 현재 청크에 문장을 추가했을 때 길이를 확인
        if len(current_chunk) + len(sentence) <= max_length:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk.strip())  # 현재 청크를 저장
            current_chunk = sentence  # 새로운 청크 시작
    
    # 마지막 청크도 저장
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    # 각 청크가 여전히 길면 재귀적으로 더 나눔
    final_chunks = []
    for chunk in chunks:
        if len(chunk) > max_length:
            final_chunks.extend(recursive_chunk(chunk, max_length))  # 재귀적으로 더 나눔
        else:
            final_chunks.append(chunk)
    
    return final_chunks

def find_relevant_chunks(input_text, text, threshold=0.7, max_length=300):
    """
    입력 문장과 텍스트의 청크들 간 유사도를 계산하여 임계치 이상인 청크만 반환.
    
    :param input_text: 입력 문장
    :param text: 전체 텍스트
    :param threshold: 유사도 임계치
    :param max_length: 청크의 최대 길이
    :return: 유사도가 임계치를 넘는 청크 리스트
    """
    # 텍스트를 재귀적으로 청크로 분리
    chunks = recursive_chunk(text, max_length)

    # 입력 문장과 청크들에 대한 임베딩 생성
    input_embedding = model.encode(input_text, convert_to_tensor=True)
    chunk_embeddings = model.encode(chunks, convert_to_tensor=True)

    # 코사인 유사도 계산
    similarities = util.cos_sim(input_embedding, chunk_embeddings)

    # 유사도가 임계치 이상인 청크 선택
    relevant_chunks = [chunks[i] for i, similarity in enumerate(similarities[0]) if similarity >= threshold]
    relevant_similarities = [similarity.item() for similarity in similarities[0] if similarity >= threshold]

    return relevant_chunks, relevant_similarities

# 테스트할 텍스트
with open('final_extracted_text.txt', 'r', encoding='utf-8') as file:
    text = file.read()
# 입력 문장
input_sentence = "kbo 결과"

# 유사도 임계치를 넘는 청크 찾기
relevant_chunks, relevant_similarities = find_relevant_chunks(input_sentence, text, threshold=0.7)

# 결과 출력
for i, (chunk, similarity) in enumerate(zip(relevant_chunks, relevant_similarities)):
    print(f"Relevant Chunk {i+1}: {chunk} (Similarity: {similarity:.2f})")
