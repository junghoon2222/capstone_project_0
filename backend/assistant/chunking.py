
import torch
from transformers import BertModel, BertTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk

# KoBERT 모델과 토크나이저 로드
tokenizer = BertTokenizer.from_pretrained('monologg/kobert')
model = BertModel.from_pretrained('monologg/kobert')

# 입력 텍스트: 한 줄로 제공된 텍스트
text = "[KBO 내일의 선발투수] 9월 27일 (금) 대전(오후 6시 30분) KIA (황동하) - (라이언 와이스) 한화 롯데는 한현희가 오프너 또는 선발로 나설 예정이다. 8월 2일 한화 원정에서 5이닝 3실점 패배를 기록했으며, KIA의 상대 상성은 불안 요소이다."

# 문장 단위로 분리하기
sentences = nltk.sent_tokenize(text)

# KoBERT 토크나이저로 문장을 토큰화하고 모델에 입력
def get_sentence_embedding(sentence):
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    # CLS 토큰에 대한 임베딩을 사용하고 2차원으로 변환
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()

# 문장 임베딩 계산
sentence_embeddings = [get_sentence_embedding(sentence) for sentence in sentences]

# 임베딩 간 유사도 계산 (코사인 유사도)
similarity_matrix = cosine_similarity(sentence_embeddings)

# 유사도 기준으로 문장들을 그룹화하여 청킹
threshold = 0.7  # 유사도 임계값 설정 (0.7 이상인 문장끼리 묶음)
chunks = []
current_chunk = [sentences[0]]

for i in range(1, len(sentences)):
    similarity = cosine_similarity([sentence_embeddings[i]], [sentence_embeddings[i-1]])[0][0]
    if similarity > threshold:
        current_chunk.append(sentences[i])
    else:
        chunks.append(current_chunk)
        current_chunk = [sentences[i]]

# 마지막 청크 추가
if current_chunk:
    chunks.append(current_chunk)

# 청킹 결과 출력
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}: {chunk}")
