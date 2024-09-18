import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1
from ultralytics import YOLO
import tkinter as tk
from PIL import Image, ImageTk
import time
import pickle
import os

# CUDA 장치 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 얼굴 검출 모델 로드 (YOLOv8n-face)
face_detector = YOLO('yolov8l-face.pt').to(device)

# 얼굴 인식 모델 로드 (FaceNet)
facenet_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# 얼굴 특징 벡터 저장 딕셔너리
face_embeddings = {}

# 얼굴 특징 벡터 저장 파일 경로
embeddings_file = 'face_embeddings.pkl'

# 얼굴 인코딩 저장 함수
def save_embeddings(embeddings, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(embeddings, f)

# 얼굴 인코딩 로드 함수
def load_embeddings(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    return {}

# 얼굴 등록 함수
def register_face(name, embedding):
    face_embeddings[name] = embedding.cpu()  # GPU 텐서를 CPU 텐서로 변환하여 저장
    save_embeddings(face_embeddings, embeddings_file)

# 얼굴 인식 함수
def recognize_face(embedding):
    min_distance = float('inf')
    recognized_name = 'Unknown'

    for name, registered_embedding in face_embeddings.items():
        distance = torch.dist(embedding, registered_embedding.to(device))  # CPU 텐서를 GPU 텐서로 변환하여 비교
        if distance < min_distance:
            min_distance = distance
            recognized_name = name

    return recognized_name

# 얼굴 등록 버튼 클릭 이벤트 처리 함수
def register_face_button_click():
    global new_face_embedding
    new_name = name_entry.get()
    register_face(new_name, new_face_embedding)
    register_button.config(state=tk.DISABLED)
    name_entry.delete(0, tk.END)

# 창 닫기 이벤트 처리 함수
def on_close():
    global cap
    cap.release()
    window.destroy()

# GUI 설정
window = tk.Tk()
window.title("Face Recognition")

# 카메라 프레임 라벨
camera_label = tk.Label(window)
camera_label.pack()

# 이름 입력 필드
name_entry = tk.Entry(window)
name_entry.pack()

# 얼굴 등록 버튼
register_button = tk.Button(window, text="Register Face", command=register_face_button_click, state=tk.DISABLED)
register_button.pack()

# 창 닫기 이벤트 바인딩
window.protocol("WM_DELETE_WINDOW", on_close)

# 웹캠 열기
cap = cv2.VideoCapture(0)

# FPS 설정
fps = 60
frame_interval = 1 / fps

# 얼굴 인코딩 로드
face_embeddings = load_embeddings(embeddings_file)

while True:
    start_time = time.time()
    
    ret, frame = cap.read()

    if not ret:
        break

    # 얼굴 검출
    results = face_detector(frame)

    # 등록되지 않은 인물 플래그 초기화
    unknown_face_detected = False

    for result in results:
        if len(result.boxes.xyxy) > 0:  # 감지된 얼굴이 있는지 확인
            # 얼굴 영역 추출
            x1, y1, x2, y2 = map(int, result.boxes.xyxy[0])
            face_img = frame[y1:y2, x1:x2]

            try:
                # 얼굴 인식
                face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                face_img_tensor = torch.from_numpy(face_img_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                face_img_tensor = face_img_tensor.to(device)  # GPU로 이동
                face_embedding = facenet_model(face_img_tensor).detach()

                recognized_name = recognize_face(face_embedding)

                # 등록되지 않은 인물인 경우 플래그 설정
                if recognized_name == 'Unknown':
                    unknown_face_detected = True
                    new_face_embedding = face_embedding

                # 결과 출력
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, recognized_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            except RuntimeError:
                # 얼굴 인식 과정에서 오류 발생 시 해당 얼굴은 무시하고 다음 얼굴로 넘어감
                continue

    # 등록되지 않은 인물이 감지된 경우 얼굴 등록 버튼 활성화
    if unknown_face_detected:
        register_button.config(state=tk.NORMAL)
    else:
        register_button.config(state=tk.DISABLED)

    # 프레임을 PIL 이미지로 변환
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_pil = Image.fromarray(frame_rgb)
    frame_tk = ImageTk.PhotoImage(frame_pil)

    # 카메라 프레임 업데이트
    camera_label.config(image=frame_tk)
    camera_label.image = frame_tk

    # GUI 이벤트 처리
    window.update()
    
    # Frame interval adjustment to match FPS
    elapsed_time = time.time() - start_time
    time_to_wait = frame_interval - elapsed_time
    if time_to_wait > 0:
        time.sleep(time_to_wait)

cap.release()
cv2.destroyAllWindows()
