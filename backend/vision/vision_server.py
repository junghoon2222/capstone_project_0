import asyncio
import websockets
import cv2
import numpy as np
from ultralytics import YOLO
async def image_handler(websocket, path):
    print("클라이언트 연결됨")

    try:
        while True:
            # 클라이언트로부터 바이너리 데이터 수신
            data = await websocket.recv()

            # 바이너리 데이터를 NumPy 배열로 변환
            np_arr = np.frombuffer(data, dtype=np.uint8)

            # 배열을 이미지로 디코딩
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if img is not None:
                # 이미지를 화면에 표시
                results = model(img)
                result = results[0].plot()
                cv2.imshow('Received Image', result)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                print("빈 프레임 수신")

    except websockets.exceptions.ConnectionClosed:
        print("클라이언트 연결 종료")
    finally:
        cv2.destroyAllWindows()


async def main():
    # 웹소켓 서버 실행
    
    async with websockets.serve(image_handler, "0.0.0.0", 8080):
        print("웹소켓 서버가 8080 포트에서 실행 중입니다.")
        await asyncio.Future()  # 서버가 종료되지 않도록 대기

# 비동기 루프 실행
if __name__ == "__main__":
    model = YOLO("yolov8l-face.engine", task="obb")

    asyncio.run(main())
