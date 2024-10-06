import asyncio
import websockets
import cv2
import numpy as np
from ultralytics import YOLO

async def sign_detection(K):
    result = []
    try:
        for k in K:
            class_num = 0
        if all(right[1] < left[1] for left in k[0:4] for right in k[4:6]):
            if all(right[1] > left[1] for left in k[2:4] for right in k[4:6]):
                if k[5][0] < k[4][0]:
                    class_num = 2 # 큰 X

            else:
                if k[5][0] < k[4][0]:
                    class_num = 3 # 큰 O

        else:
            if all(right[1] > left[1] for left in k[4:6] for right in k[2:4]):
                if k[5][0] > k[4][0]:
                    class_num = 1 # 작은 X
            result.append(class_num)
    except Exception as e:
        pass
    return result

async def image_handler(websocket, path):
    print("클라이언트 연결됨")

    try:
        while True:
            data = await websocket.recv()

            np_arr = np.frombuffer(data, dtype=np.uint8)

            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if img is not None:
                # 이미지를 화면에 표시
                results = model(img)
                result = results[0].plot()

                cv2.imshow('Received Image', result)
                sign = await sign_detection(result)
                print(sign)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            else:
                print("빈 프레임 수신")

    except websockets.exceptions.ConnectionClosed:
        print("클라이언트 연결 종료")
    finally:
        cv2.destroyAllWindows()


async def main():
    async with websockets.serve(image_handler, "0.0.0.0", 50008):
        await asyncio.Future()

# 비동기 루프 실행
if __name__ == "__main__":
    model = YOLO("yolo11l-pose.pt", task="pose")

    asyncio.run(main())
