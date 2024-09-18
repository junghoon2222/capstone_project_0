import asyncio
import websockets
from queue import Queue

result_queue = Queue()
websocket_clients = set()

async def websocket_handler(websocket, path):
    global websocket_clients
    try:
        websocket_clients.add(websocket)
        print(f"New WebSocket connection. Total connections: {len(websocket_clients)}")
        while True:
            if not result_queue.empty():
                message = result_queue.get()
                await websocket.send(message)
                print(f"Sent message to WebSocket: {message}")
            await asyncio.sleep(0.1)
    except websockets.exceptions.ConnectionClosed:
        print("WebSocket connection closed")
    finally:
        websocket_clients.remove(websocket)
        print(f"WebSocket connection removed. Total connections: {len(websocket_clients)}")

async def result_sender():
    while True:
        if not result_queue.empty() and websocket_clients:
            message = result_queue.get()
            print(f"Sending message to all clients: {message}")
            await asyncio.gather(*[client.send(message) for client in websocket_clients])
        await asyncio.sleep(0.1)

async def main():
    # 웹소켓 서버 시작
    server = await websockets.serve(websocket_handler, "localhost", 8765)
    
    # 결과 전송 태스크 시작
    sender_task = asyncio.create_task(result_sender())

    print("WebSocket server started on ws://localhost:8765")
    
    await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())
