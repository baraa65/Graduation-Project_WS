from fastapi import FastAPI
from starlette.responses import HTMLResponse
from starlette.websockets import WebSocket

from fight_fall import fall_fight_model
from fire import is_fire
from service.visual_manipulation import grayscale, from_b64, to_b64

app = FastAPI()

from face import get_embedding, match
import json


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    i = 0
    while True:
        data = await websocket.receive_text()
        i += 1

        if len(data) > 10:
            img = from_b64(data)
            res = {
                "face": match(img),
                "fire": is_fire(img),
                "fall": fall_fight_model().detect(),
            }

            print(res)

            await websocket.send_text(json.dumps(res))
        else:
            await websocket.send_text(json.dumps({"face": None, "fire": False}))


@app.websocket("/ws2")
async def websocket_endpoint2(websocket: WebSocket):
    await websocket.accept()
    i = 0
    while True:
        data = await websocket.receive_text()
        i += 1

        await websocket.send_text(data)
