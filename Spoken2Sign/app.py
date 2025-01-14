from fastapi import FastAPI, HTTPException, BackgroundTasks, Security
from fastapi.security import APIKeyHeader
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from service import Spoken2SignService
import requests
import valkey
import json
import hashlib

from enum import Enum
import time

api_key_header = APIKeyHeader(name="X-API-Key")

app = FastAPI()
app.mount('/static', StaticFiles(directory="videos"), name="static")

spoken2signservice = Spoken2SignService()

valkey_client = valkey.Valkey(host='localhost', port=6379, db=0)

class TaskStatus(str, Enum):
    in_queue = 'in_queue'
    running_t2g = 'running_t2g'
    running_s2s = 'running_s2s'
    generated = 'generated'
    rendering = 'rendering'
    completed = 'completed'
    error = 'error'


def process_request(task_id: str, text: str):
    valkey_client.set(f'task:{task_id}:status', TaskStatus.running_t2g)
    response = requests.post('http://localhost:8081/t2g', json={'text': text})
    if response.status_code != 200:
        print('[ERROR] failed to perform request to T2G backend')
        valkey_client.set(f'task:{task_id}:status', TaskStatus.error)
        return
    result = response.json()
    gloss = result['gloss']
    valkey_client.set(f'task:{task_id}:gloss', gloss)
    print(f'[DEBUG] T2G task {task_id}: {text} -> {gloss}')
    valkey_client.set(f'task:{task_id}:status', TaskStatus.running_s2s)
    result = spoken2signservice.generate(task_id, gloss)
    valkey_client.set(f'task:{task_id}:gloss_frame_mapping', json.dumps(result.gloss_frame_mapping))
    print(f"[DEBUG] Generate task {task_id}: {result.gloss_frame_mapping}")
    valkey_client.set(f'task:{task_id}:status', TaskStatus.generated)
    valkey_client.lpush(f'render:jobs', task_id)


class Spoken2SignRequest(BaseModel):
    text: str

class Spoken2SignTask(BaseModel):
    id: str
    status: TaskStatus
    progress: float

class Spoken2SignResult(BaseModel):
    id: str
    text: str
    gloss: str
    video_url: str

@app.get("/healthz")
async def healthz() -> str:
    return "OK"

@app.post("/spoken2sign")
def spoken2sign(request: Spoken2SignRequest, background_tasks: BackgroundTasks) -> Spoken2SignTask:
    task_id = hashlib.sha1(request.text.encode()).hexdigest()
    valkey_client.lpush("task:list", task_id)
    status = valkey_client.get(f'task:{task_id}:status')
    if status is not None:
        return get_status(task_id)
    valkey_client.set(f'task:{task_id}:status', TaskStatus.in_queue)
    valkey_client.set(f'task:{task_id}:progress', 0.0)
    valkey_client.set(f'task:{task_id}:text', request.text)
    background_tasks.add_task(process_request, task_id, request.text)
    return Spoken2SignTask(id=task_id, status=TaskStatus.in_queue, progress=0.0)

@app.get("/spoken2sign/{task_id}/status")
def get_status(task_id: str) -> Spoken2SignTask:
    status = valkey_client.get(f'task:{task_id}:status')
    progress = float(valkey_client.get(f'task:{task_id}:progress').decode())
    if not status:
        raise HTTPException(status_code=404, detail="Task not found")
    return Spoken2SignTask(id=task_id, status=status, progress=progress)

@app.get("/spoken2sign/status")
def list_tasks() -> list[Spoken2SignTask]:
    task_list = valkey_client.lrange('task:list', 0, -1)
    return [get_status(task.decode()) for task in task_list]

@app.get("/spoken2sign/{task_id}/video")
def get_video(task_id: str):
    # replace with MinIO later
    return RedirectResponse(f"/static/custom-input-{task_id}.mp4")

@app.get("/spoken2sign/{task_id}")
def get_result(task_id: str) -> Spoken2SignResult:
    text = valkey_client.get(f'task:{task_id}:text')
    gloss = valkey_client.get(f'task:{task_id}:gloss')
    gloss_frame_mapping = json.loads(valkey_client.get(f'task:{task_id}:gloss_frame_mapping'))
    return Spoken2SignResult(
        id=task_id,
        text=text,
        gloss=gloss,
        video_url=f"/static/custom-input-{task_id}.mp4",
    )

@app.get("/spoken2sign")
def list_result() -> list[Spoken2SignResult]:
    status = list_tasks()
    return [get_result(task.id) for task in status if task.status == TaskStatus.completed]
