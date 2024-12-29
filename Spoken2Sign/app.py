from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from service import Spoken2SignService
import requests

from enum import Enum
import time

app = FastAPI()

spoken2signservice = Spoken2SignService()

status_map = {}

class TaskStatus(str, Enum):
    in_queue = 'in_queue'
    running_t2g = 'running_t2g'
    running_s2s = 'running_s2s'
    rendering = 'rendering'
    completed = 'completed'
    error = 'error'


def process_request(task_id: int, text: str):
    status_map[task_id] = TaskStatus.running_t2g
    response = requests.post('http://localhost:8081/t2g', json={'text': text})
    if response.status_code != 200:
        print('[ERROR] failed to perform request to T2G backend')
        status_map[task_id] = TaskStatus.error
        return
    result = response.json()
    gloss = result['gloss']
    print(f'[DEBUG] T2G: {text} -> {gloss}')
    status_map[task_id] = TaskStatus.running_s2s
    spoken2signservice.generate(task_id, gloss)
    status_map[task_id] = TaskStatus.completed


class Spoken2SignRequest(BaseModel):
    text: str

class Spoken2SignTask(BaseModel):
    id: int
    status: TaskStatus

@app.post("/spoken2sign")
def spoken2sign(request: Spoken2SignRequest, background_tasks: BackgroundTasks) -> Spoken2SignTask:
    task_id = int(time.time())
    status_map[task_id] = TaskStatus.in_queue
    background_tasks.add_task(process_request, task_id, request.text)
    return Spoken2SignTask(id=task_id, status=TaskStatus.in_queue)

@app.get("/spoken2sign/{task_id}/status")
def get_status(task_id: int) -> Spoken2SignTask:
    if task_id not in status_map:
        raise HTTPException(status_code=404, detail="Task not found")
    return Spoken2SignTask(id=task_id, status=status_map[task_id])
