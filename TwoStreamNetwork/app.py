from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List
from typing_extensions import Annotated
import ffmpeg
import hashlib
import tempfile
import os
import redis

from service import FeatureExtractionService, PredictionService

app = FastAPI()
app.mount('/static', StaticFiles(directory="videos"), name="static")


class Sign2TextResult(BaseModel):
    id: str
    task_type: str
    model: str
    text: str
    gloss: str
    video_url: str


class LanguageModel(BaseModel):
    id: str
    name: str
    language: str


available_languages = [
    LanguageModel(id="phoenix-2014t",
                  name="RWTH-PHOENIX-Weather 2014 T", language="German Sign Language 德國手語"),
    LanguageModel(id="csl-daily", name="Chinese Sign Language Corpus",
                  language="Chinese Sign Language 中国手语"),
    # LanguageModel(id="tvb", name="TVB-HKSL-News",
    #               language="Hong Kong Sign Language 香港手語"),
]

feature_extraction_service = {language.id: FeatureExtractionService(
    f"experiments/configs/SingleStream/{language.id}_s2g.yaml") for language in available_languages}
prediction_service = {language.id: PredictionService(
    f"experiments/configs/SingleStream/{language.id}_s2t.yaml") for language in available_languages}

static_dir = "./videos"
BLOCK_SIZE = 1024 * 1024

valkey_client = redis.Redis(host='localhost', port=6379, db=0)


@app.post("/")
async def process_video(
    model: Annotated[str, Form()],
    video_file: Annotated[UploadFile, File()],
) -> Sign2TextResult:

    if model not in [language.id for language in available_languages]:
        raise HTTPException(
            status_code=400,
            detail="Model not found"
        )

    file_extension = video_file.filename.split(".")[-1]

    # Create temporary working directory
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Generate unique filenames
            input_path = os.path.join(temp_dir, f"input.{file_extension}")
            output_dir = os.path.join(temp_dir, "frames")

            # Generate task ID
            sha1 = hashlib.sha1()

            # Save uploaded file to temporary directory
            with open(input_path, "wb") as f:
                while True:
                    data = await video_file.read(BLOCK_SIZE)
                    if not data:
                        break
                    f.write(data)
                    sha1.update(data)

            task_id = sha1.hexdigest()

            # Create frames directory
            os.makedirs(output_dir, exist_ok=True)
            output_pattern = os.path.join(output_dir, "frame_%04d.png")

            # Process video with ffmpeg
            (
                ffmpeg
                .input(input_path)
                .output(
                    output_pattern,
                    t=5,
                    vcodec="png",
                    format="image2"
                )
                # Suppress ffmpeg output
                .run(quiet=True, overwrite_output=True)
            )

            # copy video file to static serve directory
            (
                ffmpeg
                .input(input_path)
                .output(
                    os.path.join(
                        static_dir, f"{task_id}.mp4")
                )
                # Suppress ffmpeg output
                .run(quiet=True, overwrite_output=True)
            )

        except ffmpeg.Error as e:
            raise HTTPException(
                status_code=500,
                detail=f"FFmpeg processing failed: {e.stderr.decode()}"
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Processing error: {str(e)}"
            )

        features = feature_extraction_service[model].extract_feature(
            output_dir)
        text, gloss = prediction_service[model].predict(output_dir, features)
        valkey_client.lpush("task:list", task_id)
        valkey_client.set(f'task:{task_id}:text', text)
        valkey_client.set(f'task:{task_id}:gloss', gloss)
        valkey_client.set(f'task:{task_id}:task_type', "S2T")
        valkey_client.set(f'task:{task_id}:model', model)
        return Sign2TextResult(id=task_id, task_type="S2T", model=model, gloss=gloss, text=text, video_url=f"/static/{task_id}.mp4")


@app.get("/languages")
def list_languages() -> List[LanguageModel]:
    return available_languages


@app.get("/{task_id}")
def get_result(task_id: str) -> Sign2TextResult:
    text = valkey_client.get(f'task:{task_id}:text')
    gloss = valkey_client.get(f'task:{task_id}:gloss')
    task_type = valkey_client.get(f'task:{task_id}:task_type')
    model = valkey_client.get(f'task:{task_id}:model')
    if text is None or gloss is None:
        raise HTTPException(
            status_code=404,
            detail="Task not found"
        )
    return Sign2TextResult(
        id=task_id,
        task_type=task_type,
        model=model,
        text=text,
        gloss=gloss,
        video_url=f"/static/{task_id}.mp4",
    )


@app.get("/")
def list_result() -> List[Sign2TextResult]:
    task_list = valkey_client.lrange('task:list', 0, -1)
    return [get_result(task_id.decode()) for task_id in task_list]
