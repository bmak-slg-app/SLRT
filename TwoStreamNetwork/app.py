from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import ffmpeg
import hashlib
import tempfile
import os
import redis

from service import FeatureExtractionService, PredictionService

app = FastAPI()
app.mount('/static', StaticFiles(directory="videos"), name="static")

feature_extraction_service = FeatureExtractionService(
    "experiments/configs/SingleStream/phoenix-2014t_s2g.yaml")
prediction_service = PredictionService(
    "experiments/configs/SingleStream/phoenix-2014t_s2t.yaml")

static_dir = "./videos"
BLOCK_SIZE = 1024 * 1024

valkey_client = redis.Redis(host='localhost', port=6379, db=0)


class Sign2TextResult(BaseModel):
    id: str
    task_type: str
    text: str
    gloss: str
    video_url: str


@app.post("/sign2text")
async def process_video(video_file: UploadFile = File(...)) -> Sign2TextResult:
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
                .run(quiet=True)  # Suppress ffmpeg output
            )

            # copy video file to static serve directory
            (
                ffmpeg
                .input(input_path)
                .output(
                    os.path.join(
                        static_dir, f"{task_id}.mp4")
                )
                .run(quiet=True)  # Suppress ffmpeg output
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

        features = feature_extraction_service.extract_feature(output_dir)
        text, gloss = prediction_service.predict(output_dir, features)
        valkey_client.set(f'task:{task_id}:text', text)
        valkey_client.set(f'task:{task_id}:gloss', gloss)
        return Sign2TextResult(id=task_id, task_type="S2T", gloss=gloss, text=text, video_url=f"/static/{task_id}.mp4")


@app.get("/sign2text/{task_id}")
def get_result(task_id: str) -> Sign2TextResult:
    text = valkey_client.get(f'task:{task_id}:text')
    gloss = valkey_client.get(f'task:{task_id}:gloss')
    return Sign2TextResult(
        id=task_id,
        task_type="S2T",
        text=text,
        gloss=gloss,
        video_url=f"/static/{task_id}.mp4",
    )
