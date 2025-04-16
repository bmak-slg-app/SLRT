from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
import ffmpeg
import tempfile
import os
import zipfile
import uuid

from service import FeatureExtractionService, PredictionService

app = FastAPI()

feature_extraction_service = FeatureExtractionService("experiments/configs/SingleStream/phoenix-2014t_s2g.yaml")
prediction_service = PredictionService("experiments/configs/SingleStream/phoenix-2014t_s2t.yaml")

@app.post("/upload")
async def process_video(video_file: UploadFile = File(...)) -> str:
    # Create temporary working directory
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Generate unique filenames
            input_path = os.path.join(temp_dir, "input.mp4")
            output_dir = os.path.join(temp_dir, "frames")

            # Save uploaded file to temporary directory
            with open(input_path, "wb") as f:
                f.write(await video_file.read())

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
        return prediction_service.predict(output_dir, features)
