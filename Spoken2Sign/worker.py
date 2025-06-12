from typing import TypedDict
import valkey
import json
from service import RenderAvatarService


class Avatar(TypedDict):
    id: str
    name: str


available_avatars = [
    Avatar(id="male", name="Male 男性"),
    # Avatar(id="female", name="Female 女性"),
]

if __name__ == "__main__":
    print("[INFO] Starting worker...")
    valkey_client = valkey.Valkey(host="localhost", port=6379)
    # render_service = RenderAvatarService()
    render_service = {avatar['id']:
                      RenderAvatarService(
                          blender_mainfile=f'../../pretrained_models/smplx_tommy_{avatar["id"]}_backport.blend', smplx_model_object=f'SMPLX-{avatar["id"]}')
                      for avatar in available_avatars
                      }
    print("[INFO] Worker is running...")
    while True:
        task_id = valkey_client.brpop("render:jobs")[1].decode("utf-8")
        print(f"[INFO] Received task with ID {task_id}")
        valkey_client.set(f"task:{task_id}:status", "rendering")
        avatar = valkey_client.get(f'task:{task_id}:avatar').decode()
        if avatar not in [avatar['id'] for avatar in available_avatars]:
            print(f"[WARN] Avatar {avatar} is not available")
            continue
        gloss = valkey_client.get(f'task:{task_id}:gloss').decode()
        gloss_frame_mapping = json.loads(
            valkey_client.get(f'task:{task_id}:gloss_frame_mapping'))
        render_service[avatar].generate_subtitles(
            task_id, gloss, gloss_frame_mapping)
        render_service[avatar].render_images(task_id, lambda progress: valkey_client.set(
            f"task:{task_id}:progress", progress))
        # render_service.render_video(task_id, lambda progress: valkey_client.set(
        #     f"task:{task_id}:progress", progress))
        valkey_client.set(f"task:{task_id}:status", "completed")
