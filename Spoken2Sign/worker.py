import valkey
from service import RenderAvatarService

if __name__ == "__main__":
    print("[INFO] Starting worker...")
    valkey_client = valkey.Valkey(host="localhost", port=6379)
    render_service = RenderAvatarService()
    print("[INFO] Worker is running...")
    while True:
        task_id = valkey_client.brpop("render:jobs")[1].decode("utf-8")
        print(f"[INFO] Received task with ID {task_id}")
        valkey_client.set(f"task:{task_id}:status", "rendering")
        render_service.render_images(task_id, lambda progress: valkey_client.set(f"task:{task_id}:progress", progress))
        valkey_client.set(f"task:{task_id}:status", "completed")