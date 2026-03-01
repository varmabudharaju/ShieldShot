"""Download pretrained model weights for ShieldShot."""

from pathlib import Path


MODELS_DIR = Path.home() / ".shieldshot" / "models"


def download_insightface():
    print("Checking InsightFace models...")
    try:
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(name="buffalo_sc", providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=-1, det_size=(640, 640))
        print("  InsightFace models ready.")
    except Exception as e:
        print(f"  Error: {e}")
        print("  Run: pip install insightface onnxruntime")


def download_facenet():
    print("Checking FaceNet model...")
    try:
        from facenet_pytorch import InceptionResnetV1
        model = InceptionResnetV1(pretrained="vggface2")
        print("  FaceNet model ready.")
    except Exception as e:
        print(f"  Error: {e}")
        print("  Run: pip install facenet-pytorch")


def setup_dirs():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Model directory: {MODELS_DIR}")


def main():
    setup_dirs()
    download_insightface()
    download_facenet()
    print("\nAll models ready.")


if __name__ == "__main__":
    main()
