"""Download exactly the hosted retrieval models while building the image."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.utils.models import pinned_model_path
from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder


def main() -> None:
    cache = os.environ.get("FASTEMBED_CACHE_PATH", "/opt/models")
    threads = min(8, max(1, int(os.getenv("LUMIN_MODEL_THREADS", "2"))))
    TextEmbedding(model_name="BAAI/bge-small-en-v1.5", cache_dir=cache, threads=threads,
                  specific_model_path=str(pinned_model_path("BAAI/bge-small-en-v1.5", cache)))
    TextCrossEncoder(model_name="Xenova/ms-marco-MiniLM-L-6-v2", cache_dir=cache, threads=threads,
                     specific_model_path=str(pinned_model_path("Xenova/ms-marco-MiniLM-L-6-v2", cache)))


if __name__ == "__main__":
    main()
