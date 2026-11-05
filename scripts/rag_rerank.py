"""固定版の日本語 int8 リランカーを CPU で実行する。"""
from __future__ import annotations

import urllib.request
import uuid
from pathlib import Path

from filelock import FileLock

MODELS = {
    "xsmall-int8": ("hotchpotch/japanese-reranker-xsmall-v2", "de99fd2f16c7b5df1df1bcc1d9ad2c16d88ce93a"),
    "tiny-int8": ("hotchpotch/japanese-reranker-tiny-v2", "ba95175a4d53058816b971f31929f10c5cad8560"),
}


class Reranker:
    def __init__(self, config):
        import onnxruntime as ort
        from transformers import AutoTokenizer

        self.model_id, revision = MODELS[config.reranker]
        folder = config.model_cache_dir / "rerankers"
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / f"{config.reranker}-{revision}.onnx"
        with FileLock(str(target) + ".lock", timeout=600):
            if not target.exists():
                temporary = target.with_suffix(f".{uuid.uuid4().hex}.part")
                try:
                    url = f"https://huggingface.co/{self.model_id}/resolve/{revision}/onnx/model_qint8_avx2.onnx"
                    with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as stream:
                        while block := response.read(1024 * 1024):
                            stream.write(block)
                    temporary.replace(target)
                finally:
                    temporary.unlink(missing_ok=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, revision=revision, cache_dir=str(config.model_cache_dir))
        options = ort.SessionOptions()
        options.intra_op_num_threads = config.threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(target), sess_options=options, providers=["CPUExecutionProvider"])
        self.inputs = {item.name for item in self.session.get_inputs()}

    def score(self, pairs: list[tuple[str, str]]) -> list[float]:
        values = []
        for start in range(0, len(pairs), 8):
            batch = pairs[start:start+8]
            encoded = self.tokenizer([pair[0] for pair in batch], [pair[1] for pair in batch],
                                     padding=True, truncation=True, max_length=512, return_tensors="np")
            result = self.session.run(None, {key:value for key,value in encoded.items() if key in self.inputs})[0]
            values.extend(map(float, result.reshape(-1)))
        return values
