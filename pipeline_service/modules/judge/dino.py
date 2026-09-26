
from __future__ import annotations

import asyncio
import io
from typing import Any

from logger_config import logger
from modules.judge.embedder_settings import EmbedderConfig


class DinoEmbedder:
    def __init__(self, config: EmbedderConfig) -> None:
        self.config = config
        self._model: Any = None
        self._processor: Any = None
        self._torch: Any = None
        self._np: Any = None
        self._device: str | None = None
        self._disabled = False
        self._load_lock = asyncio.Lock()

    def _load_sync(self) -> bool:
        """Import deps + load the model on the calling thread. Returns success."""
        try:
            import numpy as np  
            import torch  
            from transformers import AutoImageProcessor, AutoModel  
        except Exception as exc:  
            logger.warning(f"[DINO] dependencies unavailable ({exc!r}); S2BV disabled")
            return False

        device = self.config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        from_pretrained_kwargs = {
            "revision": self.config.revision,
            "token": self.config.hf_token,
            "trust_remote_code": self.config.trust_remote_code,
        }
        try:
            processor = AutoImageProcessor.from_pretrained(
                self.config.model_id, **from_pretrained_kwargs
            )
            model = (
                AutoModel.from_pretrained(self.config.model_id, **from_pretrained_kwargs)
                .eval()
                .to(device)
            )
        except Exception as exc:  
            logger.warning(
                f"[DINO] failed to load model {self.config.model_id!r} on {device}: "
                f"{exc!r}; S2BV disabled"
            )
            return False

        self._np = np
        self._torch = torch
        self._processor = processor
        self._model = model
        self._device = device
        logger.info(f"[DINO] loaded {self.config.model_id} on {device}")
        return True

    async def _ensure_loaded(self) -> bool:
        if self._model is not None:
            return True
        if self._disabled:
            return False
        async with self._load_lock:
            if self._model is not None:
                return True
            if self._disabled:
                return False
            ok = await asyncio.to_thread(self._load_sync)
            if not ok:
                self._disabled = True
            return ok

    def _embed_sync(self, images: list[bytes]) -> Any:
        """Return an (N, D) float32 L2-normalized numpy array for the given PNGs."""
        from PIL import Image  

        torch = self._torch
        pil = [Image.open(io.BytesIO(b)).convert("RGB") for b in images]
        vecs = []
        bs = max(1, self.config.batch_size)
        for i in range(0, len(pil), bs):
            batch = pil[i : i + bs]
            inputs = self._processor(images=batch, return_tensors="pt").to(self._device)
            with torch.inference_mode():
                out = self._model(**inputs)
            pooled = getattr(out, "pooler_output", None)
            if pooled is None:
                pooled = out.last_hidden_state[:, 0]  
            pooled = torch.nn.functional.normalize(pooled, dim=-1)
            vecs.append(pooled.detach().to("cpu").float().numpy())
        return self._np.concatenate(vecs, axis=0)

    @staticmethod
    def view_similarities(embeddings: bytes | None) -> dict[str, float]:
        """Cosine similarity of every `view_<name>` vector in a candidate npz to its `prompt` vector."""
        if not embeddings:
            return {}
        import numpy as np

        with np.load(io.BytesIO(embeddings)) as data:
            prompt = data["prompt"]
            return {
                key[len("view_"):]: round(float(np.dot(data[key], prompt)), 4)
                for key in data.files
                if key.startswith("view_")
            }

    @staticmethod
    def similarities(ref_vec: Any | None, vecs: dict[str, Any]) -> dict[str, float]:
        """Cosine similarity of in-memory view vectors to the reference vector."""
        if ref_vec is None or not vecs:
            return {}
        import numpy as np

        return {name: round(float(np.dot(v, ref_vec)), 4) for name, v in vecs.items()}

    async def embed_reference(self, image: bytes) -> Any | None:
        """Embed the reference image. Returns an opaque vector or None if disabled."""
        if not image or not await self._ensure_loaded():
            return None
        try:
            arr = await asyncio.to_thread(self._embed_sync, [image])
            return arr[0]
        except Exception as exc:  
            logger.warning(f"[DINO] reference embedding failed: {exc!r}")
            return None

    async def embed_views(self, views: dict[str, bytes]) -> dict[str, Any] | None:
        """Embed rendered views into in-memory vectors keyed by view name.

        Returns None when the embedder is disabled or nothing embeddable was given.
        """
        if not views or not await self._ensure_loaded():
            return None
        names = [n for n, b in views.items() if b]
        if not names:
            return None
        try:
            arr = await asyncio.to_thread(self._embed_sync, [views[n] for n in names])
        except Exception as exc:  
            logger.warning(f"[DINO] view embedding failed: {exc!r}")
            return None
        return dict(zip(names, arr))

    async def build_candidate_embeddings(
        self, ref_vec: Any | None, views: dict[str, bytes]
    ) -> bytes | None:
        """Embed candidate views and pack {prompt, view_<name>...} into npz bytes —
        the transport format the judge and the candidate export read.

        Returns None when the embedder is disabled, ``ref_vec`` is missing, or no
        views were rendered — the judge then runs S2BV-free.
        """
        if ref_vec is None:
            return None
        vecs = await self.embed_views(views)
        if not vecs:
            return None
        try:
            payload = {"prompt": ref_vec}
            for n, vec in vecs.items():
                payload[f"view_{n}"] = vec
            buf = io.BytesIO()
            self._np.savez(buf, **payload)
            return buf.getvalue()
        except Exception as exc:  
            logger.warning(f"[DINO] candidate embedding failed: {exc!r}")
            return None
