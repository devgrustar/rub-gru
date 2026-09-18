from __future__ import annotations

from pydantic import BaseModel


class EmbedderConfig(BaseModel):
    """DINOv3 embedder config for the judge best-view stage (S2BV)."""

    enabled: bool = True
    model_id: str = "computer-vision-ai-lab/dinov3-vits16-pretrain-lvd1689m"
    revision: str = "e2b5191960331471bf2734d372e6a7151a6079c5"
    hf_token: str | None = None
    device: str | None = None  
    batch_size: int = 8
    trust_remote_code: bool = False
