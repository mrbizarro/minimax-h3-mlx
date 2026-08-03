"""Download only the checkpoint files used by the staged 64 GB quality path."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="experiment root containing models/ and hf_home/",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    os.environ.setdefault("HF_HOME", str(root / "hf_home"))
    # Hugging Face documents this as the Xet mode that maximizes local network/disk utilization;
    # these 28–41 GB single files benefit materially from concurrent range fetches.
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

    from huggingface_hub import hf_hub_download

    selections = [
        (
            "DeepBeepMeep/MiniMax-H3",
            ["MiniMax-H3-FL2VA-pruned_bf16.safetensors"],
            root / "models" / "deepbeep-pruned-bf16",
        ),
        (
            "ddalcu/MiniMax-H3-FL2VA-MLX-Serve-8bit",
            [
                "text_encoder.safetensors",
                "video_vae.safetensors",
                "audio_vae.safetensors",
                "config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "vocab.json",
                "merges.txt",
                "LICENSE",
                "NOTICE",
                "MODIFICATIONS.md",
            ],
            root / "models" / "ddalcu-q8",
        ),
        (
            "MiniMaxAI/MiniMax-H3",
            ["FL2VA/text_encoder/config.json", "FL2VA/model_index.json"],
            root / "models" / "upstream-meta",
        ),
    ]

    for repo, files, destination in selections:
        destination.mkdir(parents=True, exist_ok=True)
        for filename in files:
            print(f"{repo}: {filename}", flush=True)
            hf_hub_download(repo, filename=filename, local_dir=destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
