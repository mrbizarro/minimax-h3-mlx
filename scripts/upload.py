"""Publish a built MiniMax-H3 MLX quant to the Hub.

    ./.venv/bin/python scripts/upload.py --dir /Volumes/models/h3-mlx-4bit --repo pipenetwork/MiniMax-H3-MLX-4bit

The MiniMax H3 Community License is not an open-source licence, and redistributing derivatives
carries obligations this script enforces rather than leaves to memory:

* ship a copy of the agreement alongside the weights,
* state prominently that the files are modified,
* display "Powered by MiniMax H3",
* record that the grant is territorially limited (worldwide **excluding** the Excluded Territories).

Nothing is uploaded unless `--yes` is passed.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

UPSTREAM = "MiniMaxAI/MiniMax-H3"
CODE_REPO = "https://github.com/PipeNetwork/minimax-h3-mlx"

CARD = """---
license: other
license_name: minimax-h3-community-license
license_link: https://huggingface.co/{upstream}/blob/main/LICENSE
base_model: {upstream}
tags:
- mlx
- apple-silicon
- text-to-video
- image-to-video
- audio-video-generation
- diffusion
pipeline_tag: image-text-to-video
library_name: mlx
---

# {repo_name}

MLX (Apple Silicon) build of the [**MiniMax-H3**](https://huggingface.co/{upstream}) diffusion
transformer. {headline}

> Powered by MiniMax H3.

**These files are modified.** {modified_note} Everything else about the model is unchanged.

## What this is

MiniMax-H3 generates **synchronized video and audio** together. It is not a language model: a 33B
diffusion transformer denoises video and audio latents jointly over one packed sequence, conditioned
by a frozen Qwen3-VL-32B encoder, with separate video and audio VAEs. Running it needs the pipeline
code, not just these weights:

```bash
git clone {code_repo}
cd minimax-h3-mlx && pip install -r requirements.txt
python scripts/generate.py "a red fox leaps over a mossy log" -o fox.mp4
```

This repository holds the **transformer only**. The VAEs and the text encoder come from the
[upstream release](https://huggingface.co/{upstream}); the pipeline loads them directly.

## Size

| | |
|---|---|
| on disk | {gb_on_disk} GB |
| resident during generation | **{gb_resident} GB** |

{precision_block}

## How the widths compare

Measured with teacher forcing — one bfloat16 trajectory recorded, each variant re-predicting the
velocity at those same latents, so the difference is quantization error alone rather than trajectory
divergence. 20 paired observations per variant, aggregated with a paired bootstrap.

| bits | video rel-L2 [95% CI] | audio rel-L2 | video cosine |
|---:|---|---:|---:|
| 8 | 0.0329 [0.0277, 0.0381] | 0.0130 | 0.99941 |
| 6 | 0.0611 [0.0501, 0.0728] | 0.0274 | 0.99791 |
| 4 | 0.1649 [0.1324, 0.1971] | 0.1016 | 0.98456 |
| 3 | 0.2842 [0.2362, 0.3358] | 0.2341 | 0.95635 |

Every interval is disjoint from its neighbours, so the ranking is solid. Two things worth noting:
the steepest step is **6 to 4 bits** (2.7x), not at the low end; and audio degrades faster in
relative terms than video (its share of the error climbs from 0.40x at 8-bit to 0.82x at 3-bit),
plausibly because audio is a small fraction of the packed rows and has less redundancy to absorb
error. 2-bit is not published — extrapolation puts it near 50% velocity error.

Note that relative velocity error is not the same as "that much worse output": the scheduler
integrates velocity across steps, so per-step error compounds along the trajectory. These numbers
rank the widths reliably; they do not by themselves locate the point where output becomes unusable.

## Read this before choosing a quant

MiniMax has not released its sparse-attention implementation, so inference runs **dense** attention
over tens of thousands of rows. On an M3 Ultra a single denoising step costs about **8.8 minutes**
for a 5-second clip (37,966 packed rows) and **1.04 hours** for 15 seconds (109,318 rows).

Quantization does not change that. The bottleneck is attention FLOPs, which quantization does not
reduce; the linear layers are ~42% of the work at 5 s and ~20% at 15 s, so a 4-bit build is worth
roughly **1.2-1.4x end to end**. Choose a quant to *fit* H3 on your machine, not to make it quick.

## Licence

Governed by the [MiniMax H3 Community License](https://huggingface.co/{upstream}/blob/main/LICENSE),
a copy of which is included in this repository. It is **not** an open-source licence. Notably:
redistribution must carry the agreement and mark modified files; commercial products above $20M
yearly revenue need separate authorization from MiniMax; and **the grant is territorially limited**
(worldwide, excluding the Excluded Territories defined in the agreement). By downloading these
weights you accept those terms.

The MLX port code is Apache-2.0 and lives at [{code_repo}]({code_repo}).
"""


ADALN_NOTE = """The gap is deliberate. ~13B of H3's 33B parameters are the per-block AdaLN projections, whose only
input is the timestep embedding. For a fixed sampler schedule every modulation tensor a run needs is
precomputed once into a small table, and the projections are then dropped — so they are on disk but
never resident. The table scales with step count, not model size: measured at **145 MB for a 9-step
schedule** and 745 MB for 40 steps, against the 26 GB it replaces."""

QUANT_PRECISION = ADALN_NOTE + """

Those projections are quantized to **8-bit** here. That was measured, not assumed: quantizing them
shifts the modulation table by **0.25%**, an order of magnitude less than the {bits}-bit core's own
velocity error, and takes 12.2 GB off this download. (4-bit AdaLN is measurably worse — 0.77% on the
table, 2.8% on its worst tensor — and is not used at any core width.)"""

BF16_PRECISION = ADALN_NOTE + """

**Nothing here is quantized.** This is the faithful conversion, and it preserves the release's
*mixed* precision rather than flattening it: MiniMax ships the two patch projections, the timestep
MLP and both output heads in **float32** and everything else in bfloat16, and that split is kept
intact. Casting those twelve tensors down to bfloat16 would be a downgrade from the release — the
timestep MLP in particular feeds every block's modulation, so rounding it perturbs all 50 blocks at
every sampling step.

Use this as the quality reference, or as the base for further conversion. If you only want to
generate and have the memory, it is the best output available; if you do not, the
[8-bit build](https://huggingface.co/pipenetwork/MiniMax-H3-MLX-8bit) is 27.6 dB PSNR against it and
roughly half the size."""

F32_PRECISION = ADALN_NOTE + """

**This build carries no more information than the
[bf16 one](https://huggingface.co/pipenetwork/MiniMax-H3-MLX-bf16).** MiniMax's weights are bfloat16
on disk, so upcasting to float32 is a lossless widening, not extra precision — you are downloading
132.6 GB of data that fits losslessly in 66.3 GB. It is published for float32 fine-tuning and
numerics work, where having the base already widened is convenient.

If you only want to generate, take bf16 instead. If you want float32 *compute* from a smaller
download, `load_dit(dtype=mx.float32)` upcasts at load time and gives an identical model."""


def build_card(repo: str, meta: dict) -> str:
    """Render the card for a quantized or an unquantized build."""
    if meta.get("quantized") is False:
        policy = meta.get("dtype_policy", "native")
        if policy == "float32":
            headline = "Unquantized, upcast to **float32**."
            modified = "The transformer weights have been converted to MLX and widened to float32;"
            precision = F32_PRECISION
        else:
            headline = "Unquantized, in the release's native **mixed bfloat16/float32** precision."
            modified = "The transformer weights have been converted to MLX;"
            precision = BF16_PRECISION
        bits = group_size = "n/a"
    else:
        bits = meta.get("bits", "?")
        group_size = meta.get("group_size", "?")
        headline = f"Quantized to **{bits}-bit** (group size {group_size})."
        modified = "The transformer weights have been converted to MLX and quantized;"
        precision = QUANT_PRECISION

    return CARD.format(
        upstream=UPSTREAM,
        code_repo=CODE_REPO,
        repo_name=repo.split("/")[-1],
        headline=headline,
        modified_note=modified + " they are not MiniMax's originals.",
        precision_block=precision.format(bits=bits),
        gb_on_disk=meta.get("gb_on_disk", "?"),
        gb_resident=meta.get("gb_resident_after_adaln_drop", "?"),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, help="a directory built by scripts/build_quant.py")
    parser.add_argument("--repo", required=True, help="e.g. pipenetwork/MiniMax-H3-MLX-4bit")
    parser.add_argument("--license", default="/Volumes/models/MiniMax-H3/LICENSE",
                        help="path to the upstream LICENSE, copied into the repo")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--yes", action="store_true", help="actually upload")
    args = parser.parse_args()

    directory = Path(args.dir)
    meta_path = next(
        (directory / n for n in ("quant_config.json", "build_config.json") if (directory / n).exists()),
        None,
    )
    if meta_path is None:
        print(f"error: no quant_config.json or build_config.json in {directory} — "
              "build with scripts/build_quant.py or scripts/build_unquantized.py first")
        return 1
    quant_meta = json.loads(meta_path.read_text())

    licence = Path(args.license)
    if licence.exists():
        shutil.copy(licence, directory / "LICENSE")
    else:
        print(f"error: upstream LICENSE not found at {licence}. The MiniMax H3 Community License "
              "requires shipping a copy of the agreement with any redistribution; refusing to upload.")
        return 1

    card = build_card(args.repo, quant_meta)
    (directory / "README.md").write_text(card)

    files = sorted(p.name for p in directory.iterdir() if p.is_file())
    total = sum(p.stat().st_size for p in directory.iterdir() if p.is_file())
    print(f"repo   {args.repo}{' (private)' if args.private else ''}")
    print(f"dir    {directory}")
    print(f"files  {len(files)}, {total / 1e9:.1f} GB")
    for name in files[:10]:
        print(f"   {name}")
    if len(files) > 10:
        print(f"   ... and {len(files) - 10} more")

    if not args.yes:
        print("\ndry run — pass --yes to upload")
        return 0

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.repo, private=args.private, exist_ok=True, repo_type="model")
    api.upload_folder(folder_path=str(directory), repo_id=args.repo, repo_type="model")
    print(f"\nuploaded https://huggingface.co/{args.repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
