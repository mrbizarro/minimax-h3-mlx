"""Media conditioning: carried context, timed/last keyframes, a held soundtrack, a held picture.

No weights. Two layers:

* the geometry and planning maths (packing with history, frozen-row timesteps, keyframe routing
  across a chain, the official alignment sentence, flag refusals) in pure NumPy, and
* one end-to-end pass of ``render_window`` through TINY random modules (DiT, both VAEs, a fake
  text encoder) on the CPU, which proves the plumbing a real render depends on: a held
  soundtrack decodes to exactly its own VAE round trip, a held picture is delivered untouched,
  history and keyframe rows land in the conditioning block, and a plain run is unchanged.

    ./.venv/bin/python tests/test_media_conditioning.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import generate_staged as gs  # noqa: E402
from minimax_h3_mlx import packing  # noqa: E402
from minimax_h3_mlx.audio_vae import AudioVAE, AudioVAEConfig  # noqa: E402
from minimax_h3_mlx.conditioning import (  # noqa: E402
    encode_audio_rows,
    encode_video_rows,
    frozen_row_mask,
    gate_ranked_skip_blocks,
    video_latents_for_frames,
)
from minimax_h3_mlx.config import TAG_AUDIO, TAG_TEXT, TAG_VIDEO, DiTConfig  # noqa: E402
from minimax_h3_mlx.dit import MiniMaxH3DiT  # noqa: E402
from minimax_h3_mlx.media import read_video_frames  # noqa: E402
from minimax_h3_mlx.video_vae import VideoVAE, VideoVAEConfig  # noqa: E402

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'ok  ' if ok else 'FAIL'}  {name}{(' — ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


# ---------------------------------------------------------------------------------------------
# geometry


def test_packing_without_history_is_unchanged() -> None:
    tags = [TAG_TEXT] * 9
    base = packing.build_packed_sequence(tags, 12, 8, 8, 20, (1, 2, 2), ("first", "last"))
    pos = np.array(base.position_ids)
    rows = 16
    first_t = pos[9, 0]
    last_t = pos[9 + rows, 0]
    check("first anchor sits at the text length", first_t == 9.0, str(first_t))
    want_last = 9.0 + packing._temporal_position_span(12) - 5.0 / 3.0
    check("last anchor at span - 5/3", abs(last_t - want_last) < 1e-5, f"{last_t} vs {want_last}")
    check("no audio conditioning rows", base.num_condition_audio_rows == 0)

    frames = 17 * 2 + 5  # 12 latent frames
    timed = packing.build_packed_sequence(
        tags, 12, 8, 8, 20, (1, 2, 2), ("frame:0", f"frame:{frames - 1}")
    )
    tpos = np.array(timed.position_ids)
    check("frame:0 == first", tpos[9, 0] == first_t)
    check("frame:F-1 == last", abs(tpos[9 + rows, 0] - last_t) < 1e-4, f"{tpos[9 + rows, 0]} vs {last_t}")
    mid = packing.build_packed_sequence(tags, 12, 8, 8, 20, (1, 2, 2), ("frame:12",))
    check("frame:12 at origin + 12*5/3", abs(np.array(mid.position_ids)[9, 0] - (9 + 20.0)) < 1e-4)
    try:
        packing.build_packed_sequence(tags, 12, 8, 8, 20, (1, 2, 2), (f"frame:{frames}",))
        check("frame past the window refused", False)
    except ValueError:
        check("frame past the window refused", True)


def test_packing_with_history() -> None:
    tags = [TAG_TEXT] * 7
    rows = 16
    hist = 5  # one 17-frame clip
    lay = packing.build_packed_sequence(
        tags, 12, 8, 8, 20, (1, 2, 2), ("first",),
        history_latent_frames=hist, history_audio_latents=30, history_audio_offset=0.25,
    )
    pos = np.array(lay.position_ids)
    span = packing._temporal_position_span(hist)
    check("history span of one clip is 17 frames", abs(span - 17 * 5 / 3) < 1e-9, str(span))
    check("history starts at the text length", pos[7, 0] == 7.0)
    check("condition video rows = history + keyframe", lay.num_condition_video_rows == (hist + 1) * rows)
    first_row = 7 + hist * rows
    check("first anchor moves to the target origin", abs(pos[first_row, 0] - (7 + span)) < 1e-4)
    audio_cond = 7 + (hist + 1) * rows
    check("carried audio starts at origin + offset", abs(pos[audio_cond, 0] - 7.25) < 1e-6)
    check("audio condition rows (stereo)", lay.num_condition_audio_rows == 60)
    target_audio = audio_cond + 60
    check("target audio starts at the target origin", abs(pos[target_audio, 0] - (7 + span)) < 1e-4)
    video_start = target_audio + 40
    check("target video starts at the target origin", abs(pos[video_start, 0] - (7 + span)) < 1e-4)
    tag = np.array(lay.token_tags)
    check("history rows are video-tagged", (tag[7 : 7 + hist * rows] == TAG_VIDEO).all())
    check("carried audio rows are audio-tagged", (tag[audio_cond : audio_cond + 60] == TAG_AUDIO).all())
    vid = np.array(lay.video_indices)
    aud = np.array(lay.audio_indices)
    check("video indices: conditions then target", vid[0] == 7 and vid[lay.num_condition_video_rows] == video_start)
    check("audio indices: conditions then target", aud[0] == audio_cond and aud[60] == target_audio)


def test_frozen_timesteps() -> None:
    tags = [TAG_TEXT] * 4
    lay = packing.build_packed_sequence(tags, 7, 4, 4, 10, (1, 2, 2), ("first",))
    n_target_video = int(lay.video_indices.shape[0]) - lay.num_condition_video_rows
    n_target_audio = int(lay.audio_indices.shape[0])
    distinct, inverse = packing.build_row_timesteps(
        lay, 0.2, 0.3, 0.999, 1.0,
        frozen_video_rows=None,
        frozen_audio_rows=frozen_row_mask(n_target_audio, n_target_audio),
    )
    table = np.array(distinct)[np.array(inverse)]
    aud = np.array(lay.audio_indices)
    vid = np.array(lay.video_indices)
    check("held soundtrack rows are clean (t=1)", (table[aud] == 1.0).all())
    check("generated video rows follow the schedule", np.allclose(table[vid[lay.num_condition_video_rows:]], 0.2))
    distinct, inverse = packing.build_row_timesteps(
        lay, 0.2, 0.3, 0.999, 1.0, frozen_video_rows=np.ones(n_target_video, bool)
    )
    table = np.array(distinct)[np.array(inverse)]
    check("held picture rows are clean (t=1)", (table[vid[lay.num_condition_video_rows:]] == 1.0).all())
    check("generated audio rows follow the schedule", np.allclose(table[aud], 0.3))


# ---------------------------------------------------------------------------------------------
# planning


def test_keyframe_routing() -> None:
    check("frame 0 -> window 0", gs.window_of_frame(0, 124) == (0, 0))
    check("frame 123 -> last of window 0", gs.window_of_frame(123, 124) == (0, 123))
    check("frame 124 -> window 1 frame 1", gs.window_of_frame(124, 124) == (1, 1))
    check("frame 246 -> last of window 1", gs.window_of_frame(246, 124) == (1, 123))
    check("seconds", gs.resolve_keyframe_time("2.5s", 240) == 60)
    check("negative counts from the end", gs.resolve_keyframe_time("-1", 240) == 239)
    try:
        gs.resolve_keyframe_time("300", 240)
        check("out of range refused", False)
    except ValueError:
        check("out of range refused", True)
    ns = argparse.Namespace(first_frame=Path("a.png"), last_frame=Path("z.png"),
                            keyframe=["m.png@124", "k.png@3s"])
    plan = gs.plan_keyframes(ns, 124, 2, 240)
    check("window 0 gets first + 3 s", [l for _, l in plan[0]] == [0, 72], str(plan[0]))
    check("window 1 gets 124 and the last delivered frame",
          [l for _, l in plan[1]] == [1, 116], str(plan[1]))
    check("anchor names", [gs.anchor_name(l, 124) for l in (0, 72, 123)] == ["first", "frame:72", "last"])


def test_alignment_line() -> None:
    p = "integrated_multimodal_description: [Shot 1] Live-action ..."
    check("plain first frame: no line (unchanged contract)", gs.alignment_line(["first"], 124, p) is None)
    fl = gs.alignment_line(["first", "last"], 124, p)
    want = ("How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns "
            "with the 0.00-second mark of the target video; Picture 2 (from Shot 1) aligns with the "
            "5.17-second mark of the target video.")
    check("FL2VA official sentence", fl == want, fl or "")
    last = gs.alignment_line(["last"], 124, p)
    check("L2VA official sentence",
          last == "How the reference pictures align with the target video — <Picture 1> (from "
                  "[Shot 1]) aligns with the 5.17-second mark of the target video.", last or "")
    timed = gs.alignment_line(["first", "frame:60"], 124, "[Shot 1] a ... [Shot 2] At 00:03.000, b")
    check("timed keyframe names its second and the final shot",
          "Picture 2 (from Shot 2) aligns with the 2.50-second mark" in (timed or ""), timed or "")
    check("an existing line is respected",
          gs.alignment_line(["first", "last"], 124, "How the reference pictures align ... x") is None)


def _parser():
    class P(argparse.ArgumentParser):
        def error(self, message):  # noqa: D401
            raise SystemExit(message)

    return P()


def _args(**kw):
    base = dict(chain_context_frames=0, keyframe=None, extend_from=None, video_to_audio=None,
                audio_drive=None, audio_mux=None, last_frame=None, first_frame=None,
                draft_fps=None, playback_fps=None, draft_seconds=None, draft_cache_dir=None,
                save_stage_a=None, frames=124, extend_frames=None, refmod=None)
    base.update(kw)
    return argparse.Namespace(**base)


def test_flag_refusals(tmp: Path) -> None:
    clip = tmp / "c.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=64x48:rate=24",
                    "-t", "1", "-pix_fmt", "yuv420p", str(clip)], check=True)
    img = tmp / "i.png"
    img.write_bytes(b"x")

    def refused(**kw) -> bool:
        try:
            gs.media_mode_errors(_args(**kw), _parser())
            return False
        except SystemExit:
            return True

    check("context must be a multiple of 17", refused(chain_context_frames=10))
    check("v2a takes no keyframes", refused(video_to_audio=clip, first_frame=img))
    check("extend takes no first frame", refused(extend_from=clip, first_frame=img))
    check("extend + drive refused", refused(extend_from=clip, audio_drive=clip))
    check("mux without drive refused", refused(audio_mux=clip))
    check("keyframes refuse draft fps", refused(last_frame=img, draft_fps=12))
    check("missing keyframe file refused", refused(keyframe=[str(tmp / "nope.png") + "@3"]))
    ns = _args(extend_from=clip)
    mode = gs.media_mode_errors(ns, _parser())
    check("extend defaults to 17 frames of context", mode == "extend" and ns.chain_context_frames == 17)
    check("plain text mode", gs.media_mode_errors(_args(), _parser()) == "text")
    real_probe = gs.probe_media
    try:
        gs.probe_media = lambda path: {"duration": gs.MEDIA_SOURCE_MAX_SECONDS + 1.0}
        check("an over-long extend source is refused", refused(extend_from=clip))
        check("an over-long add-sound source is refused", refused(video_to_audio=clip))
    finally:
        gs.probe_media = real_probe
    frames = read_video_frames(clip, 32, 32)
    check("the streaming reader returns every frame on the canvas",
          frames.shape == (24, 32, 32, 3) and frames.dtype == np.uint8, str(frames.shape))
    check("max_frames caps the read", read_video_frames(clip, 32, 32, max_frames=5).shape[0] == 5)


# ---------------------------------------------------------------------------------------------
# end to end through tiny modules


def tiny_video_vae() -> VideoVAE:
    cfg = VideoVAEConfig(
        in_channels=3, out_channels=3, latent_channels=8, block_out_channels=(8, 8, 16, 16),
        layers_per_block=1, spatial_downsample_factors=(2, 2, 2, 2),
        temporal_downsample_factors=(1, 2, 2, 1), norm_num_groups=4, decoder_num_layers=2,
        decoder_num_attention_heads=2, decoder_attention_head_dim=16,
        decoder_num_register_tokens=2, decoder_ffn_mult=2,
        latents_mean=tuple([0.1] * 8), latents_std=tuple([0.9] * 8),
    )
    mx.random.seed(1)
    vae = VideoVAE(cfg)
    mx.eval(vae.parameters())
    return vae


def tiny_audio_vae() -> AudioVAE:
    cfg = AudioVAEConfig(
        encoder_dim=4, encoder_rates=(2, 4, 4, 5, 5), latent_dim=16, latent_channels=8,
        num_attention_heads=2, decoder_dim=256, decoder_rates=(5, 5, 2, 2, 2, 2, 2),
        decoder_kernel_sizes=(9, 9, 4, 4, 4, 4, 4), resblock_kernel_sizes=(3,),
        resblock_dilation_sizes=((1,),), latents_mean=tuple([0.0] * 8), latents_std=tuple([1.0] * 8),
    )
    mx.random.seed(2)
    vae = AudioVAE(cfg)
    mx.eval(vae.parameters())
    return vae


def tiny_dit() -> MiniMaxH3DiT:
    hidden = 32
    cfg = DiTConfig(
        hidden_size=hidden, num_layers=3, token_refiner_num_layers=1, num_attention_heads=2,
        attention_head_dim=16, ffn_hidden_size=32, latents_dim=8, audio_latents_dim=8,
        patch_size=(1, 2, 2), text_dim=16, timestep_input_dim=16, time_embed_hidden_size=hidden,
        time_embed_dim=16, adaln_out_features=6 * 3 * hidden, final_adaln_out_features=2 * hidden,
        rope_inv_freq_len=2,
    )
    mx.random.seed(3)
    dit = MiniMaxH3DiT(cfg)
    mx.eval(dit.parameters())
    return dit


class FakeTextEncoder:
    seen: list = []

    def __init__(self, *a, **k):
        pass

    def encode(self, prompt, images=None):
        FakeTextEncoder.seen.append((prompt, len(images or [])))
        n = 6 + 3 * len(images or [])
        tags = np.array([TAG_TEXT] * 6 + [TAG_VIDEO] * (n - 6), dtype=np.int64)
        return mx.random.normal((1, n, 16)), tags


class Recorder:
    def __init__(self):
        self.data = {}

    def flush(self):
        pass

    def phase(self, name):
        import contextlib

        return contextlib.nullcontext()


def install_tiny_modules():
    video_vae, audio_vae = tiny_video_vae(), tiny_audio_vae()
    gs.MiniMaxH3TextEncoder = FakeTextEncoder
    gs.load_compact_video_vae = lambda *a, **k: video_vae
    gs.load_compact_audio_vae = lambda *a, **k: audio_vae
    gs.load_dit = lambda *a, **k: tiny_dit()
    gs.dit_precision = lambda *a, **k: "bf16"
    gs.check_decoded_video = lambda *a, **k: None
    return video_vae, audio_vae


def run_window(cond, frames=22, height=64, width=64):
    args = argparse.Namespace(
        compact_root=Path("/nonexistent"), text_config=None, dit=Path("/nonexistent"),
        height=height, width=width, steps=3, sigma_subset=None, lora=None, lora_adaln=None,
        step_cache=0.0, step_cache_probe=False, step_cache_max_skip=2, draft_decode="full",
        draft_cache_dir=None, save_stage_a=None, skip_blocks=None,
        sigma_knots=None, sampler="euler", video_shift=None, audio_shift=None,
    )
    rec = Recorder()
    video, audio, rate, steps = gs.render_window(
        args, rec, label="", prompt="p", frames=frames, seed=7, keyframe=None,
        first_frame_key="", prompt_cache=None, cond=cond,
    )
    return video, audio, rec


def test_end_to_end() -> None:
    video_vae, audio_vae = install_tiny_modules()
    rng = np.random.default_rng(0)
    from PIL import Image

    frames, h, w = 22, 64, 64
    plain_v, plain_a, rec = run_window(gs.WindowConditioning())
    check("plain window decodes the requested frames", plain_v.shape == (frames, h, w, 3), str(plain_v.shape))
    check("plain window: no conditioning recorded", rec.data["conditioning"]["anchors"] == [])

    still = Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8))
    hist = rng.integers(0, 255, (17, h, w, 3), dtype=np.uint8)
    hist_audio = (rng.standard_normal((2, gs.frame_samples(18))) * 0.1).astype(np.float32)
    cond = gs.WindowConditioning(
        keyframes=[(still, "first"), (still, "frame:9"), (still, "last")],
        history_frames=hist, history_audio=hist_audio,
    )
    v, a, rec = run_window(cond)
    check("conditioned window decodes", v.shape == (frames, h, w, 3))
    check("three pictures reach the text encoder", FakeTextEncoder.seen[-1][1] == 3)
    check("conditioning recorded", rec.data["conditioning"]["history_frames"] == 17
          and rec.data["conditioning"]["anchors"] == ["first", "frame:9", "last"])

    # A held soundtrack: the delivered audio is the VAE round trip of the drive, bit for bit.
    drive = (np.sin(np.linspace(0, 400, gs.frame_samples(frames))) * 0.3).astype(np.float32)
    drive = np.stack([drive, drive * 0.5])
    _, held_audio, _ = run_window(gs.WindowConditioning(drive_audio=drive))
    latents = packing.audio_latent_num_frames(frames)
    rows = encode_audio_rows(audio_vae, drive, latents)
    expected = gs.decode_audio(audio_vae, rows, latents, frames)
    check("held soundtrack survives the denoise exactly",
          np.allclose(held_audio, expected, atol=1e-5), f"max diff {np.abs(held_audio - expected).max():.2e}")

    # A held picture: delivered untouched, sound generated.
    pic = rng.integers(0, 255, (frames, h, w, 3), dtype=np.uint8)
    v2a_video, v2a_audio, _ = run_window(gs.WindowConditioning(
        keyframes=[(Image.fromarray(pic[0]), "first")], frozen_frames=pic))
    check("held picture is delivered untouched", np.array_equal(v2a_video, pic))
    check("sound is generated for it", v2a_audio.shape[0] == 2 and np.abs(v2a_audio).sum() > 0)

    # Encoders: shapes on both frame conventions.
    check("17 frames keep 5 latents", video_latents_for_frames(17) == 5)
    check("22 frames -> 7 latents", video_latents_for_frames(22) == 7)
    rows17 = encode_video_rows(video_vae, hist, (1, 2, 2), sample=True)
    check("history rows shape", rows17.shape == (5 * 4, 32), str(rows17.shape))
    again = encode_video_rows(video_vae, hist, (1, 2, 2), sample=True)
    check("history encode is seeded (repeatable)", bool(mx.array_equal(rows17, again).item()))


def test_skip_blocks() -> None:
    from minimax_h3_mlx.adaln import ModulationCache

    dit = tiny_dit()
    cache = ModulationCache.build(dit, mx.array([0.2, 0.999, 1.0]), dtype=mx.float32)
    picked = gate_ranked_skip_blocks(cache, 3, 1)
    check("only an unprotected block can be skipped", picked == [] or picked == [], str(picked))
    check("explicit protected block refused", _raises(lambda: gs.resolve_skip_blocks("0", cache, 50)))
    check("auto spec parses", gs.resolve_skip_blocks("auto:0", cache, 3) == [])


def test_seams() -> None:
    prev = np.full((5, 4, 4, 3), 100, np.uint8)
    new = np.full((30, 4, 4, 3), 110, np.uint8)
    eased = gs.seam_colour_match(prev[-1], new, 24)
    check("colour easing lands on the previous frame at the seam", int(eased[0, 0, 0, 0]) == 100)
    check("and fades back to the window's own grade", int(eased[25, 0, 0, 0]) == 110)
    check("0 frames is a no-op", gs.seam_colour_match(prev[-1], new, 0) is new)
    # A source clip plus one new window: the stitch drops the hand-over frame once.
    src = (np.zeros((73, 4, 4, 3), np.uint8), np.zeros((2, gs.frame_samples(73)), np.float32))
    win = (np.ones((124, 4, 4, 3), np.uint8), np.zeros((2, gs.frame_samples(124)), np.float32))
    video, audio, seams = gs.stitch_windows([src, win], 32000, 1 / 24, colour_match_frames=24)
    check("extend stitch = source + window - 1", len(video) == 73 + 123 and len(seams) == 1)
    check("drive slice pads with silence", gs.drive_slice(np.ones((2, 100), np.float32), 0, 24).shape
          == (2, gs.frame_samples(24)))


def _raises(fn) -> bool:
    try:
        fn()
        return False
    except ValueError:
        return True


def main() -> int:
    mx.set_default_device(mx.cpu)
    with tempfile.TemporaryDirectory() as tmp:
        test_packing_without_history_is_unchanged()
        test_packing_with_history()
        test_frozen_timesteps()
        test_keyframe_routing()
        test_alignment_line()
        test_flag_refusals(Path(tmp))
        test_skip_blocks()
        test_seams()
        test_end_to_end()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {FAILURES}")
        return 1
    print("ALL MEDIA-CONDITIONING CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
