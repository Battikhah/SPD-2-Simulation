"""Frozen pretrained clip embeddings for the E-* runs (handoff 7.4), one vector per development clip, in the
row order of outputs/mil/cache/clips.csv. Frame-level embeddings are max-pooled over time.

  panns: PANNs CNN14 (AudioSet, 32 kHz; Zenodo record 3987831). Frame features = conv block 6 output averaged
         over frequency (2048 x T), max over T.
  beats: BEATs iter3+ AS2M pre-trained (16 kHz; microsoft/unilm code; checkpoint SHA-256 d43cbfad...8d34,
         identical on two Hugging Face mirrors because the official OneDrive link refuses scripted download).
         Patch tokens (768 x T), max over tokens.
  ast:   MIT/ast-finetuned-audioset-10-10-0.4593 (16 kHz, 10.24 s input). The 20 s clip is cut into two 10 s
         halves; patch tokens of both halves (CLS/distillation tokens excluded), max over tokens.

Input: the raw first 20 s of each 8 kHz clip (zero-padded), resampled with scipy resample_poly. Writes
outputs/mil/cache/emb_<name>.npy (float32). Sealed groups are not in the index, so they are never read.

    $env:PYTHONPATH = "<torch-cu121>;<outputs/mil/pretrained/beats>"; .venv/Scripts/python.exe mil/build_embeddings.py panns beats ast
"""

import importlib.util
import sys
import time
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

PRE = C.OUT / "pretrained"


def audio(rel, rate):
    import soundfile as sf
    from scipy.signal import resample_poly

    x, sr = sf.read(C.ROOT / rel, dtype="float32", always_2d=True)
    assert sr == C.SR
    x = x.mean(1)[: C.N_SAMPLES]
    x = np.pad(x, (0, C.N_SAMPLES - len(x)))
    return resample_poly(x, rate // C.SR, 1).astype(np.float32)


def panns_model(torch):
    pkg_dir = Path(importlib.util.find_spec("torchlibrosa").origin).parent.parent / "panns_inference"
    pkg = types.ModuleType("panns_inference")  # skip the package __init__, which downloads a labels file
    pkg.__path__ = [str(pkg_dir)]
    sys.modules["panns_inference"] = pkg
    from panns_inference.models import Cnn14

    m = Cnn14(sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64, fmin=50, fmax=14000, classes_num=527)
    ck = torch.load(PRE / "panns" / "Cnn14_mAP=0.431.pth", map_location="cpu", weights_only=True)
    m.load_state_dict(ck["model"])
    m.eval().cuda()

    def f(batch):
        x = m.spectrogram_extractor(batch)
        x = m.logmel_extractor(x)
        x = x.transpose(1, 3)
        x = m.bn0(x)
        x = x.transpose(1, 3)
        for blk, pool in ((m.conv_block1, 2), (m.conv_block2, 2), (m.conv_block3, 2), (m.conv_block4, 2),
                          (m.conv_block5, 2), (m.conv_block6, 1)):
            x = blk(x, pool_size=(pool, pool), pool_type="avg")
        return x.mean(dim=3).amax(dim=2)  # (B, 2048)

    return f, 32000, 8


def beats_model(torch):
    from BEATs import BEATs, BEATsConfig

    ck = torch.load(PRE / "beats" / "BEATs_iter3_plus_AS2M.pt", map_location="cpu", weights_only=True)
    m = BEATs(BEATsConfig(ck["cfg"]))
    m.load_state_dict(ck["model"])
    m.eval().cuda()

    def f(batch):
        rep = m.extract_features(batch, padding_mask=torch.zeros_like(batch).bool())[0]
        return rep.amax(dim=1)

    return f, 16000, 4


def ast_model(torch):
    from transformers import ASTFeatureExtractor, ASTModel

    name = "MIT/ast-finetuned-audioset-10-10-0.4593"
    fe = ASTFeatureExtractor.from_pretrained(name)
    m = ASTModel.from_pretrained(name).eval().cuda()
    half = 16000 * 10

    def f(batch):
        wav = batch.cpu().numpy()
        segs = [w[i * half:(i + 1) * half] for w in wav for i in range(2)]
        inp = fe(segs, sampling_rate=16000, return_tensors="pt")["input_values"].cuda()
        tok = m(input_values=inp).last_hidden_state[:, 2:]  # patch tokens
        return tok.amax(dim=1).view(len(wav), 2, -1).amax(dim=1)

    return f, 16000, 8


def main():
    import torch

    names = sys.argv[1:] or ["panns", "beats", "ast"]
    index = C.load_index()
    for name in names:
        out = C.CACHE / f"emb_{name}.npy"
        if out.exists():
            print(f"{name}: exists, skipping", flush=True)
            continue
        f, rate, bs = {"panns": panns_model, "beats": beats_model, "ast": ast_model}[name](torch)
        embs, t0 = [], time.time()
        with torch.no_grad():
            for i in range(0, len(index), bs):
                batch = torch.from_numpy(np.stack([audio(p, rate) for p in index.path[i:i + bs]])).cuda()
                embs.append(f(batch).float().cpu().numpy())
                if i % (bs * 50) == 0:
                    print(f"{name} {i}/{len(index)} ({time.time() - t0:.0f}s)", flush=True)
        e = np.concatenate(embs)
        assert e.shape[0] == len(index) and np.isfinite(e).all(), e.shape
        np.save(out, e.astype(np.float32))
        print(f"{name}: {e.shape} in {time.time() - t0:.0f}s", flush=True)
        del f
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
