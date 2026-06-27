"""
Offline verification of the combined BargeInGate (Silero VAD + level heuristic).

Runs the REAL shared decision logic (mr_sip/stt/bargein_gate.py) with ONNX Silero
(same weights as the deployed JIT; only runtime differs). Audio is ulaw-roundtripped
per 32ms chunk to match the SIP wire.

Runs the gate on THREE inputs and writes a foreground + background clip for each
(6 clips total), so each can be auditioned:

  full     -> full_fore.wav     full_back.wav      (real call: fg+bg)
  foresep  -> foresep_fore.wav  foresep_back.wav   (level-foreground-only input)
  backsep  -> backsep_fore.wav  backsep_back.wav   (level-background-only input)

Foreground clips are PADDED (PAD_PRE/POST ms) so onsets/offsets are not truncated,
mirroring the live STT which pre-rolls ~500ms and runs through turn-end before
sending audio to Cohere. Background clips keep everything NOT originally foreground.
"""
import os, sys, audioop, math
import numpy as np
import soundfile as sf
import onnxruntime as ort

import importlib.util as _ilu
_GATE_PATH = os.path.join(os.path.dirname(__file__), '..', 'src', 'mr_sip', 'stt', 'bargein_gate.py')
_spec = _ilu.spec_from_file_location('bargein_gate', _GATE_PATH)
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
BargeInGate = _mod.BargeInGate

REC = '/files/upd6/mr_verification_dashboard/recordings'
FULL = REC + '/hannah_farend_from2_46_5.wav'
FG_ONLY = REC + '/sep_LEVEL_foreground.wav'
BG_ONLY = REC + '/sep_LEVEL_background.wav'
ONNX = '/xfiles/localmr/.venv/lib/python3.12/site-packages/silero_vad/data/silero_vad.onnx'
FR = 256
PAD_PRE_MS = 300
PAD_POST_MS = 200


def run_gate(wav, **kw):
    d, sr = sf.read(wav, dtype='int16', always_2d=True)
    x = d[:, 0]
    sess = ort.InferenceSession(ONNX, providers=['CPUExecutionProvider'])
    state = np.zeros((2, 1, 128), np.float32); sri = np.array(8000, np.int64)
    gate = BargeInGate(**kw)
    nf = len(x) // FR
    labels = []
    for i in range(nf):
        raw = x[i*FR:(i+1)*FR].tobytes()
        pcm = audioop.ulaw2lin(audioop.lin2ulaw(raw, 2), 2)
        af = np.frombuffer(pcm, np.int16).astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(af**2)))
        o, state = sess.run(None, {'input': af.reshape(1, -1), 'state': state, 'sr': sri})
        labels.append(gate.process(float(o[0][0]), rms)['label'])
    return x, sr, nf, labels


def merged(labels, want):
    segs = []; i = 0; n = len(labels)
    while i < n:
        if labels[i] == want:
            j = i
            while j < n and labels[j] == want:
                j += 1
            segs.append((i*FR/8000, j*FR/8000)); i = j
        else:
            i += 1
    return segs


def counts(labels):
    from collections import Counter
    c = Counter(labels)
    return f"FG={c.get('FG',0)} BG={c.get('BG',0)} NS={c.get('NS',0)}"


def write_fore_back(x, sr, nf, labels, fore_path, back_path):
    fg = np.array([l == 'FG' for l in labels], bool)
    pre = PAD_PRE_MS // 32; post = PAD_POST_MS // 32
    padded = fg.copy()
    idx = np.where(fg)[0]
    for i in idx:
        a = max(0, i - pre); b = min(nf, i + post + 1)
        padded[a:b] = True
    def expand(m):
        s = np.repeat(m, FR)
        if len(s) < len(x): s = np.concatenate([s, np.zeros(len(x)-len(s), bool)])
        return s[:len(x)]
    sm_fore = expand(padded)
    sm_back = expand(fg)            # unpadded -> background keeps all non-FG audio
    fore = x.copy(); fore[~sm_fore] = 0
    back = x.copy(); back[sm_back] = 0
    sf.write(fore_path, fore.astype(np.int16), sr, subtype='PCM_16')
    sf.write(back_path, back.astype(np.int16), sr, subtype='PCM_16')


for tag, path in [('full', FULL), ('foresep', FG_ONLY), ('backsep', BG_ONLY)]:
    x, sr, nf, labels = run_gate(path)
    fore_p = f'{REC}/{tag}_fore.wav'; back_p = f'{REC}/{tag}_back.wav'
    write_fore_back(x, sr, nf, labels, fore_p, back_p)
    print(f'[{tag}] {counts(labels)}')
    print(f'    FG segments: {[f"{a:.2f}-{b:.2f}" for a,b in merged(labels,"FG")]}')
    print(f'    -> {tag}_fore.wav , {tag}_back.wav')
