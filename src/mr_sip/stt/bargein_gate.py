"""
Combined barge-in / foreground-vs-background gate (shared, dependency-free).

Decides, per 32ms audio chunk, whether the chunk is near-end FOREGROUND speech
(should barge-in / be transcribed), quieter BACKGROUND speech/cross-talk (should
be ignored), or NON-SPEECH.

Deliberately free of torch / numpy / audio I/O so the SAME logic runs:
  - live in the SIP STT provider (fed Silero prob + raw RMS per chunk), and
  - offline in a test harness (fed ONNX-Silero prob + RMS per chunk).

Silero is primary; level is a second opinion:

  Path A - normal near-end speech (Silero-driven):
      voiced AND (no near-end reference yet OR level within REL_LEVEL_DB of the
      near-end reference). Fires immediately (no added latency); with no
      reference yet it accepts quiet speech (a quiet caller on a quiet line is
      still foreground - avoids the "quiet == background" reversion).

  Path B - loud-foreground rescue (level-driven):
      a SUSTAINED segment loud vs the running noise floor and not clearly-quieter
      than the near-end reference, even if Silero scored it non-speech. Recovers
      clipped/distorted near-end speech Silero misses (e.g. a hot, low-ZCR
      "Endeavor Health, how can I help you?"). Sustain rejects thuds/DTMF/clicks.
      Adds latency only on this rescue path, which today produces dead air, so it
      is strictly better.

Background = voiced speech clearly quieter than the near-end reference (only
possible once a louder near-end reference exists). Non-speech = neither.

The near-end reference adapts asymmetrically (fast up, slow down) so it locks to
the near-end speaker level and does NOT drift down to follow quiet background.
"""

import math
from collections import deque
from typing import Optional


def _db(v: float) -> float:
    return 20.0 * math.log10(v) if v > 1e-9 else -120.0


class BargeInGate:
    FG = "FG"   # near-end foreground speech
    BG = "BG"   # background speech / cross-talk (quieter than near-end)
    NS = "NS"   # non-speech (silence / ambient / transient)

    def __init__(
        self,
        vad_threshold: float = 0.5,
        frame_ms: int = 32,
        rel_level_db: float = 15.0,
        min_rms: float = 0.01,            # ~ -40 dBFS absolute floor
        ref_attack_alpha: float = 0.4,
        ref_decay_alpha: float = 0.05,
        onset_voiced_frames: int = 1,     # 1 = fire on first voiced frame (0 latency)
        level_window_ms: int = 160,       # backward peak window => no latency
        rescue_enabled: bool = True,
        rescue_snr_db: float = 12.0,      # dB above noise floor to count as "loud"
        rescue_sustain_ms: int = 160,     # must persist this long (rejects thuds)
        noise_alpha: float = 0.05,
        noise_init_rms: float = 1e-4,     # ~ -80 dBFS
        rescue_warmup_frames: int = 0,    # NS frames the noise floor must adapt
                                          # over before loud-rescue may fire.
                                          # 0 = disabled (legacy). Prevents the
                                          # cold-noise-floor false onset on the
                                          # call-answer click.
    ):
        self.vad_threshold = vad_threshold
        self.frame_ms = frame_ms
        self.rel_level_db = rel_level_db
        self.min_rms = min_rms
        self.ref_attack_alpha = ref_attack_alpha
        self.ref_decay_alpha = ref_decay_alpha
        self.onset_voiced_frames = max(1, onset_voiced_frames)
        self.level_window_ms = level_window_ms
        self.rescue_enabled = rescue_enabled
        self.rescue_snr_db = rescue_snr_db
        self.rescue_sustain_ms = rescue_sustain_ms
        self.noise_alpha = noise_alpha
        self.rescue_warmup_frames = max(0, rescue_warmup_frames)

        win = max(1, level_window_ms // frame_ms)
        self._rms_win = deque(maxlen=win)

        self.near_end_ref: Optional[float] = None
        self.noise_floor: float = noise_init_rms

        self._onset_voiced_count = 0
        self._rescue_run_frames = 0
        self._in_fg = False
        # Count of non-speech frames the noise floor has adapted over (gates the
        # loud-rescue warm-up).
        self._noise_frames = 0

    def _rel_db(self, lvl: float) -> Optional[float]:
        if self.near_end_ref is None or self.near_end_ref <= 0 or lvl <= 0:
            return None
        return 20.0 * math.log10(lvl / self.near_end_ref)

    def _update_near_end_ref(self, lvl: float) -> None:
        if lvl <= 0 or (self.min_rms > 0 and lvl < self.min_rms):
            return
        if self.near_end_ref is None:
            self.near_end_ref = lvl
        elif lvl >= self.near_end_ref:
            a = self.ref_attack_alpha
            self.near_end_ref = a * lvl + (1 - a) * self.near_end_ref
        else:
            if self.rel_level_db <= 0 or 20.0 * math.log10(lvl / self.near_end_ref) >= -self.rel_level_db:
                a = self.ref_decay_alpha
                self.near_end_ref = a * lvl + (1 - a) * self.near_end_ref

    def _update_noise_floor(self, rms: float) -> None:
        a = self.noise_alpha
        self.noise_floor = a * rms + (1 - a) * self.noise_floor
        if self.noise_floor < 1e-6:
            self.noise_floor = 1e-6
        self._noise_frames += 1

    def process(self, prob: float, rms: float) -> dict:
        """Feed one chunk. Returns dict(label, barge_in, rel_db, snr_db, reason)."""
        self._rms_win.append(rms)
        lvl = max(self._rms_win)              # backward window peak (no latency)
        voiced = prob >= self.vad_threshold
        floor_ok = (self.min_rms <= 0) or (lvl >= self.min_rms)
        rel_db = self._rel_db(lvl)
        snr_db = _db(lvl) - _db(self.noise_floor)

        within_band = (rel_db is None) or (rel_db >= -self.rel_level_db)

        # Path A: normal near-end speech via Silero.
        pathA = False
        if voiced and floor_ok:
            if self.near_end_ref is None or self.rel_level_db <= 0 or within_band:
                self._onset_voiced_count += 1
                if self._onset_voiced_count >= self.onset_voiced_frames:
                    pathA = True
            else:
                self._onset_voiced_count = 0   # voiced but clearly quieter -> background
        else:
            self._onset_voiced_count = 0

        # Path B: loud-foreground rescue.
        pathB = False
        # Warm-up gate: until the noise floor has actually been measured over
        # rescue_warmup_frames non-speech frames, do NOT allow loud-rescue. At
        # call answer the floor is cold (~-80 dBFS) so the first transient (the
        # pickup click) otherwise reads as huge SNR and trips a false onset.
        rescue_warmed = (self.rescue_warmup_frames <= 0
                         or self._noise_frames >= self.rescue_warmup_frames)
        loud = floor_ok and (snr_db >= self.rescue_snr_db) and within_band and rescue_warmed
        if self.rescue_enabled and loud:
            self._rescue_run_frames += 1
            if self._rescue_run_frames * self.frame_ms >= self.rescue_sustain_ms:
                pathB = True
        else:
            self._rescue_run_frames = 0

        is_fg = pathA or pathB
        if is_fg:
            label = self.FG
            reason = "silero" if pathA else "loud_rescue"
        elif voiced:
            label = self.BG
            reason = "quiet_vs_nearend"
        else:
            label = self.NS
            reason = "nonspeech"

        barge_in = is_fg and not self._in_fg
        self._in_fg = is_fg

        if label == self.FG:
            self._update_near_end_ref(lvl)
        elif label == self.NS:
            self._update_noise_floor(rms)

        return {
            "label": label,
            "barge_in": barge_in,
            "rel_db": rel_db,
            "snr_db": snr_db,
            "near_end_ref": self.near_end_ref,
            "noise_floor": self.noise_floor,
            "reason": reason,
        }
