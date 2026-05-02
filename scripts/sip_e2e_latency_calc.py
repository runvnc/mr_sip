#!/usr/bin/env python3
"""
SIP E2E Latency Calculator

Parses /tmp/sip_e2e_latency.log and computes user-perceived latency:
  VAD_EAGER_END (user stopped speaking) -> FIRST_RTP_SENT (first audio packet on wire)

Also breaks down each pipeline segment for analysis.

Usage:
  python3 /tmp/sip_e2e_latency_calc.py              # parse entire log
  python3 /tmp/sip_e2e_latency_calc.py --tail 20     # last 20 utterances
  python3 /tmp/sip_e2e_latency_calc.py --watch        # follow log in real-time
"""
import re
import sys
import time
import argparse
from collections import defaultdict

LOG_FILE = '/tmp/sip_e2e_latency.log'

# Parse format: [2026-05-01 22:00:00.123] [E2E] EVENT perf_counter=1234.567890 utterance=N key=val ...
LINE_RE = re.compile(
    r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})\] '
    r'\[E2E\] (\w+) '
    r'perf_counter=([\d.]+)'
    r'(?: utterance=(\d+))?'  # utterance is optional (PySIP may omit or use utterance_num)
    r'(.*)'
)
KV_RE = re.compile(r'(\w+)=([^\s]+)')


def parse_log(lines):
    """Parse log lines into structured events."""
    events = []
    for line in lines:
        m = LINE_RE.match(line.strip())
        if not m:
            continue
        wall_ts, event, perf_counter, utterance, rest = m.groups()
        kv = dict(KV_RE.findall(rest))
        # Handle utterance_num= from PySIP as fallback for utterance=
        if utterance is None:
            utterance = kv.pop('utterance_num', '0')
        events.append({
            'wall_ts': wall_ts,
            'event': event,
            'perf_counter': float(perf_counter),
            'utterance': int(utterance),
            **kv,
        })
    return events


def compute_latencies(events):
    """Group events by utterance number and compute latencies."""
    by_utterance = defaultdict(dict)
    for e in events:
        by_utterance[e['utterance']][e['event']] = e

    results = []
    for utt_num in sorted(by_utterance.keys()):
        evts = by_utterance[utt_num]
        r = {'utterance': utt_num}

        # Primary metric: VAD_EAGER_END -> FIRST_RTP_SENT
        if 'VAD_EAGER_END' in evts and 'FIRST_RTP_SENT' in evts:
            vad_pc = evts['VAD_EAGER_END']['perf_counter']
            rtp_pc = evts['FIRST_RTP_SENT']['perf_counter']
            r['e2e_ms'] = (rtp_pc - vad_pc) * 1000
        else:
            r['e2e_ms'] = None

        # Segment breakdowns
        segments = [
            ('vad_to_transcribe_ms', 'VAD_EAGER_END', 'TRANSCRIBE_DONE'),
            ('transcribe_to_eager_cb_ms', 'TRANSCRIBE_DONE', 'EAGER_EOT_CALLBACK'),
            ('eager_cb_to_utterance_cb_ms', 'EAGER_EOT_CALLBACK', 'UTTERANCE_CALLBACK'),
            ('utterance_cb_to_tts_start_ms', 'UTTERANCE_CALLBACK', 'TTS_RESPONSE_START'),
            ('tts_start_to_chunk_queued_ms', 'TTS_RESPONSE_START', 'FIRST_CHUNK_QUEUED'),
            ('chunk_queued_to_dequeued_ms', 'FIRST_CHUNK_QUEUED', 'FIRST_CHUNK_DEQUEUED'),
            ('chunk_dequeued_to_pysip_ms', 'FIRST_CHUNK_DEQUEUED', 'FIRST_TTS_CHUNK_PYSIP'),
            ('pysip_to_rtp_sent_ms', 'FIRST_TTS_CHUNK_PYSIP', 'FIRST_RTP_SENT'),
        ]
        for seg_name, start_event, end_event in segments:
            if start_event in evts and end_event in evts:
                r[seg_name] = (evts[end_event]['perf_counter'] - evts[start_event]['perf_counter']) * 1000
            else:
                r[seg_name] = None

        # Extra context
        if 'VAD_EAGER_END' in evts:
            r['wall_ts'] = evts['VAD_EAGER_END']['wall_ts']
        if 'TRANSCRIBE_DONE' in evts and 'transcribe_ms' in evts['TRANSCRIBE_DONE']:
            r['transcribe_ms'] = evts['TRANSCRIBE_DONE']['transcribe_ms']
        if 'FIRST_RTP_SENT' in evts and 'prebuffer_frames' in evts['FIRST_RTP_SENT']:
            r['prebuffer_frames'] = evts['FIRST_RTP_SENT']['prebuffer_frames']

        # Use pre-computed E2E_LATENCY if available (from PySIP rtp_handler)
        if 'E2E_LATENCY' in evts and 'e2e_ms' in evts['E2E_LATENCY']:
            r['e2e_ms'] = float(evts['E2E_LATENCY']['e2e_ms'])
        # User-perceived e2e (from last speech audio, not VAD decision)
        if 'E2E_LATENCY' in evts and 'user_e2e_ms' in evts['E2E_LATENCY']:
            r['user_e2e_ms'] = float(evts['E2E_LATENCY']['user_e2e_ms'])
        elif 'VAD_EAGER_END' in evts and 'last_speech_audio_pc' in evts['VAD_EAGER_END'] and 'FIRST_RTP_SENT' in evts:
            r['user_e2e_ms'] = (evts['FIRST_RTP_SENT']['perf_counter'] - float(evts['VAD_EAGER_END']['last_speech_audio_pc'])) * 1000

        results.append(r)
    return results


def print_results(results, show_all=False):
    if not results:
        print('No utterance pairs found in log.')
        return

    e2e_values = [r['e2e_ms'] for r in results if r.get('e2e_ms') is not None]
    user_e2e_values = [r['user_e2e_ms'] for r in results if r.get('user_e2e_ms') is not None]

    print(f'{'='*80}')
    print(f'SIP E2E Latency Report')
    print(f'{'='*80}')
    print()

    if e2e_values:
        avg = sum(e2e_values) / len(e2e_values)
        mn = min(e2e_values)
        mx = max(e2e_values)
        print(f'  Utterances with full e2e: {len(e2e_values)}')
        print(f'  Average e2e latency:     {avg:.0f} ms')
        print(f'  Min e2e latency:         {mn:.0f} ms')
        print(f'  Max e2e latency:         {mx:.0f} ms')
        print()

    if user_e2e_values:
        avg_u = sum(user_e2e_values) / len(user_e2e_values)
        mn_u = min(user_e2e_values)
        mx_u = max(user_e2e_values)
        print(f'  User-perceived e2e (last speech audio -> first RTP):')
        print(f'    Average: {avg_u:.0f} ms  Min: {mn_u:.0f} ms  Max: {mx_u:.0f} ms')
        print()

    # Per-utterance table
    header = f"{'Utt':>4} {'Wall Time':>20} {'UserE2E':>7} {'E2E(ms)':>8} {'VAD->X':>7} {'X->CB':>7} {'CB->UT':>7} {'UT->TTS':>8} {'TTS->Q':>7} {'Q->DQ':>7} {'DQ->PS':>7} {'PS->RTP':>8} {'Prebuf':>6}"
    print(header)
    print('-' * len(header))

    for r in results:
        e2e = f"{r['e2e_ms']:.0f}" if r.get('e2e_ms') is not None else '-'
        user_e2e = f"{r['user_e2e_ms']:.0f}" if r.get('user_e2e_ms') is not None else '-'
        wall = r.get('wall_ts', '-')
        segs = []
        seg_keys = ['vad_to_transcribe_ms', 'transcribe_to_eager_cb_ms',
                     'eager_cb_to_utterance_cb_ms', 'utterance_cb_to_tts_start_ms',
                     'tts_start_to_chunk_queued_ms', 'chunk_queued_to_dequeued_ms',
                     'chunk_dequeued_to_pysip_ms', 'pysip_to_rtp_sent_ms']
        for k in seg_keys:
            v = r.get(k)
            segs.append(f"{v:.0f}" if v is not None else '-')
        prebuf = r.get('prebuffer_frames', '-')
        print(f"{r['utterance']:>4} {wall:>20} {user_e2e:>7} {e2e:>8} {segs[0]:>7} {segs[1]:>7} {segs[2]:>7} {segs[3]:>8} {segs[4]:>7} {segs[5]:>7} {segs[6]:>7} {segs[7]:>8} {prebuf:>6}")

    print()
    print('Column legend:')
    print('  VAD->X  = VAD eager end -> Transcription done')
    print('  X->CB   = Transcription done -> Eager EOT callback')
    print('  CB->UT  = Eager EOT callback -> Utterance callback (agent input)')
    print('  UT->TTS = Utterance callback -> TTS response start')
    print('  TTS->Q  = TTS response start -> First chunk queued in SIPSession')
    print('  Q->DQ   = First chunk queued -> First chunk dequeued')
    print('  DQ->PS  = First chunk dequeued -> First chunk in PySIP AudioStream')
    print('  PS->RTP = First chunk in PySIP -> First RTP packet on wire (includes prebuffer)')
    print('  Prebuf  = PySIP outgoing prebuffer frames (each 20ms)')

    # Also print any E2E_LATENCY lines from PySIP (pre-computed)
    e2e_auto = [(r['utterance'], r.get('user_e2e_ms', r.get('e2e_ms')), r.get('e2e_ms')) for r in results if r.get('e2e_ms') is not None]
    if e2e_auto:
        print(f'\nAuto-computed E2E_LATENCY events from PySIP:')
        for utt, user_ms, vad_ms in e2e_auto:
            print(f'  Utterance #{utt}: user_e2e={user_ms:.0f}ms (vad_e2e={vad_ms:.0f}ms)')
def main():
    parser = argparse.ArgumentParser(description='SIP E2E Latency Calculator')
    parser.add_argument('--tail', type=int, default=None, help='Show last N utterances')
    parser.add_argument('--watch', action='store_true', help='Follow log in real-time')
    args = parser.parse_args()

    if args.watch:
        print('Watching /tmp/sip_e2e_latency.log (Ctrl+C to stop)...')
        try:
            with open(LOG_FILE, 'r') as f:
                f.seek(0, 2)  # seek to end
                while True:
                    lines = []
                    while True:
                        line = f.readline()
                        if not line:
                            break
                        lines.append(line)
                    if lines:
                        events = parse_log(lines)
                        for e in events:
                            if e['event'] == 'E2E_LATENCY':
                                # Pre-computed e2e from PySIP - print immediately
                                print(f"  UTT#{e['utterance']}: e2e={e.get('e2e_ms', '?')}ms (pre-computed)")
                            elif e['event'] == 'FIRST_RTP_SENT':
                                # Find matching VAD_EAGER_END
                                # Read full log to compute
                                with open(LOG_FILE, 'r') as f2:
                                    all_events = parse_log(f2.readlines())
                                results = compute_latencies(all_events)
                                if results:
                                    last = results[-1]
                                    if last.get('e2e_ms') is not None:
                                        print(f"  UTT#{last['utterance']}: e2e={last['e2e_ms']:.0f}ms "
                                              f"(VAD->X={last.get('vad_to_transcribe_ms',0):.0f} "
                                              f"X->CB={last.get('transcribe_to_eager_cb_ms',0):.0f} "
                                              f"CB->UT={last.get('eager_cb_to_utterance_cb_ms',0):.0f} "
                                              f"UT->TTS={last.get('utterance_cb_to_tts_start_ms',0):.0f} "
                                              f"TTS->RTP={last.get('pysip_to_rtp_sent_ms',0):.0f})")
                    else:
                        time.sleep(0.5)
        except KeyboardInterrupt:
            print('\nStopped.')
        return

    try:
        with open(LOG_FILE, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f'Log file not found: {LOG_FILE}')
        return

    events = parse_log(lines)
    results = compute_latencies(events)

    if args.tail and len(results) > args.tail:
        results = results[-args.tail:]

    print_results(results)


if __name__ == '__main__':
    main()
