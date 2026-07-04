#!/usr/bin/env python3
"""
SIP E2E Latency Calculator

Parses /tmp/sip_e2e_latency.log and computes user-perceived latency:
  VAD_EAGER_END (user stopped speaking) -> FIRST_RTP_SENT (first audio packet on wire)

Also breaks down each pipeline segment for analysis.

Now supports session IDs for concurrent-call isolation and concurrency analysis.

Usage:
  python3 /tmp/sip_e2e_latency_calc.py              # parse entire log
  python3 /tmp/sip_e2e_latency_calc.py --tail 20     # last 20 utterances
  python3 /tmp/sip_e2e_latency_calc.py --watch        # follow log in real-time
  python3 /tmp/sip_e2e_latency_calc.py --concurrency  # show concurrency analysis only
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
        # Extract session ID (added for concurrent-call isolation)
        session = kv.pop('session', None) or 'unknown'
        events.append({
            'wall_ts': wall_ts,
            'event': event,
            'perf_counter': float(perf_counter),
            'utterance': int(utterance),
            'session': session,
            **kv,
        })
    return events


def compute_session_spans(events):
    """Compute the time span [first_event, last_event] for each session.

    Returns dict: session_id -> (start_pc, end_pc)
    """
    spans = {}
    for e in events:
        s = e['session']
        pc = e['perf_counter']
        if s not in spans:
            spans[s] = [pc, pc]
        else:
            if pc < spans[s][0]:
                spans[s][0] = pc
            if pc > spans[s][1]:
                spans[s][1] = pc
    return {s: (v[0], v[1]) for s, v in spans.items()}


def count_concurrent(spans, t_start, t_end):
    """Count how many session spans overlap the interval [t_start, t_end]."""
    count = 0
    for s_start, s_end in spans.values():
        if s_start <= t_end and s_end >= t_start:
            count += 1
    return count


def compute_latencies(events):
    """Group events by (session, utterance) and compute latencies.

    Falls back to utterance-only grouping for old log entries without session=.
    """
    # Group by (session, utterance) tuple for proper concurrent-call isolation
    by_key = defaultdict(dict)
    for e in events:
        key = (e['session'], e['utterance'])
        # Keep FIRST occurrence of each event type per (session, utterance)
        if e['event'] not in by_key[key]:
            by_key[key][e['event']] = e

    # E2E_LATENCY events contain rtp_sent_pc which is equivalent to FIRST_RTP_SENT.
    # Use it as a synthetic FIRST_RTP_SENT when the real one is missing (it lacks utterance= field).
    for key, evts in by_key.items():
        if 'E2E_LATENCY' in evts and 'FIRST_RTP_SENT' not in evts:
            e2e = evts['E2E_LATENCY']
            if 'rtp_sent_pc' in e2e:
                evts['FIRST_RTP_SENT'] = {
                    'perf_counter': float(e2e['rtp_sent_pc']),
                    'wall_ts': e2e.get('wall_ts', ''),
                }

    # Compute session spans for concurrency analysis
    session_spans = compute_session_spans(events)

    results = []
    for key in sorted(by_key.keys()):
        session, utt_num = key
        evts = by_key[key]
        r = {'utterance': utt_num, 'session': session}

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
        # Kyutai streaming path segments (added for streaming TTS profiling)
        kyutai_segments = [
            ('utterance_cb_to_first_partial_ms', 'UTTERANCE_CALLBACK', 'LLM_FIRST_PARTIAL_SPEAK'),
            ('first_partial_to_first_text_delta_ms', 'LLM_FIRST_PARTIAL_SPEAK', 'KYUTAI_FIRST_TEXT_DELTA'),
            ('first_text_delta_to_first_audio_ms', 'KYUTAI_FIRST_TEXT_DELTA', 'KYUTAI_FIRST_AUDIO_FRAME'),
            ('first_audio_to_chunk_queued_ms', 'KYUTAI_FIRST_AUDIO_FRAME', 'FIRST_CHUNK_QUEUED'),
        ]
        segments.extend(kyutai_segments)
        for seg_name, start_event, end_event in segments:
            if start_event in evts and end_event in evts:
                r[seg_name] = (evts[end_event]['perf_counter'] - evts[start_event]['perf_counter']) * 1000
            else:
                r[seg_name] = None

        # Extra context
        if 'VAD_SPEECH_START' in evts:
            r['speech_start_ts'] = evts['VAD_SPEECH_START']['wall_ts']
        if 'VAD_SPEECH_START' in evts and 'VAD_EAGER_END' in evts:
            r['utterance_duration_ms'] = (evts['VAD_EAGER_END']['perf_counter'] - evts['VAD_SPEECH_START']['perf_counter']) * 1000
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

        # Concurrency: count how many sessions were active during this utterance's
        # e2e latency window (VAD_EAGER_END -> FIRST_RTP_SENT). If we don't have
        # both endpoints, use the full event span for this utterance.
        if 'VAD_EAGER_END' in evts and 'FIRST_RTP_SENT' in evts:
            t_start = evts['VAD_EAGER_END']['perf_counter']
            t_end = evts['FIRST_RTP_SENT']['perf_counter']
        else:
            pcs = [e['perf_counter'] for e in evts.values()]
            t_start = min(pcs) if pcs else 0
            t_end = max(pcs) if pcs else 0
        r['concurrent_calls'] = count_concurrent(session_spans, t_start, t_end)

        results.append(r)
        # Also compute LLM-to-speech latency (from first partial speak to first audio chunk)
        if 'LLM_FIRST_PARTIAL_SPEAK' in evts and 'KYUTAI_FIRST_AUDIO_FRAME' in evts:
            r['llm_to_first_audio_ms'] = (evts['KYUTAI_FIRST_AUDIO_FRAME']['perf_counter'] - evts['LLM_FIRST_PARTIAL_SPEAK']['perf_counter']) * 1000

    return results


def print_concurrency_analysis(results):
    """Print concurrency vs latency analysis."""
    print(f'{'='*80}')
    print(f'Concurrency Analysis')
    print(f'{'='*80}')
    print()

    # Group results by concurrent_calls count
    by_concurrency = defaultdict(list)
    for r in results:
        cc = r.get('concurrent_calls', 1)
        e2e = r.get('e2e_ms')
        user_e2e = r.get('user_e2e_ms')
        if e2e is not None:
            by_concurrency[cc].append(e2e)
        if user_e2e is not None:
            by_concurrency.setdefault(cc, []).append(user_e2e)

    if not by_concurrency:
        print('  No latency data available for concurrency analysis.')
        return

    print(f"{'Concurrent':>10} {'Samples':>8} {'Avg E2E':>10} {'Min E2E':>10} {'Max E2E':>10} {'P50':>10} {'P95':>10}")
    print('-' * 70)
    for cc in sorted(by_concurrency.keys()):
        vals = by_concurrency[cc]
        if not vals:
            continue
        avg = sum(vals) / len(vals)
        mn = min(vals)
        mx = max(vals)
        sorted_vals = sorted(vals)
        p50 = sorted_vals[len(sorted_vals) // 2]
        p95 = sorted_vals[int(len(sorted_vals) * 0.95)] if len(sorted_vals) >= 20 else mx
        print(f"{cc:>10} {len(vals):>8} {avg:>10.0f} {mn:>10.0f} {mx:>10.0f} {p50:>10.0f} {p95:>10.0f}")

    print()
    print('  This shows how e2e latency scales with the number of concurrent calls.')
    print('  If Avg E2E increases significantly at higher concurrency, the system')
    print('  is bottlenecking (GPU contention, queue depth, etc.).')

    # Also show unique sessions
    sessions = set(r.get('session', 'unknown') for r in results)
    print(f'\n  Unique sessions in log: {len(sessions)}')
    if len(sessions) <= 20:
        for s in sorted(sessions):
            count = sum(1 for r in results if r.get('session') == s)
            print(f'    {s}: {count} utterances')


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

    # Kyutai streaming metrics
    llm_to_audio_values = [r['llm_to_first_audio_ms'] for r in results if r.get('llm_to_first_audio_ms') is not None]
    if llm_to_audio_values:
        avg_la = sum(llm_to_audio_values) / len(llm_to_audio_values)
        print(f'  Kyutai streaming: LLM first partial -> first audio frame:')
        print(f'    Average: {avg_la:.0f} ms  Min: {min(llm_to_audio_values):.0f} ms  Max: {max(llm_to_audio_values):.0f} ms')
        print()

    if user_e2e_values:
        avg_u = sum(user_e2e_values) / len(user_e2e_values)
        mn_u = min(user_e2e_values)
        mx_u = max(user_e2e_values)
        print(f'  User-perceived e2e (last speech audio -> first RTP):')
        print(f'    Average: {avg_u:.0f} ms  Min: {mn_u:.0f} ms  Max: {mx_u:.0f} ms')
        print()

    # Per-utterance table
    header = f"{'Sess':>8} {'Utt':>4} {'Wall Time':>20} {'UserE2E':>7} {'E2E(ms)':>8} {'Conc':>4} {'VAD->X':>7} {'X->CB':>7} {'CB->UT':>7} {'UT->TTS':>8} {'TTS->Q':>7} {'Q->DQ':>7} {'DQ->PS':>7} {'PS->RTP':>8} {'Prebuf':>6}"
    print(header)
    print('-' * 130)

    for r in results:
        e2e = f"{r['e2e_ms']:.0f}" if r.get('e2e_ms') is not None else '-'
        user_e2e = f"{r['user_e2e_ms']:.0f}" if r.get('user_e2e_ms') is not None else '-'
        wall = r.get('wall_ts', '-')
        conc = r.get('concurrent_calls', '-')
        sess = r.get('session', 'unknown')
        # Truncate session ID for display
        if len(str(sess)) > 8:
            sess = str(sess)[:6] + '..'
        segs = []
        seg_keys = ['vad_to_transcribe_ms', 'transcribe_to_eager_cb_ms',
                     'eager_cb_to_utterance_cb_ms', 'utterance_cb_to_tts_start_ms',
                     'tts_start_to_chunk_queued_ms', 'chunk_queued_to_dequeued_ms',
                     'chunk_dequeued_to_pysip_ms', 'pysip_to_rtp_sent_ms']
        for k in seg_keys:
            v = r.get(k)
            segs.append(f"{v:.0f}" if v is not None else '-')
        prebuf = r.get('prebuffer_frames', '-')
        print(f"{str(sess):>8} {r['utterance']:>4} {wall:>20} {user_e2e:>7} {e2e:>8} {str(conc):>4} {segs[0]:>7} {segs[1]:>7} {segs[2]:>7} {segs[3]:>8} {segs[4]:>7} {segs[5]:>7} {segs[6]:>7} {segs[7]:>8} {str(prebuf):>6}")

    # Kyutai streaming breakdown table
    kyutai_results = [r for r in results if r.get('utterance_cb_to_first_partial_ms') is not None]
    if kyutai_results:
        print(f'\nKyutai Streaming Breakdown:')
        header_k = f"{'Sess':>8} {'Utt':>4} {'CB->LLM':>8} {'LLM->TXT':>8} {'TXT->AUD':>8} {'AUD->Q':>8} {'LLM->AUD':>8}"
        print(header_k)
        print('-' * len(header_k))
        for r in kyutai_results:
            sess = r.get('session', 'unknown')
            if len(str(sess)) > 8:
                sess = str(sess)[:6] + '..'
            print(f"{str(sess):>8} {r['utterance']:>4} {r.get('utterance_cb_to_first_partial_ms', 0):>8.0f} {r.get('first_partial_to_first_text_delta_ms', 0):>8.0f} {r.get('first_text_delta_to_first_audio_ms', 0):>8.0f} {r.get('first_audio_to_chunk_queued_ms', 0):>8.0f} {r.get('llm_to_first_audio_ms', 0):>8.0f}")
        print('  CB->LLM = Utterance callback -> First partial speak (LLM TTFS)')
        print('  LLM->TXT = First partial -> Kyutai text delta (routing)')
        print('  TXT->AUD = Kyutai text delta -> First audio frame (TTS TTFA)')
        print('  AUD->Q = First audio frame -> Chunk queued (audio routing)')
        print('  LLM->AUD = First partial -> First audio frame (total TTS pipeline)')

    print()
    print('Column legend:')
    print('  Sess    = Session ID (truncated)')
    print('  Conc    = Number of concurrent calls active during this utterance')
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
    e2e_auto = [(r['utterance'], r.get('session', '?'), r.get('user_e2e_ms', r.get('e2e_ms')), r.get('e2e_ms')) for r in results if r.get('e2e_ms') is not None]
    if e2e_auto:
        print(f'\nAuto-computed E2E_LATENCY events from PySIP:')
        for utt, sess, user_ms, vad_ms in e2e_auto:
            print(f'  [{sess}] Utterance #{utt}: user_e2e={user_ms:.0f}ms (vad_e2e={vad_ms:.0f}ms)')


def main():
    parser = argparse.ArgumentParser(description='SIP E2E Latency Calculator')
    parser.add_argument('--tail', type=int, default=None, help='Show last N utterances')
    parser.add_argument('--watch', action='store_true', help='Follow log in real-time')
    parser.add_argument('--concurrency', action='store_true', help='Show concurrency analysis only')
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
                                print(f"  [{e.get('session', '?')}] UTT#{e['utterance']}: e2e={e.get('e2e_ms', '?')}ms (pre-computed)")
                            elif e['event'] == 'FIRST_RTP_SENT':
                                # Find matching VAD_EAGER_END
                                # Read full log to compute
                                with open(LOG_FILE, 'r') as f2:
                                    all_events = parse_log(f2.readlines())
                                results = compute_latencies(all_events)
                                if results:
                                    last = results[-1]
                                    if last.get('e2e_ms') is not None:
                                        print(f"  [{last.get('session', '?')}] UTT#{last['utterance']}: e2e={last['e2e_ms']:.0f}ms "
                                              f"(VAD->X={last.get('vad_to_transcribe_ms',0):.0f} "
                                              f"X->CB={last.get('transcribe_to_eager_cb_ms',0):.0f} "
                                              f"CB->UT={last.get('eager_cb_to_utterance_cb_ms',0):.0f} "
                                              f"UT->TTS={last.get('utterance_cb_to_tts_start_ms',0):.0f} "
                                              f"TTS->RTP={last.get('pysip_to_rtp_sent_ms',0):.0f} "
                                              f"conc={last.get('concurrent_calls', '?')})")
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

    if args.concurrency:
        print_concurrency_analysis(results)
    else:
        print_results(results)
        print()
        print_concurrency_analysis(results)


if __name__ == '__main__':
    main()
