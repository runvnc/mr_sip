START_RAW
# Handover: SIP hangup detection + delegate_call_job completion

_Last updated: 2026-06-29. Context: mr_sip / PySIP / mindroot outbound-call hangup + job completion work._

## TL;DR of the current open problem
Outbound call flow via `delegate_call_job` **mostly works now**, but **occasionally a call does not return from `delegate_call_job`** (the parent VerifierAgent session shows `delegate_call_job(...)` as the last thing and the job is never recorded). Several fixes already shipped (below). The remaining suspect is **mr_job_queue** (worker-slot deadlock or job-status-never-transitions). Next step is a targeted log grep to see WHERE `delegate_call_job` stalls for a stuck run.

---

## Projects / paths
- **mr_sip**: `/xfiles/update_plugins/mr_sip` (src at `src/mr_sip/`). Repo: github runvnc/mr_sip.
- **PySIP**: `/files/PySIP` (code under `PySIP/`). Repo: github runvnc/PySIP.
- **mindroot** source: `/files/mindroot/src/mindroot`. Repo moved to github runvnc/mindroot (old remote `ah.git` still redirects on push).
- **localmr** (dev/test box) running mindroot from a SEPARATE copy: `/xfiles/localmr/.venv/lib/python3.12/site-packages/mindroot` (NOT a symlink to source — must patch both or reinstall).
- **H200** (RunPod, real stack) mindroot at `/app/.venv/lib/python3.12/site-packages/mindroot`. Logs at `/workspace/logs/mindroot.log` and `/workspace/logs/mindroot.err`. Diagnostic file `/tmp/sip_hangup.log`.
- mr_job_queue: `/xfiles/update_plugins/mr_job_queue/src/mr_job_queue`.
- Verification dashboard/container: `/files/upd6/mr_verification_dashboard/` (container runs LLM+TTS+STT on H200).

---

## What was SHIPPED (committed + pushed) and why

### PySIP — commit `818a757` (main)
1. **BUG B FIX (behavior)** `PySIP/sip_core.py` `SipDialogue.update_state` (~line 530): changed
   `elif message.method == "BYE" and (message.type == MESSAGE or message.status is OK):`
   to just `elif message.method == "BYE":`.
   Reason: an **inbound BYE request** from the callee (type REQUEST, status None) did NOT match the old condition, so the dialog stayed CONFIRMED; `stop()` then sent a **spurious second BYE** and timed out 5s (STOP_BYE_TIMEOUT). Now ANY BYE (inbound request OR 200-OK-to-our-BYE) -> TERMINATED. Independent of From/To direction logic.
2. **Diagnostics (log-only)** to `/tmp/sip_hangup.log` via `_hangup_log()` helper:
   - `sip_core.py`: `CORE_WIRE_BYE_RECEIVED` (a BYE seen off the socket in the receive loop).
   - `sip_call.py`: `CALL_BYE_CALLID_MISMATCH`, `CALL_REMOTE_BYE_BRANCH`, `CALL_REMOTE_BYE_STATE_ENDED`.

### mr_sip — commit `df78765` (sip_client_v2.py diag) + the commands.py latch commit
1. `sip_client_v2.py`: `_hangup_log()` markers `MR_SIP_ON_STATE`, `MR_SIP_ON_CALL_ENDED_*`, `MR_SIP_TERMINATE_CALL_*` (log-only). (Originally from an open-source-LLM agent; kept.)
2. `commands.py` `await_call_result` + `delegate_call_job`: **LATCH disconnect detection + hardened parsing**.
   - Old code only finished if the SAME 1s poll both saw `-- CALL DISCONNECTED --` AND `idle >= finish_timeout`. But `idle = time.time() - log.last_modified`, and `last_modified` = **file mtime** (`ChatLog.__init__` sets `self.last_modified = os.path.getmtime`). `_show_disconnected` ALSO calls `send_message_to_agent`, which makes the agent reply -> bumps mtime -> resets the finish timer -> fell through to the 120s idle timeout. => "takes forever to realize call is over".
   - Now: latch `disconnected_at = time.time()` on first sight; finish when wall-clock since >= `finish_timeout`.
   - Added `_message_text()` / `_log_has_disconnect()` that scan ALL content parts AND handle content stored as a plain **string** (old code only checked `content[0]['text']` and required a list). This mattered: the disconnect user message is stored as a **plain string** `"\n\nSYSTEM: -- CALL DISCONNECTED --\n\n"`, which the old check silently skipped.

### mindroot — v14.6.0, commit `540fb0f`, published to PyPI (https://pypi.org/project/mindroot/14.6.0/)
Root cause: `context.data['active_command_task']` holds a live `asyncio.Task` (set at `agent.py:445` and `agent.py:672`). `save_context()` does `json.dumps({'data': self.data, ...})` -> **`Object of type Task is not JSON serializable`**.
- Two call sites were changed to `await context.save_context()` (they were fire-and-forget before): `chat/commands.py:225` (`task_result`) and `chat/services.py` `cancel_and_wait`. On the OLD H200 code these were **un-awaited** -> coroutine never ran -> RuntimeWarning `coroutine 'save_context' was never awaited` AND context never actually persisted (so `finished_conversation`/`task_result` flags were lost).
- FIX: added `_json_safe_data()` in `lib/chatcontext.py` that strips non-JSON-serializable values from `context.data` before writing; used in BOTH `save_context()` and `save_context_data()`. So serialization now runs AND won't crash on the Task.
- Also bumped version 14.4.0 -> 14.5.0 -> **14.6.0** (14.6.0 is the one actually published to PyPI).

**Deploy to H200:** inside container venv `/app/.venv`:

    pip install -U "mindroot==14.6.0"   # then restart server
    python -c "import mindroot.lib.chatcontext as c; print(c.__file__); print('_json_safe_data' in dir(c))"  # expect True
    sed -n '224,226p' /app/.venv/lib/python3.12/site-packages/mindroot/coreplugins/chat/commands.py  # expect 'await context.save_context()'

If the container image pins/copies mindroot at build time, bump the pin to 14.6.0 and rebuild so it doesn't revert on restart.

---

## CURRENT STATUS (2026-06-29)
- 14.6.0 was installed on H200. **A few calls now work end-to-end.**
- **One call still did NOT return from `delegate_call_job`** (parent session ends with the `delegate_call_job` call; job not recorded). This is the OPEN ISSUE.
- Verified earlier via `/tmp/sip_hangup.log` that on a good run the BYE chain fires correctly: `CORE_WIRE_BYE_RECEIVED -> CALL_REMOTE_BYE_BRANCH -> MR_SIP_ON_STATE(ENDED) -> MR_SIP_ON_CALL_ENDED_ENTRY`, and the +20s gap before final terminate = the finish-timeout latch working as designed (NOT a hang).
- Confirmed XML/raw command mode is fine: `ChatLog.parsed_commands()` (chatlog.py:270) reads pre-parsed `message['commands']` (set by `agent.py:600` `replace_last_assistant(content, commands=list(collected))`); entries are `{name: args}` dicts so `'task_result' in cmd` / `'hangup' in cmd` work in XML mode. XML was NOT the cause.

---

## OPEN ISSUE: delegate_call_job sometimes doesn't return

### delegate_call_job flow (mr_sip/src/mr_sip/commands.py, ~line 522)
1. `service_manager.add_job(...)` -> queues the CALL agent as a separate job (mr_job_queue).
2. Queue-wait loop: poll `service_manager.get_job_data_service(queued_job_id)` until status in (active, completed, failed), up to `max_queue_wait = min(timeout, 420)` (~7 min). If never starts -> cancel_job + return a message.
3. Monitor loop (`while not finished`): breaks on max_call_length (~300s) / idle_timeout (120s, file-mtime based) / `task_result`|`hangup` in `parsed_commands()` / disconnect latch (`_log_has_disconnect` + finish_timeout).
4. Cleanup: `get_session`, `baresip_bot.hangup_call()`, `session_manager.end_session()`, `context.close_s2s_session()` (each in try/except).
5. `call_result = json.dumps(log.messages)`; return `f'Job ID: {queued_job_id}. Result: {call_result}'`.

### Diagnostic plan (run on H200)
Step 1 — find the stuck job id:

    grep -nE "Queued call job|Job ID:" /workspace/logs/mindroot.log | tail -20

Step 2 — full timeline for that id (LAST `Call job <ID> ...` line = where it stalled):

    grep -nE "Call job <ID>|Job ID: <ID>|Error in delegate_call_job" /workspace/logs/mindroot.log

Interpretation:
- Last line `Queued call job <ID>` with NO `Call job <ID> is now active` => stuck in **queue-wait** (worker never started it) => **mr_job_queue**.
- `is now active` present but no `idle timeout`/`finish timeout`/`received task_result/hangup`/`exceeded max call length` => stuck in **monitor loop** (shouldn't happen; has max_call_length ~300s backstop unless an awaited call inside hung).
- A monitor-exit line present but NO `Job ID: <ID>. Result:` => stuck in **cleanup** (`hangup_call`/`end_session`/`close_s2s_session`):

      grep -nE "Error hanging up call during cleanup|Error ending session during cleanup|Could not stop silence monitor|Could not close s2s session" /workspace/logs/mindroot.err

Step 3 — did the call itself end?

    grep -nE "CALL DISCONNECTED|MR_SIP_ON_CALL_ENDED_ENTRY|MR_SIP_TERMINATE" /tmp/sip_hangup.log | tail

Step 4 — mr_job_queue angle:

    grep -niE "job_queue|mr_job_queue|worker|concurren" /workspace/logs/mindroot.err | tail -40

Step 5 — confirm the mindroot fix is really live (should be True + awaited):

    python -c "import mindroot.lib.chatcontext as c; print(c.__file__); print('_json_safe_data' in dir(c))"
    grep -nC3 "Task is not JSON serializable" /workspace/logs/mindroot.err /workspace/logs/mindroot.log

### mr_job_queue hypotheses (the user's hunch — plausible)
1. **Worker-slot deadlock (most likely).** `delegate_call_job` queues the call agent as a job and then BLOCKS watching it. If the **VerifierAgent task is itself a queued job** and the queue has limited concurrency (e.g. 1–2 workers), the parent job holds a worker slot while waiting for the child call job, but the child can't get a slot -> never goes active -> parent waits out `max_queue_wait` (~7 min) then cancels. With multiple simultaneous verifications this manifests as "sometimes hangs." ACTION: check whether the VerifierAgent is dispatched via the job queue, and the queue's max worker/concurrency setting. If parent+child share one pool, that's the bug.
2. **Status never transitions.** If `get_job_data_service` keeps returning `queued`/`pending` (status not updated on worker pickup), the wait loop never sees active/completed/failed.

Note: given the monitor-loop backstops (max_call_length ~300s, idle_timeout 120s, queue-wait ~420s), a TRUE infinite hang most likely means an awaited call that never returns: `get_job_data_service`, `hangup_call`, `end_session`, or `close_s2s_session` — OR the mr_job_queue deadlock above (which caps at ~7 min then should cancel; if it's been longer, the poll/queue itself is wedged).

---

## DEFERRED / NOT done
- **BUG A (PySIP `SipMessage.is_from_client`, sip_core.py:875):** naive `str(uac_username) in From_header` substring match can misclassify the callee's BYE as our-own-loopback -> routes to the branch at sip_call.py:~937 that only sets ENDED (no 200 OK, no stop). Correct fix is tag-based (`msg.from_tag == dialogue.local_tag`). LEFT ALONE on purpose: `is_from_client` is load-bearing direction logic that ALSO governs our own outbound-hangup path and sits on top of recently-tuned From/To tag work (mis-tagged UAS BYE / Telnyx 403 / double-hangup, commit 4146e6a 'hangup bug', comment ~sip_call.py:965 UAC-only remote-tag reset). Fix it tag-based ONLY with a real `/tmp/sip_hangup.log` showing mis-routing (i.e. `CORE_WIRE_BYE_RECEIVED` present but `CALL_REMOTE_BYE_BRANCH` absent). Bug B alone may make A moot.
- Two still-un-awaited `save_context()` calls in `chat/commands.py:282` and `:317` (harmless now — don't persist — but should be awaited later; the guard makes it safe).

---

## Handy references
- Disconnect message injected by `mr_sip/src/mr_sip/sip_client_v2.py:1167` `_show_disconnected()` -> `backend_user_message` (role user, content is a plain string) + `send_message_to_agent`.
- Poller reads it in `commands.py` `await_call_result` (~356) and `delegate_call_job` (~522).
- `ChatLog` at `/files/mindroot/src/mindroot/lib/chatlog.py` (`__init__` sets last_modified = file mtime; `parsed_commands` at :270).
- `chatcontext.py` `_json_safe_data` guard + `save_context`/`save_context_data`.
END_RAW
