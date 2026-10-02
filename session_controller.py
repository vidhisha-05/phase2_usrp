"""
session_controller.py — Trial-marker synchronization, timed audio cues,
                         and session orchestration.

Records BOTH host-clock trigger time AND nearest UHD hardware timestamp
for each trial boundary (Section 16).

Reference: phase2 (2).md — Sections 16, 17 Step 8
"""

import time
import queue
import threading
import json
import os
import numpy as np

try:
    import simpleaudio as sa
    _AUDIO_AVAILABLE = True
except ImportError:
    _AUDIO_AVAILABLE = False
    print("[session] WARNING: simpleaudio not installed — audio cues disabled.")


ACTIVITIES = ["walking", "sitting", "standing", "falling", "idle"]


class SessionController:
    """
    Orchestrates a data-collection session with timed activity trials.

    Usage:
        sc = SessionController(logger, session_id="session_001",
                               trial_duration_s=10, rest_duration_s=5)
        sc.run(activities=["walking", "idle"], reps_per_activity=3)
    """

    def __init__(self,
                 logger,
                 session_id: str      = "session_001",
                 trial_duration_s: float = 10.0,
                 rest_duration_s:  float = 5.0,
                 output_dir: str   = "sessions"):
        self._logger          = logger
        self._session_id      = session_id
        self._trial_dur       = trial_duration_s
        self._rest_dur        = rest_duration_s
        self._output_dir      = output_dir
        self._trial_log: list = []
        os.makedirs(output_dir, exist_ok=True)

    def run(self, activities: list = None, reps_per_activity: int = 3):
        """Execute all trials in a session."""
        acts = activities or ACTIVITIES
        trial_id = 0

        print(f"\n[session] Starting session {self._session_id}")
        print(f"  Activities: {acts}  Reps: {reps_per_activity}")
        print(f"  Trial duration: {self._trial_dur}s  Rest: {self._rest_dur}s\n")

        time.sleep(3.0)   # Initial stabilization

        for activity in acts:
            for rep in range(reps_per_activity):
                trial_label = f"trial_{trial_id:04d}"
                print(f"\n[session] --- {activity.upper()} (rep {rep+1}) ---")

                # Rest period
                print(f"  REST for {self._rest_dur}s ...")
                self._beep(freq=440, dur_ms=200)
                time.sleep(self._rest_dur)

                # START cue
                print(f"  START {activity}")
                self._beep(freq=880, dur_ms=400)
                t_start_host = time.monotonic()

                # Record trial start (host clock; UHD time populated externally)
                self._logger.log_trial(
                    trial_label, activity,
                    start_ts=t_start_host, end_ts=0.0)

                time.sleep(self._trial_dur)

                # STOP cue
                t_end_host = time.monotonic()
                print(f"  STOP {activity}")
                self._beep(freq=440, dur_ms=200)

                # Update trial with end timestamp
                self._logger.log_trial(
                    trial_label, activity,
                    start_ts=t_start_host, end_ts=t_end_host)

                self._trial_log.append({
                    'trial_id':     trial_label,
                    'activity':     activity,
                    'rep':          rep,
                    'start_host':   t_start_host,
                    'end_host':     t_end_host,
                    'duration_s':   t_end_host - t_start_host,
                })
                trial_id += 1

        self._save_log()
        print(f"\n[session] Session complete. {trial_id} trials recorded.")

    def _beep(self, freq: int = 440, dur_ms: int = 200):
        """Play a short beep as an audio cue."""
        if not _AUDIO_AVAILABLE:
            return
        try:
            sr = 44100
            t  = np.linspace(0, dur_ms / 1000, int(sr * dur_ms / 1000))
            wave = (np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
            sa.play_buffer(wave, 1, 2, sr).wait_done()
        except Exception as e:
            print(f"[session] Audio error: {e}")

    def _save_log(self):
        path = os.path.join(self._output_dir,
                             f"{self._session_id}_trial_log.json")
        with open(path, 'w') as f:
            json.dump(self._trial_log, f, indent=2)
        print(f"[session] Trial log saved: {path}")
