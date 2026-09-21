"""Bounded worker subprocesses; setup and model execution have separate deadlines."""
from __future__ import annotations

import subprocess
import time
from evaluation.provider import TrialControlStopped


class WorkerDeadlineExceeded(TimeoutError):
    stop_class = "harness_timeout"

    def __init__(self, stage, seconds):
        super().__init__(f"worker exceeded frozen {stage} deadline ({seconds} seconds)")
        self.stage = stage
        self.seconds = seconds


def run_worker(command, *, log, server, generation, env=None, cwd=None, model_completed=None):
    started = time.monotonic()
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=cwd)
    verifier_started = None
    try:
        while True:
            stop = getattr(server, "terminal_error", None)
            if stop is not None:
                raise TrialControlStopped(stop["category"], stop["error"])
            try:
                return process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                now = time.monotonic()
                first = server.first_model_request_at
                if model_completed is not None and verifier_started is None and model_completed():
                    verifier_started = now
                if first is None and now - started > generation["agent_setup_timeout_seconds"]:
                    raise WorkerDeadlineExceeded("setup", generation["agent_setup_timeout_seconds"])
                if first is not None and verifier_started is None and now - first > generation["wall_time_seconds"]:
                    raise WorkerDeadlineExceeded("model-execution", generation["wall_time_seconds"])
                if verifier_started is not None and now - verifier_started > generation["verifier_timeout_seconds"]:
                    raise WorkerDeadlineExceeded("verifier", generation["verifier_timeout_seconds"])
                if now - started > generation["trial_timeout_seconds"]:
                    raise WorkerDeadlineExceeded("total-trial", generation["trial_timeout_seconds"])
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)
