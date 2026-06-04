"""Parallel execution helpers for container environments without multiprocessing semaphores."""

import os
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool, RawArray, cpu_count

thread_count = max(cpu_count() - 1, 1)
fork_batch_count = 100 * cpu_count()
_mp_available = None


def _parallel_backend():
    return os.environ.get("SAS_PARALLEL_BACKEND", "auto").lower()


def _multiprocessing_available():
    global _mp_available
    if _mp_available is not None:
        return _mp_available
    try:
        from multiprocessing import Lock
        Lock()
        _mp_available = True
    except PermissionError:
        _mp_available = False
    return _mp_available


def _use_process_pool():
    backend = _parallel_backend()
    if backend == "process":
        return True
    if backend in ("fork", "thread", "threads"):
        return False
    return _multiprocessing_available()


def _use_fork_pool():
    backend = _parallel_backend()
    if backend == "fork":
        return True
    if backend in ("thread", "threads", "process"):
        return False
    return not _multiprocessing_available() and hasattr(os, "fork")


def _fork_map(func, items, workers, progress_bar=None):
    n = len(items)
    if n == 0:
        return []
    workers = min(workers, n)
    results = RawArray("d", n)
    progress = RawArray("i", workers)
    chunk = (n + workers - 1) // workers
    pids = []
    slot = 0
    for w in range(workers):
        start = w * chunk
        end = min(start + chunk, n)
        if start >= end:
            continue
        worker_slot = slot
        pid = os.fork()
        if pid == 0:
            for i in range(start, end):
                results[i] = func(items[i])
                progress[worker_slot] += 1
            os._exit(0)
        pids.append(pid)
        slot += 1
    active_slots = slot

    refresh_step = max(1, n // fork_batch_count)
    last_shown = 0
    alive = set(pids)
    while alive:
        done = sum(progress[i] for i in range(active_slots))
        if progress_bar is not None and (done - last_shown >= refresh_step or done == n):
            progress_bar.n = done
            progress_bar.refresh()
            last_shown = done
        for pid in list(alive):
            reaped, status = os.waitpid(pid, os.WNOHANG)
            if reaped == 0:
                continue
            alive.discard(pid)
            if os.WIFEXITED(status) and os.WEXITSTATUS(status) != 0:
                raise RuntimeError(f"worker {pid} exited with status {os.WEXITSTATUS(status)}")
            if os.WIFSIGNALED(status):
                raise RuntimeError(f"worker {pid} killed by signal {os.WTERMSIG(status)}")
        if alive:
            time.sleep(0.05)

    return [results[i] for i in range(n)]


def parallel_imap(func, iterable, workers=None, progress_bar=None):
    workers = workers or thread_count
    items = list(iterable)
    if not items:
        return
    if _use_process_pool():
        pool = Pool(workers)
        try:
            yield from pool.imap(func, items)
        finally:
            pool.close()
            pool.join()
        return
    if _use_fork_pool():
        yield from _fork_map(func, items, workers, progress_bar=progress_bar)
        return
    with ThreadPoolExecutor(max_workers=workers) as ex:
        yield from ex.map(func, items)


def parallel_map(func, iterable, workers=None, progress_bar=None):
    workers = workers or thread_count
    items = list(iterable)
    if not items:
        return []
    if _use_process_pool():
        with Pool(workers) as pool:
            return pool.map(func, items)
    if _use_fork_pool():
        return _fork_map(func, items, workers, progress_bar=progress_bar)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(func, items))
