"""Real Ray/VERL transport regression; reuses the existing owner API fixtures."""
import ast
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
import os
from pathlib import Path
from threading import Condition

import pytest
import ray

import test_rank_completion_candidate as original
from test_rank_completion_candidate import cpu_ray


def owner_class(path):
    spec = importlib.util.spec_from_file_location('completion_batching_candidate', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.LoopOwner


@pytest.fixture(autouse=True)
def select_candidate(monkeypatch):
    path = Path(__file__).with_name('appworld-batch-coalescing-20261002') / 'loop_owner_rollout.py'
    monkeypatch.setattr(original, 'candidate', lambda: owner_class(path))


@pytest.mark.parametrize('name,args', [
    ('test_actual_owner_pool_preserves_exact_requests_and_replies', (64, None)),
    ('test_actual_owner_pool_preserves_exact_requests_and_replies', (63, None)),
    ('test_actual_owner_pool_preserves_exact_requests_and_replies', (64, 0)),
    ('test_actual_owner_pool_preserves_exact_requests_and_replies', (1, None)),
    ('test_fast_rank_can_advance_again_before_slow_rank_returns', ()),
    ('test_original_engine_error_is_not_hidden', ()),
    ('test_original_verl_empty_input_is_not_a_supported_rpc_contract', ()),
])
def test_unchanged_owner_transport_contract(name, args):
    getattr(original, name)(*args)


class AcceptedRequests(original.Requests):
    """Acknowledge each fixture arrival at the next blocking queue read.

    This supplies the same input schedule for old and new transport without
    sleeps, generated trajectories or relying on a fast host's race timing.
    """
    def __init__(self):
        super().__init__()
        self.condition = Condition()
        self.seen, self.accepted = set(), set()

    def event(self):
        with self.condition:
            self.accepted.update(self.seen)
            self.condition.notify_all()
        return self.observe(super().event())

    def queued_event(self):
        return self.observe(super().queued_event())

    def observe(self, event):
        if event[0] == 'completion':
            self.seen.add(event[2])
        return event

    def wait_accepted(self, key):
        with self.condition:
            assert self.condition.wait_for(lambda: key in self.accepted, timeout=15)


def test_busy_owner_requests_are_coalesced_without_changing_reply_artifacts(monkeypatch):
    paths = [Path(os.environ['DEPLOYED_LOOP_ENTRY']) / 'loop_owner_rollout.py',
             Path(__file__).with_name('appworld-batch-coalescing-20261002') / 'loop_owner_rollout.py']
    reports = []
    for path in paths:
        with monkeypatch.context() as selection:
            selection.setattr(original, 'candidate', lambda: owner_class(path))
            requests = AcceptedRequests()
            actors = [original.ReplyActor.remote(blocked=True), original.ReplyActor.remote(blocked=True)]
            for i in range(2):
                requests.request(i % 2, str(i), i + 1)
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(original.collect, requests, original.owner_group(actors), [])
                try:
                    ray.get([a.wait_started.remote() for a in actors], timeout=15)
                    for i in range(2, 34):
                        requests.request(i % 2, str(i), i + 1)
                        requests.wait_accepted(str(i))
                    ray.get([a.release.remote() for a in actors])
                    for rank, queue in enumerate(requests.replies):
                        expected = {str(i): i + 1 for i in range(rank, 34, 2)}
                        while expected:
                            key, reply = queue.get(timeout=15)
                            assert reply.token_ids == [expected.pop(key)]
                            assert reply.logprobs == [-.25]
                            assert reply.finish_reason == 'stop'
                        requests.finish(rank)
                    records = future.result(timeout=15)
                    assert set(records) == {str(i) for i in range(34)}
                    calls = ray.get([a.snapshot.remote() for a in actors])
                    assert sorted(token for rank_calls in calls for batch in rank_calls for token in batch) == list(range(1, 35))
                    reports.append(dict(source=str(path), batches=[len(batch) for rank_calls in calls for batch in rank_calls]))
                finally:
                    ray.get([a.release.remote() for a in actors])
                    requests.finish(0)
                    requests.finish(1)
                    for actor in actors:
                        ray.kill(actor)
    # Same 32 arrivals while both original actors are blocked: the deployed
    # bridge freezes 32 singleton RPCs; the patch retains two native B16 chunks.
    assert sorted(reports[0]['batches']) == [1] * 34
    assert sorted(reports[1]['batches']) == [1, 1, 16, 16]
    Path('busy-owner-batches.json').write_text(json.dumps(dict(
        scope='CPU exact-artifact and RPC-work-count regression; not model throughput or numerical tolerance',
        cases=reports), indent=2) + '\n')


def test_cancellation_covers_requests_held_while_native_actors_are_busy():
    requests = AcceptedRequests()
    actors = [original.ReplyActor.remote(blocked=True), original.ReplyActor.remote(blocked=True)]
    requests.request(0, 'first', 1)
    requests.request(1, 'second', 2)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(original.collect, requests, original.owner_group(actors), [])
        try:
            ray.get([a.wait_started.remote() for a in actors], timeout=15)
            requests.request(0, 'cancelled', 3)
            requests.wait_accepted('cancelled')
            requests.request(1, 'live', 4)
            requests.wait_accepted('live')
            requests.cancellations[0].set()
            ray.get([a.release.remote() for a in actors])
            assert requests.replies[1].get(timeout=15)[0] == 'second'
            assert requests.replies[1].get(timeout=15)[0] == 'live'
            requests.finish(0)
            requests.finish(1)
            records = future.result(timeout=15)
            assert set(records) == {'first', 'second', 'live'}
            generated = [v for calls in ray.get([a.snapshot.remote() for a in actors]) for batch in calls for v in batch]
            assert sorted(generated) == [1, 2, 4]
        finally:
            ray.get([a.release.remote() for a in actors])
            requests.finish(0)
            requests.finish(1)
            for actor in actors:
                ray.kill(actor)


def test_only_completion_transport_changes_from_deployed_source():
    def methods(path):
        tree = ast.parse(path.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'LoopOwner')
        return {n.name: ast.dump(n) for n in cls.body if isinstance(n, ast.FunctionDef)}
    before = methods(Path(os.environ['DEPLOYED_LOOP_ENTRY']) / 'loop_owner_rollout.py')
    after = methods(Path(__file__).with_name('appworld-batch-coalescing-20261002') / 'loop_owner_rollout.py')
    assert before.keys() == after.keys()
    assert {k for k in before if before[k] != after[k]} == {'collect_native_trajectories'}
