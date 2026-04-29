# tests/runtime/test_context.py
from forge.runtime.context import RuntimeContext


def test_same_seed_produces_same_rng_sequence():
    ctx1 = RuntimeContext(seed=42)
    ctx2 = RuntimeContext(seed=42)
    assert [ctx1.rng.random() for _ in range(10)] == [ctx2.rng.random() for _ in range(10)]


def test_different_seeds_produce_different_sequences():
    ctx1 = RuntimeContext(seed=1)
    ctx2 = RuntimeContext(seed=2)
    assert ctx1.rng.random() != ctx2.rng.random()


def test_id_generator_is_sequential_and_deterministic():
    ctx = RuntimeContext(seed=0)
    assert ctx.id_generator.next("email") == "email_0000"
    assert ctx.id_generator.next("email") == "email_0001"
    assert ctx.id_generator.next("thread") == "thread_0000"


def test_same_seed_produces_same_id_sequence():
    ctx1 = RuntimeContext(seed=99)
    ctx2 = RuntimeContext(seed=99)
    ids1 = [ctx1.id_generator.next("x") for _ in range(5)]
    ids2 = [ctx2.id_generator.next("x") for _ in range(5)]
    assert ids1 == ids2


def test_clock_starts_at_epoch_and_advances():
    ctx = RuntimeContext(seed=0)
    t0 = ctx.clock.now()
    ctx.clock.advance()
    t1 = ctx.clock.now()
    assert t1 > t0


def test_same_seed_produces_same_clock_sequence():
    ctx1 = RuntimeContext(seed=5)
    ctx2 = RuntimeContext(seed=5)
    for _ in range(3):
        ctx1.clock.advance()
        ctx2.clock.advance()
    assert ctx1.clock.now() == ctx2.clock.now()
