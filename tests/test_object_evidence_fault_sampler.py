from models.object_evidence.fault_sampler import PairedFaultSampler


def test_sampler_is_reproducible():
    left, right = PairedFaultSampler(42), PairedFaultSampler(42)
    assert [left.sample(5) for _ in range(20)] == [right.sample(5) for _ in range(20)]


def test_temporal_fault_is_constant_after_onset():
    episode = PairedFaultSampler(7).sample(5)
    assert episode.active == (False, False, True, True, True)
    assert episode.spec.severity in (0.3, 0.6, 0.9, 1.0)


def test_single_frame_is_faulted():
    assert PairedFaultSampler(7).sample(1).active == (True,)
