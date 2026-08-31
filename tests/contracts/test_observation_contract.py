from __future__ import annotations

import pytest

from forge.contracts import Observation, ObservationEncoder


class _Passthrough(ObservationEncoder):
    def encode(self, state: dict, ctx) -> Observation:
        return Observation(payload=state)


def test_an_encoder_wraps_state_in_an_observation():
    obs = _Passthrough().encode({"tickets": []}, None)
    assert obs.payload == {"tickets": []}


def test_an_encoder_returns_an_observation_not_a_dict():
    # Negative: consumers rely on the typed shape; a bare dict would break them.
    assert isinstance(_Passthrough().encode({}, None), Observation)


def test_an_encoder_missing_encode_cannot_be_instantiated():
    class Incomplete(ObservationEncoder):
        pass

    with pytest.raises(TypeError, match="abstract"):
        Incomplete()
