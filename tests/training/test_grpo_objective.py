from __future__ import annotations

import pytest

from forge.training._backends import _clipped_grpo_loss


def test_positive_advantage_policy_ratio_is_clipped():
    torch = pytest.importorskip("torch")
    current = torch.tensor([[2.0]])
    old = torch.tensor([[0.0]])
    advantage = torch.tensor([[1.0]])
    mask = torch.tensor([[True]])

    loss = _clipped_grpo_loss(
        torch,
        current,
        old,
        advantage,
        mask,
        clip_epsilon=0.2,
        kl_beta=0.0,
    )

    assert loss.item() == pytest.approx(-1.2)


def test_zero_advantage_produces_no_policy_loss():
    torch = pytest.importorskip("torch")
    loss = _clipped_grpo_loss(
        torch,
        torch.tensor([[0.0]]),
        torch.tensor([[0.0]]),
        torch.tensor([[0.0]]),
        torch.tensor([[True]]),
        clip_epsilon=0.2,
        kl_beta=0.0,
    )
    assert loss.item() == 0.0
