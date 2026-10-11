"""Pseudonyms keep one real value mapped to one stable, keyed placeholder."""
import pytest

from forge.safety.pseudonymizer import PSEUDONYM_PATTERN, Pseudonymizer

KEY = b"test-key-not-a-secret"


def test_same_value_gets_the_same_pseudonym_every_time():
    p = Pseudonymizer(KEY)
    assert p.replace("phone", "415-555-0134") == p.replace("phone", "415-555-0134")


def test_normalized_spellings_of_one_value_collide():
    p = Pseudonymizer(KEY)
    assert p.replace("email", "Alice@Acme.com") == p.replace("email", "alice@acme.com")
    assert p.replace("phone", "(415) 555-0134") == p.replace("phone", "415.555.0134")
    assert p.replace("card", "4111 1111 1111 1111") == p.replace("card", "4111-1111-1111-1111")


def test_different_values_and_kinds_get_different_pseudonyms():
    p = Pseudonymizer(KEY)
    assert p.replace("person", "Sarah Chen") != p.replace("person", "Marcus Webb")
    assert p.replace("ssn", "123-45-6789") != p.replace("phone", "123-45-6789")


def test_a_different_key_gives_a_different_pseudonym():
    assert Pseudonymizer(KEY).replace("person", "Sarah Chen") != Pseudonymizer(b"other").replace(
        "person", "Sarah Chen"
    )


def test_pseudonyms_keep_a_usable_shape_and_hide_the_raw_value():
    p = Pseudonymizer(KEY)
    email = p.replace("email", "alice@acme.com")
    assert email.startswith("user_") and email.endswith("@example.invalid")
    assert "alice" not in email
    assert p.replace("person", "Sarah Chen").startswith("Person_")
    assert p.replace("phone", "415-555-0134").startswith("PHONE_")


def test_every_pseudonym_is_recognized_and_real_values_are_not():
    p = Pseudonymizer(KEY)
    for kind, value in [("email", "a@b.co"), ("person", "Sarah"), ("ip", "8.8.8.8"), ("address", "1 Main St")]:
        assert PSEUDONYM_PATTERN.fullmatch(p.replace(kind, value))
    for text in ["alice@acme.com", "Sarah Chen", "PHONE_xyz", "commit_deadbeef", "user_ab12cd34@realcorp.com"]:
        assert not PSEUDONYM_PATTERN.fullmatch(text)


def test_an_empty_key_is_rejected():
    with pytest.raises(ValueError):
        Pseudonymizer(b"")
