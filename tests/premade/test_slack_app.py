from __future__ import annotations

from fastapi.testclient import TestClient

from tests.premade.conftest import count_selects


def _add_channel(app, db, name: str, *, top_level: int, replies: int, pinned: int, unread: int):
    channel_id = f"C_{name}"
    db.add(app.Channel(id=channel_id, name=name, created_at=app._now(db)))
    for i in range(top_level):
        db.add(app.Message(
            id=f"{channel_id}_m{i}", channel_id=channel_id, user_name="ana",
            text=f"{name} message {i}", timestamp=app._now(db), is_pinned=i < pinned,
        ))
    for i in range(replies):
        db.add(app.Message(
            id=f"{channel_id}_r{i}", channel_id=channel_id, user_name="bo",
            text="reply", timestamp=app._now(db), thread_parent_id=f"{channel_id}_m0",
        ))
    db.add(app.ChannelReadState(channel_id=channel_id, unread_count=unread))
    db.commit()


def test_state_query_count_does_not_grow_with_channels(load_premade):
    app = load_premade("slack")
    with app.SessionLocal() as db:
        before = count_selects(app.engine, lambda: app._get_state_dict(db))
        for i in range(5):
            _add_channel(app, db, f"extra{i}", top_level=2, replies=1, pinned=1, unread=i)
        after = count_selects(app.engine, lambda: app._get_state_dict(db))
    assert after == before


def test_state_channel_counts_match_the_data(load_premade):
    app = load_premade("slack")
    with app.SessionLocal() as db:
        _add_channel(app, db, "ops", top_level=3, replies=2, pinned=1, unread=4)
        state = app._get_state_dict(db)
        channel = next(c for c in state["channels"] if c["name"] == "ops")
        unread_sum = sum(c["unread"] for c in state["channels"])

    assert (channel["message_count"], channel["pinned_count"], channel["unread"]) == (3, 1, 4)
    assert state["total_unread"] == state["dm_unread"] + unread_sum


def test_channel_without_read_state_has_zero_unread(load_premade):
    app = load_premade("slack")
    with app.SessionLocal() as db:
        db.add(app.Channel(id="C_new", name="new", created_at=app._now(db)))
        db.commit()
        channel = next(c for c in app._get_state_dict(db)["channels"] if c["name"] == "new")
    assert (channel["message_count"], channel["pinned_count"], channel["unread"]) == (0, 0, 0)


def test_message_listing_loads_reactions_in_one_query(load_premade):
    app = load_premade("slack")
    client = TestClient(app.app)

    def listing_queries(total: int) -> tuple[int, list[dict]]:
        with app.SessionLocal() as db:
            db.query(app.Message).filter(app.Message.channel_id == "C_react").delete()
            db.query(app.Reaction).delete()
            db.merge(app.Channel(id="C_react", name="react", created_at=app._now(db)))
            for i in range(total):
                db.add(app.Message(
                    id=f"rm{i}", channel_id="C_react", user_name="ana",
                    text=f"hello {i}", timestamp=f"2026-01-01T00:00:{i:02d}",
                ))
                db.add(app.Reaction(id=f"rx{i}", message_id=f"rm{i}", emoji="+1", user_name="bo"))
                db.add(app.Reaction(id=f"ry{i}", message_id=f"rm{i}", emoji="+1", user_name="cy"))
            db.commit()
        body = {}
        count = count_selects(app.engine, lambda: body.update(
            client.post("/get_channel_messages", json={"channel": "react"}).json()
        ))
        return count, body["messages"]

    few, _ = listing_queries(2)
    many, messages = listing_queries(8)
    assert many == few
    assert [m["reactions"] for m in messages] == [{"+1": ["bo", "cy"]}] * 8
