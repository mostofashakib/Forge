from __future__ import annotations

from fastapi.testclient import TestClient

from tests.premade.conftest import count_selects


def test_bulk_archive_query_count_does_not_grow_with_ids(load_premade):
    app = load_premade("gmail")
    client = TestClient(app.app)
    with app.SessionLocal() as db:
        ids = [e.id for e in db.query(app.Email).all()]

    few = count_selects(app.engine, lambda: client.post("/bulk_archive", json={"email_ids": ids[:1]}))
    many = count_selects(app.engine, lambda: client.post("/bulk_archive", json={"email_ids": ids[1:6]}))
    assert many == few


def test_bulk_archive_reports_found_ids_in_request_order(load_premade):
    app = load_premade("gmail")
    client = TestClient(app.app)
    with app.SessionLocal() as db:
        first, second = [e.id for e in db.query(app.Email).limit(2).all()]

    body = client.post("/bulk_archive", json={"email_ids": [second, "missing", first]}).json()

    assert body["archived"] == [second, first]
    assert body["count"] == 2
    with app.SessionLocal() as db:
        assert {db.get(app.Email, i).folder for i in (first, second)} == {"archive"}
