"""Seed data for the Slack service.

This module is the deterministic generator: it defines the seed rows that are
written into the SQLite database, which is the single source of truth for
state. No parallel in-memory state is built from these constants.
"""

from __future__ import annotations

START_MS = 1_700_000_000_000
STEP_MS = 1_000

SLACK_USERS = [
    ("U001", "Alice Nguyen", "alice@example.local", "admin", "sre", "alice"),
    ("U002", "Ben Ortiz", "ben@example.local", "member", "platform", "ben"),
    ("U003", "Cara Singh", "cara@example.local", "member", "security", "cara"),
    ("U004", "Diego Kim", "diego@example.local", "member", "support", "diego"),
    ("U005", "Eva Smith", "eva@example.local", "member", "design", "eva"),
    ("U006", "Ben Ortíz", "ben.ortiz@example.local", "member", "support", "benortiz"),
]

SLACK_CHANNELS = [
    ("C001", "general", False, "U001", START_MS),
    ("C002", "platform-outages", False, "U002", START_MS),
    ("C003", "incidents", False, "U001", START_MS),
    ("C004", "design-confidential", True, "U005", START_MS),
    ("C005", "security-alerts", False, "U003", START_MS),
    ("C006", "security-alert", False, "U003", START_MS),
    ("C007", "platform-outage", False, "U002", START_MS),
]

SLACK_MESSAGES = [
    ("MSG001", "C001", "U002", "Daily handoff is complete.", 1),
    ("MSG002", "C002", "U003", "Latency watch is normal.", 2),
    ("MSG003", "C003", "U001", "Initial incident review references #general for comms.", 3),
    ("MSG004", "C003", "U002", "Incident bridge opened with SRE.", 4),
    ("MSG005", "C001", "U004", "Support queue is clear.", 5),
    ("MSG006", "C003", "U001", "Follow-up notes reference #platform-outages for customer impact.", 6),
    ("MSG007", "C002", "U002", "Database failover rehearsal rescheduled to next Tuesday.", 7),
    ("MSG008", "C003", "U003", "Security confirms no credential exposure.", 8),
    ("MSG009", "C001", "U001", "Alice will summarize the incident later.", 9),
    ("MSG010", "C001", "U004", "Support has prepared status page copy.", 10),
    ("MSG011", "C002", "U001", "Alice checked platform outage graphs.", 11),
    ("MSG012", "C003", "U001", "Most recent update: coordinate remediation in #platform-outages.", 20),
    
    # Threaded replies
    ("MSG013", "C003", "U003", "Joining the bridge now.", 13),
    ("MSG014", "C003", "U001", "I'm on as well.", 14),
    
    # Group chat messages
    ("MSG015", "G001", "U002", "Let's keep the core updates here.", 15),
    ("MSG016", "G001", "U003", "Agreed, too much noise in public channels.", 16),
    
    # DM chat messages
    ("MSG017", "D001", "U001", "Did you verify the platform database connection?", 17),
    ("MSG018", "D001", "U002", "Yes, verified and all good.", 18),
    ("MSG019", "C003", "U004", "Incident bridge opened with SRE team.", 12),
    ("MSG020", "C006", "U003", "Alert routing pipeline test.", 19),
    ("MSG021", "C007", "U002", "Outage tracker placeholder.", 21),
    # Newest channel-referencing message in #incidents (authored by Cara, not
    # Alice, so Alice's most recent reference stays MSG012/#platform-outages).
    ("MSG022", "C003", "U003", "Postmortem doc drafting moved to #general.", 23),

    # Threaded replies under MSG002 in #platform-outages: Ben's own reply (the
    # one tasks may ask him to edit) next to the look-alike author's reply.
    ("MSG023", "C002", "U002", "Watching the graphs closely.", 24),
    ("MSG024", "C002", "U006", "Support is standing by.", 25),
    ("MSG025", "C001", "U003", "Status page draft looks good.", 26),
    ("MSG026", "C001", "U006", "Support will mirror the update.", 27),
]

# Thread parents for seeded replies, shared by both state builders.
SLACK_THREAD_PARENTS = {
    "MSG013": "MSG004",
    "MSG014": "MSG004",
    "MSG023": "MSG002",
    "MSG024": "MSG002",
    "MSG025": "MSG010",
    "MSG026": "MSG010",
}

# Seeded edits as {message_id: edited_at offset in STEP_MS}.
SLACK_EDITED_MESSAGES = {
    "MSG007": 15,
}

# Seeded group/DM chats as (chat_id, type, name, created_at_ms).
SLACK_CHATS = [
    ("G001", "group", "Incident Response Core", START_MS + 10 * STEP_MS),
    ("D001", "dm", None, START_MS + 11 * STEP_MS),
]

# Seeded chat membership as (chat_id, user_id).
SLACK_CHAT_PARTICIPANTS = [
    ("G001", "U001"),
    ("G001", "U002"),
    ("G001", "U003"),
    ("D001", "U001"),
    ("D001", "U002"),
]


def build_slack_memberships() -> list[tuple[str, str, str, str, int]]:
    """Channel membership rows as (membership_id, channel_id, user_id, role, joined_at_ms)."""
    memberships = []
    for channel_id, _, _, owner_id, _ in SLACK_CHANNELS:
        for user_id, *_ in SLACK_USERS:
            # Eva (U005) is only in her private channel C004
            if user_id == "U005" and channel_id != "C004":
                continue
            # Non-Eva users are not in C004
            if channel_id == "C004" and user_id != "U005":
                continue

            memberships.append(
                (
                    f"M{channel_id[-3:]}{user_id[-3:]}",
                    channel_id,
                    user_id,
                    "owner" if user_id == owner_id else "member",
                    START_MS,
                )
            )
    return memberships
