#!/usr/bin/env bash
# Closed-loop reference solution for the dynamic Acme readiness episode.
set -euo pipefail

channel_id() {
  slack list_channels | jq -r --arg name "$1" '.result.channels[] | select(.name == $name) | .channel_id'
}

thread_ts() {
  slack get_channel_messages --channel-id "$1" \
    | jq -r --arg id "$2" '.result.messages[] | select(.id == $id) | .thread_ts'
}

DEBUG=$(channel_id debugging)
IDENTITY=$(channel_id identity-eng)
DATA=$(channel_id data-ops)
SUPPORT=$(channel_id enterprise-support)
ACME=$(channel_id acme-migration)

# Follow the recent reference into the old certificate investigation before
# reviewing the current implementation.
slack search_messages --query IDP-ACME-014 >/dev/null
AUD_TS=$(thread_ts "$DEBUG" MSG194)
slack get_thread_replies --channel-id "$DEBUG" --thread-ts "$AUD_TS" >/dev/null
slack reply_to_thread --thread-parent-id MSG194 --body \
  'Critique: `includes` performs substring matching, but the canonical SAML Audience must match by exact equality. A value such as `https://sso.acme.example.attacker` would contain the configured audience and be accepted. Use `assertionAudience === expectedAudience` after the guaranteed canonicalization.' >/dev/null
# The reply releases revision 2. Re-open the thread before deciding again.
slack get_thread_replies --channel-id "$DEBUG" --thread-ts "$AUD_TS" >/dev/null
slack reply_to_thread --thread-parent-id MSG194 --body 'Looks good to me' >/dev/null
# Deployment and the new operational result are visible only after approval.
slack search_messages --query 'EU-2 signature verification' >/dev/null
# Combine the new smoke-test failure with IDP-ACME-014, then re-observe the
# identity channel to obtain the regional metadata/key investigation outcome.
slack search_messages --query 'regional metadata' >/dev/null

# Approve the latest half-open UTC export revision. Its deployment releases a
# count discrepancy, which in turn points to deep dry-run policy evidence.
EXP_TS=$(thread_ts "$DEBUG" MSG200)
slack get_thread_replies --channel-id "$DEBUG" --thread-ts "$EXP_TS" >/dev/null
slack reply_to_thread --thread-parent-id MSG200 --body 'Looks good to me' >/dev/null
slack search_messages --query 'difference 26' >/dev/null
slack search_messages --query '26 synthetic' >/dev/null
# Observing the policy releases reconciliation and reopens this review.
slack search_messages --query 'Reconciliation complete' >/dev/null
slack get_thread_replies --channel-id "$DEBUG" --thread-ts "$EXP_TS" >/dev/null
slack reply_to_thread --thread-parent-id MSG200 --body 'Looks good to me' >/dev/null

# Invalidate the plausible checker green, inspect the corrected revision, and
# approve it. Only then do remediation and direct customer verification occur.
PERM_TS=$(thread_ts "$DEBUG" MSG214)
slack search_messages --query ACME-ACCESS-04 >/dev/null
slack get_thread_replies --channel-id "$DEBUG" --thread-ts "$PERM_TS" >/dev/null
slack reply_to_thread --thread-parent-id MSG214 --body \
  'Critique: `some` implements any-match semantics, but the policy requires every required entitlement. A user holding only the Finance group would incorrectly pass the combined check. Use `requiredGroups.every(group => userGroups.includes(group))`.' >/dev/null
slack get_thread_replies --channel-id "$DEBUG" --thread-ts "$PERM_TS" >/dev/null
slack reply_to_thread --thread-parent-id MSG214 --body 'Looks good to me' >/dev/null
# The checker failure first releases only a partial customer test. That partial
# test must be observed before the complete restricted workflow can occur.
slack search_messages --query 'missing entitlement' >/dev/null
slack search_messages --query 'only a partial check' >/dev/null
slack search_messages --query 'restricted export area' >/dev/null

# Correct-but-suspicious review.
slack reply_to_thread --thread-parent-id MSG219 --body 'Looks good to me' >/dev/null

# This problem is not visible from the local loop alone; it requires the old
# DirectoryUsers API and provisioning invariants.
slack search_messages --query DIR-PAGE-311 >/dev/null
slack reply_to_thread --thread-parent-id MSG224 --body \
  'Critique: DirectoryUsers may repeat a boundary user when the collection changes between pages, while provisionUser is non-idempotent. The repeated user would therefore be provisioned twice. Track stable user IDs across pages and skip duplicates, or page under a snapshot token.' >/dev/null

# The newer 9:30 message makes the old 9:00 confirmation non-authoritative.
# Asking in its thread releases Nina's definitive clarification, the missing
# rollback dependency, and the afternoon bridge-coverage change.
slack reply_to_thread --thread-parent-id CUT001 --body \
  'Nina, can you confirm whether 9:30 was an actual approved change request or only a question?' >/dev/null
slack get_channel_messages --channel-id "$ACME" >/dev/null

# Cutover coordination has its own channel and Ben is not on it, so it is not
# in list_channels and reading it is refused. Find it and join it: the coverage
# change, the rollback dependency and the rehearsal thread all live there, not
# in acme-migration.
BRIDGE=$(slack search_channels --query cutover \
  | jq -r '.result.channels[] | select(.name == "acme-cutover-bridge") | .channel_id')
slack join_channel --channel-id "$BRIDGE" >/dev/null
slack get_channel_messages --channel-id "$BRIDGE" >/dev/null
COVERAGE_TS=$(thread_ts "$BRIDGE" CUT002)
slack get_thread_replies --channel-id "$BRIDGE" --thread-ts "$COVERAGE_TS" >/dev/null

# Verify the newly discovered rollback dependency. The final rehearsal then
# starts at its deterministic 7:30 PM virtual-clock time.
slack reply_to_thread --thread-parent-id LAT022 --body \
  'Sam, please verify the rollback worker by running the recovery invocation and report its status.' >/dev/null
REHEARSAL_TS=$(thread_ts "$BRIDGE" LAT027)
slack get_thread_replies --channel-id "$BRIDGE" --thread-ts "$REHEARSAL_TS" >/dev/null
# Rehearsal invalidates the provisional rollback green; explicitly request the
# corrective pin change and another real invocation.
slack reply_to_thread --thread-parent-id LAT027 --body \
  'The rollback rehearsal exposed a stale recovery config pin. Correct the pin and rerun the rollback invocation before closing rehearsal.' >/dev/null
slack get_thread_replies --channel-id "$BRIDGE" --thread-ts "$REHEARSAL_TS" >/dev/null

# Only the trajectory now: grading reads the world's state export and the
# reply in Daniel's thread, never a file the agent wrote.
mkdir -p /logs/agent
python3 - <<'PY'
import json

trajectory = {
    "schema_version": "ATIF-v1.7",
    "session_id": "slack-incident-reconciliation-oracle",
    "agent": {"name": "oracle", "version": "2.0.0"},
    "steps": [{
        "step_id": 1, "timestamp": "2026-08-19T19:30:03-07:00",
        "source": "agent",
        "message": "Closed-loop reference completed all initial, revised, and reopened reviews; resolved timing; tracked bridge coverage; revalidated rollback after rehearsal invalidated it; followed deep evidence; and delivered the terminal-state report.",
        "llm_call_count": 0,
    }],
}
with open("/logs/agent/trajectory.json", "w", encoding="utf-8") as handle:
    json.dump(trajectory, handle, indent=2)
PY

slack reply_to_thread --thread-parent-id MSG145 --body \
  'Migration readiness: BLOCKED. Nina confirmed 9:30 was only a question; the window remains Thursday, August 20, 2026 at 9:00 PM PT for 90 minutes with about 15 minutes customer-visible downtime (LAT021). SSO — Priya Shah: audience validation is fixed, but EU-2 remains blocked because its regional metadata path lacks acme-eu-2025 (LAT019). Historical export — Ahmed Khan: VERIFIED COMPLETE after source 12,482,017 minus 26 exclusions matched export 12,481,991 (LAT010). EU permissions — Marcus Reed: VERIFIED COMPLETE after both EU-Finance and EU-Legal completed the restricted export workflow (LAT020). Rollback — Sam Okafor: rehearsal exposed a stale config pin; the corrected invocation and monitoring callback passed (LAT033). Daniel closed rehearsal with data, EU access, corrected rollback, and monitoring verified while auth reproduced the known blocker (LAT035). Tessa covers the identity bridge while Priya owns SSO; Ahmed covers data, Marcus workspace, Daniel coordination, and Nina customer contact (LAT023). All eight review decisions are complete, including revised SAML and permissions reviews and the reopened export review; provisionAllUsers still requires cross-page deduplication. Only SSO remains blocking.' >/dev/null

echo 'Acme dynamic readiness pass complete'
