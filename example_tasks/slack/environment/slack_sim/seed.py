"""Deterministic, lived-in Slack workspace for the Acme migration episode.

The seed includes ordinary company history, threaded side-conversations,
group chats, unread direct messages, and reactions that predate the episode.
Acme evidence uses explicit virtual-clock instants; surrounding background
traffic uses deterministic positions. The migration truth is defined only by
the Acme fixture and its transition graph.
"""

from __future__ import annotations

from dataclasses import replace

from slack_sim.clock import moment
from slack_sim.identity import LOGGED_IN_USER
from slack_sim.scenario import Scenario
from slack_sim.models import (
    Channel,
    ChannelTopic,
    Chat,
    ChatParticipant,
    ConversationRead,
    LatentMessage,
    Membership,
    Message,
    LatentMembership,
    LatentNotification,
    MessageMention,
    Notification,
    Pin,
    SavedItem,
    ThreadFollow,
    ScenarioEvent,
    ScenarioRule,
    Reaction,
    SlackUser,
    UserGroup,
    UserGroupMember,
    UserProfile,
)

BEN = LOGGED_IN_USER.user_id

SLACK_USERS = [
    SlackUser("U001", "Alice Nguyen", "alice@example.local", "admin", "incident-command", "alice"),
    LOGGED_IN_USER.as_user(),
    SlackUser("U003", "Cara Singh", "cara@example.local", "member", "reliability", "cara"),
    SlackUser("U004", "Diego Kim", "diego@example.local", "member", "customer-support", "diego"),
    SlackUser("U005", "Eva Smith", "eva@example.local", "member", "payments", "eva"),
    SlackUser("U006", "Frank Moore", "frank@example.local", "member", "checkout", "frank"),
    SlackUser("U007", "Grace Liu", "grace@example.local", "member", "inventory", "grace"),
    SlackUser("U008", "Hana Rossi", "hana@example.local", "member", "security", "hana"),
    SlackUser("U009", "Iris Chen", "iris@example.local", "member", "product", "iris"),
    SlackUser("U010", "Jon Bell", "jon@example.local", "guest", "legal", "jon"),
    SlackUser("U011", "Kira Patel", "kira@example.local", "member", "data", "kira"),
    SlackUser("U012", "Leo Martin", "leo@example.local", "member", "sales", "leo"),
    SlackUser("U013", "Mina Haddad", "mina@example.local", "member", "platform", "mina"),
    SlackUser("U014", "Noah Weber", "noah@example.local", "member", "checkout", "noah"),
    SlackUser("U015", "Idris Farouk", "idris@example.local", "member", "payments", "idris"),
    SlackUser("U016", "Ravi Raman", "ravi@example.local", "member", "reliability", "ravi"),
    SlackUser("U017", "Quinn Doyle", "quinn@example.local", "member", "quality", "quinn"),
    SlackUser("U018", "Rosa Ibarra", "rosa@example.local", "member", "customer-support", "rosa"),
    SlackUser("U019", "Sam Okafor", "sam@example.local", "member", "infrastructure", "sam"),
    SlackUser("U020", "Tara Volkov", "tara@example.local", "member", "data", "tara"),
    SlackUser("U021", "Umi Tanaka", "umi@example.local", "member", "design", "umi"),
    SlackUser("U022", "Viktor Novak", "viktor@example.local", "member", "security", "viktor"),
    SlackUser("U023", "Wendy Cho", "wendy@example.local", "member", "product", "wendy"),
    SlackUser("U024", "Xavier Dupont", "xavier@example.local", "member", "mobile", "xavier"),
    SlackUser("U025", "Yara Haddadi", "yara@example.local", "member", "marketing", "yara"),
    SlackUser("U026", "Zane Whitfield", "zane@example.local", "member", "finance", "zane"),
    SlackUser("U027", "Aisha Bello", "aisha@example.local", "member", "reliability", "aisha"),
    SlackUser("U028", "Boris Lang", "boris@example.local", "member", "checkout", "boris"),
    SlackUser("U029", "Chloe Dubois", "chloe@example.local", "member", "documentation", "chloe"),
    SlackUser("U030", "Dmitri Sokolov", "dmitri@example.local", "member", "infrastructure", "dmitri"),
    SlackUser("U031", "Elena Marchetti", "elena@example.local", "member", "payments", "elena"),
    SlackUser("U032", "Felix Braun", "felix@example.local", "member", "quality", "felix"),
    SlackUser("U033", "Gita Nair", "gita@example.local", "member", "data", "gita"),
    SlackUser("U034", "Hugo Silva", "hugo@example.local", "member", "sales", "hugo"),
    SlackUser("U035", "Ines Costa", "ines@example.local", "member", "customer-support", "ines"),
    SlackUser("U036", "Jamal Idris", "jamal@example.local", "member", "security", "jamal"),
    SlackUser("U037", "Kai Lindqvist", "kai@example.local", "member", "product", "kai"),
    SlackUser("U038", "Greta Fischer", "greta@example.local", "guest", "legal", "greta"),
    SlackUser("U039", "Milo Achebe", "milo@example.local", "member", "platform", "milo"),
    SlackUser("U040", "Nadia Petrova", "nadia@example.local", "admin", "incident-command", "nadia"),
    # Acme migration cast.
    SlackUser("U041", "Daniel Cho", "daniel@example.local", "member", "program-management", "daniel"),
    SlackUser("U042", "Priya Shah", "priya@example.local", "member", "identity", "priya"),
    SlackUser("U043", "Lena Ortiz", "lena@example.local", "member", "workspace-platform", "lena"),
    SlackUser("U044", "Marcus Reed", "marcus@example.local", "member", "workspace-platform", "marcus"),
    SlackUser("U045", "Ahmed Khan", "ahmed@example.local", "member", "data-operations", "ahmed"),
    SlackUser("U046", "Nina Foster", "nina@example.local", "member", "customer-success", "nina"),
    SlackUser("U047", "Omar Blake", "omar@example.local", "member", "support", "omar"),
    SlackUser("U048", "Tessa Vogel", "tessa@example.local", "member", "identity", "tessa"),
    SlackUser("U049", "Ethan Park", "ethan@example.local", "member", "backend", "ethan"),
    SlackUser("U050", "Sofia Chen", "sofia@example.local", "member", "platform", "sofia"),
    # A second Felix. Felix Braun in QA already existed and stays; the identity
    # rota needs its own, and `search_users` separates them on title and team.
    SlackUser("U051", "Felix Moreau", "felix.moreau@example.local", "member", "identity", "felixm"),
]

# User profiles capture reporting lines and operational identity data.
SLACK_USER_PROFILES = [
    UserProfile("U001", None, "VP, Reliability", "America/Los_Angeles", "incident executive"),
    UserProfile(BEN, "U001", "Platform Engineer", "America/Chicago", "Acme migration readiness"),
    UserProfile("U003", "U001", "Site Reliability Engineer", "Europe/London", "primary on-call"),
    UserProfile("U004", "U001", "Customer Support Lead", "America/New_York", "customer communications"),
    UserProfile("U005", "U001", "Payments Lead", "America/Los_Angeles", "payments escalation"),
    UserProfile("U006", "U001", "Checkout Lead", "America/Denver", "checkout mitigation"),
    UserProfile("U007", "U006", "Inventory Engineer", "Asia/Singapore", "inventory on-call"),
    UserProfile("U008", "U001", "Security Engineer", "Europe/Berlin", "security on-call"),
    UserProfile("U009", "U001", "Product Manager", "America/Los_Angeles", "launch coordinator"),
    UserProfile("U010", "U001", "Legal Counsel", "America/New_York", "restricted guest"),
    UserProfile("U011", "U006", "Data Engineer", "Asia/Kolkata", "analytics on-call"),
    UserProfile("U012", "U009", "Enterprise Account Executive", "America/New_York", "customer escalation"),
    UserProfile("U013", "U001", "Staff Platform Engineer", "Europe/Lisbon", "config-service migration"),
    UserProfile("U014", "U006", "Checkout Engineer", "America/New_York", "experiment owner"),
    UserProfile("U015", "U005", "Payments Engineer", "Africa/Cairo", "authorization metrics"),
    UserProfile("U016", "U003", "Site Reliability Engineer", "Asia/Kolkata", "secondary on-call"),
    UserProfile("U017", "U009", "QA Lead", "Europe/Dublin", "release verification"),
    UserProfile("U018", "U004", "Support Engineer", "America/Mexico_City", "front-line queue"),
    UserProfile("U019", "U001", "Infrastructure Engineer", "Africa/Lagos", "load-balancer upgrade"),
    UserProfile("U020", "U011", "Analytics Engineer", "Europe/Moscow", "warehouse loads"),
    UserProfile("U021", "U009", "Product Designer", "Asia/Tokyo", "checkout error states"),
    UserProfile("U022", "U008", "Security Engineer", "Europe/Prague", "secret rotation"),
    UserProfile("U023", "U009", "Product Manager", "America/Los_Angeles", "checkout surface"),
    UserProfile("U024", "U009", "Mobile Engineer", "Europe/Paris", "flagged rollout"),
    UserProfile("U025", "U009", "Marketing Manager", "Asia/Beirut", "all-hands logistics"),
    UserProfile("U026", "U001", "Finance Analyst", "America/Chicago", "revenue impact"),
    UserProfile("U027", "U003", "Site Reliability Engineer", "Africa/Accra", "weekend coverage"),
    UserProfile("U028", "U006", "Checkout Engineer", "Europe/Berlin", "latency watch"),
    UserProfile("U029", "U025", "Technical Writer", "Europe/Paris", "runbook upkeep"),
    UserProfile("U030", "U019", "Infrastructure Engineer", "Europe/Moscow", "network dashboards"),
    UserProfile("U031", "U005", "Payments Engineer", "Europe/Rome", "chargeback batches"),
    UserProfile("U032", "U017", "QA Engineer", "Europe/Berlin", "regression suite"),
    UserProfile("U033", "U011", "Data Engineer", "Asia/Kolkata", "partition backfills"),
    UserProfile("U034", "U012", "Account Executive", "America/Sao_Paulo", "ACME account"),
    UserProfile("U035", "U004", "Support Engineer", "Europe/Lisbon", "escalation queue"),
    UserProfile("U036", "U008", "Security Analyst", "Africa/Nairobi", "rotation window"),
    UserProfile("U037", "U009", "Product Manager", "Europe/Stockholm", "checkout copy"),
    UserProfile("U038", "U010", "Legal Counsel", "Europe/Berlin", "restricted guest"),
    UserProfile("U039", "U013", "Platform Engineer", "Africa/Lagos", "incident tooling"),
    UserProfile("U040", "U001", "Incident Commander", "Europe/Sofia", "postmortem program"),
    UserProfile("U041", "U001", "Migration Program Manager", "America/Los_Angeles", "Acme cutover — go/no-go 8:30 PM"),
    UserProfile("U042", "U001", "Identity Engineer", "America/Los_Angeles", "Acme SSO investigation"),
    UserProfile("U043", "U001", "Workspace Platform Engineer", "America/Denver", "OOO — Marcus covering Acme"),
    UserProfile("U044", "U001", "Workspace Platform Engineer", "America/New_York", "Covering Acme EU access"),
    UserProfile("U045", "U011", "Data Operations Engineer", "Europe/Berlin", "Acme export reconciliation"),
    UserProfile("U046", "U004", "Customer Success Manager", "America/New_York", "Acme cutover — available until 5:30 PM"),
    UserProfile("U047", "U004", "Support Engineer", "America/Chicago", "Enterprise queue — Acme EU-SAML"),
    UserProfile("U048", "U042", "Identity Engineer", "Europe/Dublin", "SSO / SAML"),
    UserProfile("U049", "U001", "Senior Backend Engineer", "America/New_York", "SAML validation path"),
    UserProfile("U050", "U013", "Platform Engineer", "America/Los_Angeles", "export partitioning"),
    UserProfile("U051", "U001", "Identity Engineer", "Europe/Berlin", "Identity on-call"),
]

SLACK_CHANNELS = [
    Channel("C001", "general", False, "U001"),
    Channel("C002", "checkout-ops", False, "U006"),
    Channel("C003", "incidents", False, "U001"),
    Channel("C004", "customer-comms", False, "U004"),
    Channel("C005", "payments-ops", False, "U005"),
    Channel("C006", "incident-482-private", True, "U001"),
    Channel("C007", "invoice-ops", False, "U005"),
    Channel("C008", "launch-orion", False, "U009"),
    Channel("C009", "security-operations", True, "U008"),
    Channel("C010", "enterprise-escalations", True, "U004"),
    Channel("C011", "data-quality", False, "U011"),
    Channel("C012", "engineering", False, "U001"),
    Channel("C013", "checkout-web", False, "U014"),
    Channel("C014", "oncall-handoffs", False, "U003"),
    Channel("C015", "design-review", False, "U021"),
    Channel("C016", "incident-postmortems", False, "U040"),
    Channel("C017", "exec-briefing", True, "U001"),
    Channel("C018", "social-coffee", False, "U025"),
    # Acme migration channels.
    Channel("C019", "acme-migration", False, "U041"),
    Channel("C020", "identity-eng", False, "U042"),
    Channel("C021", "data-ops", False, "U045"),
    Channel("C022", "enterprise-support", False, "U047"),
    Channel("C023", "debugging", False, "U049"),
    # Same-day cutover coordination. Public, so Ben can find it, but he is not
    # on the roster: joining it is part of the work.
    Channel("C024", "acme-cutover-bridge", False, "U041"),
]

SLACK_CHANNEL_TOPICS = [
    ChannelTopic("C001", "Company-wide announcements and discussion"),
    ChannelTopic("C002", "Checkout operations and on-call handoffs"),
    ChannelTopic("C003", "Canonical incident command threads"),
    ChannelTopic("C004", "Approved external communication drafts"),
    ChannelTopic("C005", "Payments service operations"),
    ChannelTopic("C006", "Restricted INC-482 responder room"),
    ChannelTopic("C007", "Invoice operations"),
    ChannelTopic("C008", "Project Orion launch coordination"),
    ChannelTopic("C009", "Restricted security investigations"),
    ChannelTopic("C010", "Named-account customer escalations"),
    ChannelTopic("C011", "Analytics pipeline and data-quality operations"),
    ChannelTopic("C012", "Engineering-wide planning and announcements"),
    ChannelTopic("C013", "Checkout web surface and experiments"),
    ChannelTopic("C014", "On-call rotation handoffs across teams"),
    ChannelTopic("C015", "Design critique and review requests"),
    ChannelTopic("C016", "Postmortem drafting and review"),
    ChannelTopic("C017", "Restricted executive incident briefings"),
    ChannelTopic("C018", "Office and social chatter"),
    ChannelTopic("C019", "Acme enterprise migration coordination"),
    ChannelTopic("C020", "Identity, SSO and SAML engineering"),
    ChannelTopic("C021", "Data operations, exports and backfills"),
    ChannelTopic("C022", "Enterprise customer support escalations"),
    ChannelTopic("C023", "Debugging investigations and code review requests. Tag [review-needed] for a senior pass."),
]

SLACK_USER_GROUPS = [
    UserGroup("S001", "Incident Commanders", "incident-commanders", "U001"),
    UserGroup("S002", "Checkout On-call", "checkout-oncall", "U006"),
    UserGroup("S003", "Customer Communications", "customer-comms", "U004"),
    UserGroup("S004", "Orion Launch Team", "orion-launch", "U009"),
    UserGroup("S005", "Security On-call", "security-oncall", "U008"),
    UserGroup("S006", "Platform Engineering", "platform-eng", "U001"),
    UserGroup("S007", "Data Platform", "data-platform", "U011"),
    UserGroup("S008", "Quality Guild", "quality-guild", "U017"),
    UserGroup("S009", "Payments On-call", "payments-oncall", "U005"),
    UserGroup("S010", "Design Systems", "design-systems", "U021"),
    UserGroup("S011", "Acme Migration Team", "acme-migration", "U041"),
]

SLACK_USER_GROUP_MEMBERS = [
    UserGroupMember("S001", "U001"), UserGroupMember("S001", "U003"), UserGroupMember("S001", "U040"),
    UserGroupMember("S002", BEN), UserGroupMember("S002", "U006"), UserGroupMember("S002", "U007"),
    UserGroupMember("S003", "U004"), UserGroupMember("S003", "U012"), UserGroupMember("S003", "U018"),
    UserGroupMember("S003", "U035"),
    UserGroupMember("S004", "U006"), UserGroupMember("S004", "U009"), UserGroupMember("S004", "U011"),
    UserGroupMember("S004", "U012"), UserGroupMember("S004", "U024"),
    UserGroupMember("S005", "U008"), UserGroupMember("S005", "U022"), UserGroupMember("S005", "U036"),
    UserGroupMember("S006", BEN), UserGroupMember("S006", "U013"), UserGroupMember("S006", "U039"),
    UserGroupMember("S007", "U011"), UserGroupMember("S007", "U020"), UserGroupMember("S007", "U033"),
    UserGroupMember("S008", "U017"), UserGroupMember("S008", "U032"),
    UserGroupMember("S009", "U005"), UserGroupMember("S009", "U015"), UserGroupMember("S009", "U031"),
    UserGroupMember("S010", "U021"), UserGroupMember("S010", "U023"), UserGroupMember("S010", "U037"),
    UserGroupMember("S011", BEN), UserGroupMember("S011", "U041"), UserGroupMember("S011", "U042"),
    UserGroupMember("S011", "U043"), UserGroupMember("S011", "U044"), UserGroupMember("S011", "U045"),
    UserGroupMember("S011", "U046"),
]

# ---------------------------------------------------------------------------
# Message log
#
# Written in chronological order. `created_step` is assigned from position
# below, so a new message can be dropped between two existing ones without
# renumbering anything. The step values in each Message() literal are
# placeholders and are overwritten.
#
# IDs are stable identifiers, not timestamps: MSG001-MSG040 keep the IDs the
# reward, the oracle, and the tests already reference, and later additions
# carry higher numbers regardless of where in time they land.
# ---------------------------------------------------------------------------

_BACKGROUND_LOG: tuple[Message, ...] = (
    # -- Ordinary workspace traffic, before the incident ---------------------
    Message("MSG041", "C001", "U025", "Reminder: the all-hands moves to Thursday this week.", 0),
    Message("MSG042", "C001", "U029", "Calendar invites are updated.", 0, "MSG041"),
    Message("MSG043", "C001", "U021", "Will the recording be posted for the Tokyo timezone?", 0, "MSG041", reply_to_id="MSG042"),
    Message("MSG044", "C001", "U025", "Yes, within a day of the session.", 0, "MSG041"),
    Message("MSG045", "C012", "U001", "Q3 reliability goals are published; please review before Friday.", 0),
    Message("MSG046", "C012", "U013", "Platform has comments on the error-budget section.", 0, "MSG045"),
    Message("MSG047", "C012", "U016", "Reliability agrees with the checkout SLO change.", 0, "MSG045", reply_to_id="MSG046"),
    Message("MSG048", "C012", "U039", "I will collect the platform comments into one doc.", 0, "MSG045", reply_to_id="MSG046"),
    Message("MSG049", "C014", "U003", "On-call handoff: cara to priya at 09:00 UTC tomorrow.", 0),
    Message("MSG050", "C014", "U016", "Acknowledged. Paging rules verified on my side.", 0, "MSG049"),
    Message("MSG051", "C014", "U027", "Weekend coverage is aisha and dmitri.", 0),
    Message("MSG052", "C013", "U014", "Express checkout experiment is at 25 percent.", 0),
    Message("MSG053", "C013", "U028", "Latency looks flat at 25 percent.", 0, "MSG052"),
    Message("MSG054", "C013", "U006", "Hold at 25 until the inventory read path is fixed.", 0, "MSG052", reply_to_id="MSG053"),
    Message("MSG055", "C015", "U021", "Checkout error-state mockups are ready for review.", 0),
    Message("MSG056", "C015", "U023", "The retry copy still needs product signoff.", 0, "MSG055"),
    Message("MSG057", "C015", "U037", "Product signs off on the retry copy.", 0, "MSG055", reply_to_id="MSG056"),
    Message("MSG058", "C011", "U011", "Nightly warehouse load finished 40 minutes late.", 0),
    Message("MSG059", "C011", "U020", "The upstream vendor feed was delayed again.", 0, "MSG058"),
    Message("MSG060", "C011", "U033", "Backfill is queued for the affected partitions.", 0, "MSG058", reply_to_id="MSG059"),
    Message("MSG061", "G005", "U013", "Standup: I am on the config-service migration today.", 0),
    Message("MSG062", "G005", "U019", "Infra: finishing the load-balancer upgrade.", 0),
    Message("MSG063", "G005", BEN, "Platform: working through the incident tooling backlog.", 0),
    Message("MSG064", "G004", "U006", "Rotation swap: noah covers Thursday.", 0),
    Message("MSG065", "G004", "U014", "Confirmed.", 0),
    Message("MSG066", "D004", "U004", "When you get a chance, the status-page template needs an engineering reviewer.", 0),
    Message("MSG067", "D004", BEN, "I will look at it after the current on-call block.", 0),
    Message("MSG068", "D001", "U001", "Ben, can you own incident comms reconciliation this quarter?", 0),
    Message("MSG069", "D001", "U001", "No rush, but I would like an answer before the next review.", 0),
    Message("MSG070", "C018", "U026", "Coffee cart is on the third floor today.", 0),
    Message("MSG071", "C018", "U035", "Thank you, that saved my morning.", 0, "MSG070"),
    Message("MSG072", "C005", "U015", "Authorization success rate is steady at 99.4 percent.", 0),
    Message("MSG073", "C005", "U031", "Chargeback batch completed without retries.", 0, "MSG072"),
    Message("MSG074", "C008", "U009", "Orion go/no-go is Thursday; owners please post evidence.", 0),
    Message("MSG075", "C008", "U024", "Mobile client is ready for the flagged rollout.", 0, "MSG074"),

    # -- INC-482 opens -------------------------------------------------------
    Message("MSG001", "C003", "U001", "[INC-482] SEV-1 checkout failures — OPEN", 0),
    Message("MSG076", "C003", "U003", "Paging the checkout rotation now.", 0, "MSG001"),
    Message("MSG002", "C003", "U001", "[INC-428] SEV-2 invoice delays — RESOLVED", 0),
    Message("MSG077", "C014", "U016", "Escalating INC-482 to the checkout rotation.", 0),
    Message("MSG003", "C004", "U004", "[INC-482] Customer status thread", 0),
    Message("MSG078", "C004", "U018", "Support queue is already seeing checkout complaints.", 0, "MSG003"),
    Message("MSG004", "C002", "U006", f"[INC-482] ACTION owner=@{LOGGED_IN_USER.handle} Verify checkout feature-flag state.", 0),
    Message("MSG079", "C002", "U028", "[INC-482] ACTION owner=@noah Roll the express-checkout experiment back to 0.", 0),
    Message("MSG005", "C005", "U005", "[INC-482] Payments telemetry is healthy.", 0),
    Message("MSG080", "C005", "U015", "No authorization errors in the last 15 minutes.", 0, "MSG005"),
    Message("MSG006", "C007", "U005", f"[INC-428] ACTION owner=@{LOGGED_IN_USER.handle} Check invoice retry queue.", 0),
    Message("MSG007", "C001", "U008", "Checkout reports are being discussed in #incidents.", 0),
    Message("MSG081", "C001", "U029", "Please keep incident updates in the incident channel.", 0, "MSG007"),
    Message("MSG008", "C006", "U001", "[INC-482] Private note: no evidence of credential exposure.", 0),
    Message("MSG082", "C006", "U003", "[INC-482] Private note: the affected account list stays internal until legal clears it.", 0),
    Message("MSG083", "C017", "U001", f"[INC-482] ACTION owner=@{LOGGED_IN_USER.handle} Prepare the executive one-pager.", 0),
    Message("MSG009", "C003", "U001", "MITIGATION: roll back checkout-api.", 0, "MSG001"),
    Message("MSG084", "C003", "U006", "Rollback of checkout-api is queued behind the current deploy.", 0, "MSG001", reply_to_id="MSG009"),
    Message("MSG010", "C003", "U001", "OWNER: @alice is incident commander.", 0, "MSG001"),
    Message("MSG011", "C003", "U004", "OWNER: @diego taking over.", 0, "MSG001", reply_to_id="MSG010"),
    Message("MSG085", "C003", "U003", "SCOPE: checkout-api impacted; inventory-api is not impacted.", 0, "MSG001"),
    Message("MSG012", "C003", "U001", "HANDOFF: incident command to @cara.", 0, "MSG001"),
    Message("MSG086", "C003", "U004", "HANDOFF: incident command to @frank.", 0, "MSG001", reply_to_id="MSG012"),
    Message("MSG013", "C003", "U004", "HANDOFF: incident command to @eva.", 0, "MSG001"),
    Message("MSG087", "C003", "U003", "MITIGATION: roll back checkout-api and hold the express-checkout experiment.", 0, "MSG001", reply_to_id="MSG009"),
    Message("MSG014", "C003", "U003", "SCOPE: checkout-api and inventory-api impacted; payments-api is not impacted.", 0, "MSG001"),
    Message("MSG015", "C003", "U006", "SCOPE: payments-api also impacted.", 0, "MSG001", reply_to_id="MSG014"),
    Message("MSG088", "C003", "U007", "Inventory reads are still landing on the secondary region.", 0, "MSG001", reply_to_id="MSG014"),
    Message("MSG016", "C003", "U003", "MITIGATION: disable express checkout and pin inventory reads to the primary region.", 0, "MSG001"),
    Message("MSG017", "C003", "U001", "HANDOFF: incident command to @alice.", 0, "MSG001"),
    Message("MSG089", "C009", "U022", "[SEC-91] Webhook secret rotation is scheduled after INC-482 closes.", 0),
    Message("MSG018", "C004", "U004", "DRAFT UPDATE: Checkout is unavailable for every customer.", 0, "MSG003"),
    Message("MSG019", "C004", "U005", "APPROVED UPDATE: Payments are unavailable.", 0, "MSG003", reply_to_id="MSG018"),
    Message("MSG020", "C004", "U004", "APPROVED UPDATE: Some checkout attempts are failing. Please retry in 10 minutes while we mitigate.", 0, "MSG003"),
    Message("MSG090", "C004", "U012", "APPROVED UPDATE: Checkout is fully restored for all customers.", 0, "MSG003", reply_to_id="MSG020"),
    Message("MSG091", "C004", "U004", "DRAFT UPDATE: We will publish a closing note once the incident resolves.", 0, "MSG003", reply_to_id="MSG090"),
    Message("MSG021", "C002", "U006", f"[INC-482] ACTION owner=@{LOGGED_IN_USER.handle} Capture checkout error samples. DONE", 0),
    Message("MSG092", "C002", "U006", f"[INC-482] ACTION owner=@{LOGGED_IN_USER.handle} Draft the customer timeline. CANCELLED", 0),
    Message("MSG022", "C002", "U007", f"[INC-482] ACTION owner=@{LOGGED_IN_USER.handle} Confirm inventory primary-region health.", 0),
    Message("MSG023", "C002", "U006", "[INC-482] ACTION owner=@eva Review payment authorization logs.", 0),
    Message("MSG093", "C002", "U007", "Primary-region reads look healthy from the inventory side.", 0, "MSG022", reply_to_id="MSG022"),
    Message("MSG024", "C007", "U005", f"[INC-428] ACTION owner=@{LOGGED_IN_USER.handle} Validate invoice recovery.", 0),
    Message("MSG025", "C003", "U006", "[INC-482] I still think rollback is safer.", 0, "MSG001", reply_to_id="MSG016"),
    Message("MSG094", "C003", "U016", "Agreed with the current mitigation; no further changes tonight.", 0, "MSG001", reply_to_id="MSG016"),
    Message("MSG026", "C003", BEN, "SUMMARY DRAFT: checkout-api only; owner @alice; rollback pending.", 0, "MSG001"),
    Message("MSG027", "C001", "U004", "The customer update for INC-482 is ready in #customer-comms.", 0),
    Message("MSG028", "C005", "U005", "[INC-428] Invoice retry metrics have recovered.", 0),
    Message("MSG121", "G001", "U001", "Using this room for coordination that does not belong in the incident thread.", 0),
    Message("MSG122", "G001", "U004", "Customer comms will stay in the customer thread.", 0),
    Message("MSG029", "D002", "U003", "Ping me when the INC-482 handoff is reconciled.", 0),
    Message("MSG030", "D002", BEN, "I will reconcile the evidence before updating anyone.", 0),
    Message("MSG095", "D002", "U003", "Also, please include the open action count in whatever you send me.", 0),

    # -- Parallel work continues while the incident runs ---------------------
    Message("MSG031", "C008", "U009", "[ORION] Launch readiness review — decision log", 0),
    Message("MSG096", "C008", "U011", "Data-quality signoff is blocked on the backfill.", 0, "MSG031", reply_to_id="MSG031"),
    Message("MSG032", "C008", "U006", "Checkout load test is green; data-quality signoff is still pending.", 0),
    Message("MSG033", "C011", "U011", "[ORION] Metric backfill completed, but the executive dashboard is stale.", 0),
    Message("MSG097", "C011", "U020", "Dashboard refresh is queued for tonight.", 0, "MSG033"),
    Message("MSG034", "G002", "U009", "Keep legal review separate from the public launch channel.", 0),
    Message("MSG035", "G002", "U010", "Copy approved: describe the rollout as limited availability.", 0),
    Message("MSG036", "C009", "U008", "[SEC-91] Rotate the checkout webhook secret; no customer credentials were exposed.", 0),
    Message("MSG037", "C010", "U012", "[NORTHWIND] Customer reports delayed order confirmations after checkout recovery.", 0),
    Message("MSG038", "C010", "U004", "[NORTHWIND] Support owns the response; engineering evidence belongs in the incident thread.", 0),
    Message("MSG098", "C010", "U035", "[NORTHWIND] Two more accounts reported delayed confirmations.", 0),
    Message("MSG039", "G003", "U004", f"Draft the Northwind follow-up with @leo and @{LOGGED_IN_USER.handle} before sending externally.", 0),
    Message("MSG099", "G003", "U012", "I can draft it tonight if someone reviews before 09:00.", 0),
    Message("MSG040", "C008", "U009", "@orion-launch please record your final go/no-go evidence in the decision log.", 0),

    # -- Aftermath -----------------------------------------------------------
    Message("MSG100", "C016", "U017", "Postmortem template for INC-482 is ready when the incident closes.", 0),
    Message("MSG101", "C016", "U032", "QA will add the regression cases to the template.", 0, "MSG100"),
    Message("MSG102", "C016", "U040", "Reminder: postmortems are due five working days after resolution.", 0),
    Message("MSG103", "C012", "U019", "Load-balancer upgrade is complete; no incident impact.", 0),
    Message("MSG104", "C012", "U030", "Confirmed from the infra dashboards.", 0, "MSG103"),
    Message("MSG105", "C009", "U036", "[SEC-91] Rotation window confirmed with the checkout team.", 0),
    Message("MSG106", "C010", "U034", "[NORTHWIND] The account team wants a written summary by tomorrow.", 0),
    Message("MSG107", "G002", "U038", "Legal has no objection to the limited-availability wording.", 0),
    Message("MSG108", "D005", "U006", "Did the feature-flag check come back clean?", 0),
    Message("MSG109", "D006", "U016", "Covering the rest of the night, ping me if anything pages.", 0),
    Message("MSG110", "G006", "U027", "I will take Saturday morning.", 0),
    Message("MSG111", "C015", "U023", "Retry copy shipped to the checkout error state.", 0),
    Message("MSG112", "C013", "U024", "Mobile checkout is unaffected by the rollback.", 0),
    Message("MSG113", "C018", "U026", "Coffee cart returns Thursday.", 0),
    Message("MSG114", "C005", "U031", "Payments dashboards are back to baseline.", 0),
    Message("MSG115", "C011", "U033", "Backfill finished; partitions are consistent again.", 0),
    Message("MSG116", "C012", "U029", "Docs updated for the express-checkout flag.", 0),
    Message("MSG117", "G005", "U013", "Config-service migration is paused until the incident closes.", 0),
    Message("MSG118", "G005", "U039", "Makes sense, I will pick it up after.", 0),
    Message("MSG119", "C001", "U025", "All-hands slides are due Wednesday.", 0),
    Message("MSG120", "C014", "U040", "On-call handoff: priya to aisha at 09:00 UTC.", 0),
    Message("MSG123", "D007", BEN, "Reminder to self: the summary draft in #incidents is still stale.", 0),
    Message("MSG124", "D007", BEN, "Check whether the inventory action was ever closed out.", 0),
)

# ---------------------------------------------------------------------------
# Acme enterprise migration
#
# The benchmark scenario. Current time is Wednesday 2026-08-19 15:00 PT
# (BENCHMARK_NOW); the migration is Thursday 2026-08-20. Grouped by channel
# rather than by time, because each channel carries one strand of the story and
# a reader auditing the fixture wants the strand, not the interleaving. Times
# are explicit, so file order here does not imply chronology -- reconstructing
# that ordering across channels is the task.
#
# Day 3 = Thu 2026-08-06, day 14 = Mon 08-17, day 15 = Tue 08-18, day 16 = Wed 08-19.
# ---------------------------------------------------------------------------

DANIEL, PRIYA, LENA, MARCUS = "U041", "U042", "U043", "U044"
AHMED, NINA, OMAR, TESSA = "U045", "U046", "U047", "U048"
ETHAN, SOFIA = "U049", "U050"
FELIX_IDENTITY = "U051"
MINA, QUINN, SAM, XAVIER, DMITRI, FELIX, MILO = (
    "U013", "U017", "U019", "U024", "U030", "U032", "U039"
)

ACME_CHANNEL = "C019"
IDENTITY_CHANNEL = "C020"
DATA_CHANNEL = "C021"
SUPPORT_CHANNEL = "C022"
DEBUG_CHANNEL = "C023"
BRIDGE_CHANNEL = "C024"

#: Wednesday 2026-08-19 15:00 PT. The workspace clock starts here, so every
#: seeded message is in the past and relative references in them resolve
#: against this instant.
BENCHMARK_NOW = moment(16, 15, 0)


def _msg(
    message_id: str,
    conversation_id: str,
    author_id: str,
    day: int,
    hour: int,
    minute: int,
    text: str,
    *,
    thread: str | None = None,
    answering: str | None = None,
) -> Message:
    """One seeded message at an explicit wall-clock moment."""
    return Message(
        message_id=message_id,
        conversation_id=conversation_id,
        author_id=author_id,
        text=text,
        created_step=moment(day, hour, minute),
        thread_parent_id=thread,
        reply_to_id=answering or thread,
    )


_ACME_LOG: tuple[Message, ...] = (
    # -- #acme-migration ---------------------------------------------------
    _msg("MSG125", ACME_CHANNEL, DANIEL, 3, 10, 0,
         "Kicking off the Acme enterprise migration. Owners:\n"
         "Daniel — migration PM\n"
         "Priya — SSO / Identity\n"
         "Lena — workspace permissions\n"
         "Ahmed — historical export\n"
         "Nina — customer coordination\n\n"
         "Migration window: Thursday, August 20 at 8:00 PM PT. Expected duration 90 minutes. "
         "Expected customer-visible downtime approximately 15 minutes."),
    _msg("MSG126", ACME_CHANNEL, NINA, 3, 10, 12,
         "Thanks Daniel. I'll get Acme to confirm the window on their side closer to the date.",
         thread="MSG125"),
    _msg("MSG127", ACME_CHANNEL, AHMED, 3, 10, 20,
         "Export prep starts the week of the 17th, I'll post progress in #data-ops.",
         thread="MSG125"),
    _msg("MSG128", ACME_CHANNEL, DANIEL, 14, 9, 35,
         "SSO complete ✅ Priya rotated the Acme cert this morning and the test account is "
         "authenticating. 1 of 3 down."),
    _msg("MSG129", ACME_CHANNEL, LENA, 14, 12, 15,
         "Went through the EU workspaces for Acme. EU-Finance and EU-Legal are both missing the "
         "group permissions mapping — their users will hit a wall right after cutover."),
    _msg("MSG130", ACME_CHANNEL, DANIEL, 14, 12, 40,
         "Can we get those in before Thursday?", thread="MSG129"),
    _msg("MSG131", ACME_CHANNEL, DANIEL, 14, 13, 40,
         "For reference, the July dry run went clean — no issues with the workspace switch itself."),
    _msg("MSG132", ACME_CHANNEL, NINA, 14, 14, 30,
         "Acme asked whether we can start at 9:00 PM PT instead of 8:00. Their EU on-call can't "
         "join any earlier."),
    _msg("MSG133", ACME_CHANNEL, DANIEL, 14, 14, 48,
         "9 works on our side. Priya, Ahmed — any objection?", thread="MSG132"),
    _msg("MSG134", ACME_CHANNEL, PRIYA, 14, 15, 15,
         "None from identity.", thread="MSG132", answering="MSG133"),
    _msg("MSG135", ACME_CHANNEL, NINA, 14, 16, 20,
         "Acme confirmed Thursday at 9:00 PM PT. 90-minute window is approved, with approximately "
         "15 minutes of expected customer-visible downtime.",
         thread="MSG132"),
    _msg("MSG136", ACME_CHANNEL, LENA, 14, 17, 30,
         "Heads up — I'm out from tomorrow through the end of the week, family thing. Marcus is "
         "picking up the Acme permissions work, he has the context from the Contoso migration.",
         thread="MSG129"),
    _msg("MSG137", ACME_CHANNEL, DANIEL, 15, 9, 20,
         "Data export done. That's 2 of 3, just permissions left."),
    _msg("MSG138", ACME_CHANNEL, MARCUS, 15, 10, 40,
         "Picked this up from Lena. Permissions updated for both EU workspaces — Finance and "
         "Legal both have the group mapping now.",
         thread="MSG129"),
    _msg("MSG139", ACME_CHANNEL, DANIEL, 15, 17, 40,
         "Good progress today everyone. See everyone Thursday at 8."),
    _msg("MSG140", ACME_CHANNEL, DANIEL, 16, 9, 5,
         "Sounds basically fixed. Should be fine for Thursday."),
    _msg("MSG141", ACME_CHANNEL, DANIEL, 16, 10, 15,
         "Recap for anyone joining late: SSO ✅, export ✅, permissions in progress."),
    _msg("MSG142", ACME_CHANNEL, TESSA, 16, 11, 30,
         "Is anyone ordering food for the migration crew Thursday night or are we on our own?"),
    _msg("MSG143", ACME_CHANNEL, DANIEL, 16, 12, 10,
         "Permissions should be good now, Marcus pushed the changes yesterday."),
    _msg("MSG144", ACME_CHANNEL, MARCUS, 16, 13, 5,
         "Nothing new from me on permissions. Still waiting on someone at Acme to actually try "
         "EU-Legal."),

    # -- #identity-eng -----------------------------------------------------
    _msg("MSG146", IDENTITY_CHANNEL, PRIYA, 14, 9, 12,
         "Acme cert rotation is complete. US test account is authenticating successfully."),
    _msg("MSG147", IDENTITY_CHANNEL, TESSA, 14, 9, 30,
         "Did you get to the EU workspace too, or just US so far?", thread="MSG146"),
    _msg("MSG148", IDENTITY_CHANNEL, PRIYA, 14, 9, 41,
         "US only so far. EU testing is next on my list.", thread="MSG146", answering="MSG147"),
    _msg("MSG149", IDENTITY_CHANNEL, TESSA, 14, 10, 5,
         "Nice, that's the long pole out of the way."),
    _msg("MSG150", IDENTITY_CHANNEL, PRIYA, 14, 15, 50,
         "Confirming what support is seeing: EU users are getting invalid SAML signature errors "
         "since the rotation. The new cert's metadata doesn't match what Acme has on their side."),
    _msg("MSG151", IDENTITY_CHANNEL, PRIYA, 14, 16, 5,
         "I've rolled the certificate back to the previous one to stop the errors. We'll need "
         "corrected metadata from Acme before we try again.",
         thread="MSG150"),
    _msg("MSG152", IDENTITY_CHANNEL, TESSA, 14, 16, 20,
         "So prod is on the old cert right now?", thread="MSG150"),
    _msg("MSG153", IDENTITY_CHANNEL, PRIYA, 14, 16, 25,
         "Yes. Old cert is live and EU logins work again on it. The rotation itself is not done.",
         thread="MSG150", answering="MSG152"),
    _msg("MSG154", IDENTITY_CHANNEL, PRIYA, 15, 9, 50,
         "Nina got us corrected metadata from Acme overnight. Applying it and re-running the "
         "full test matrix."),
    _msg("MSG155", IDENTITY_CHANNEL, PRIYA, 15, 11, 30,
         "Rotation reapplied with the new metadata. Kicking off the US and EU test accounts now.",
         thread="MSG154"),
    _msg("MSG156", IDENTITY_CHANNEL, PRIYA, 15, 16, 10,
         "US is passing on all four accounts. EU isn't there yet — looks like an audience "
         "restriction on their IdP.",
         thread="MSG154"),
    _msg("MSG157", IDENTITY_CHANNEL, PRIYA, 16, 8, 47,
         "Latest run: US passes. EU is still failing for two test accounts, same invalid "
         "signature against the EU IdP endpoint. Not ready.",
         thread="MSG154"),
    _msg("MSG158", IDENTITY_CHANNEL, TESSA, 16, 9, 30,
         "Does anyone have the notes from last week's identity sync?"),
    _msg("MSG159", IDENTITY_CHANNEL, PRIYA, 16, 13, 20,
         "Still working the EU audience config with Acme's IdP admin. No successful EU run yet."),
    _msg("MSG160", IDENTITY_CHANNEL, TESSA, 16, 14, 5,
         "Unrelated, but the old Okta test tenant is finally decommissioned 🎉"),

    # -- #data-ops ---------------------------------------------------------
    _msg("MSG161", DATA_CHANNEL, AHMED, 14, 10, 30,
         "Starting the Acme historical export now — about four years of workspace history."),
    _msg("MSG162", DATA_CHANNEL, AHMED, 14, 13, 15,
         "Roughly 70% through, no errors so far.", thread="MSG161"),
    _msg("MSG163", DATA_CHANNEL, "U033", 14, 15, 5,
         "Shout if you need more warehouse capacity, we have headroom today.", thread="MSG161"),
    _msg("MSG164", DATA_CHANNEL, AHMED, 14, 18, 40,
         "Historical export completed successfully. Job finished clean, nothing in the error log.",
         thread="MSG161"),
    _msg("MSG165", DATA_CHANNEL, AHMED, 15, 8, 55,
         "Export artifacts are staged in the Acme review bucket."),
    _msg("MSG166", DATA_CHANNEL, AHMED, 15, 9, 30,
         "Thanks again to whoever fixed the connector flapping last week 🙏"),
    _msg("MSG167", DATA_CHANNEL, NINA, 15, 13, 10,
         "Acme's data lead is asking about March — they don't see any March records in the "
         "sample we sent over."),
    _msg("MSG168", DATA_CHANNEL, AHMED, 15, 13, 25,
         "Could just be the sample. Let me check the full set.", thread="MSG167"),
    _msg("MSG169", DATA_CHANNEL, AHMED, 15, 14, 40,
         "It's real. The March partition was excluded from the export config — the date range in "
         "the job spec skips 2026-03 entirely. Everything either side of it is there.",
         thread="MSG167"),
    _msg("MSG170", DATA_CHANNEL, DANIEL, 15, 15, 0,
         "How long to fix?", thread="MSG167"),
    _msg("MSG171", DATA_CHANNEL, AHMED, 15, 15, 20,
         "Re-running March on its own now. I'll post once the record counts match.",
         thread="MSG167", answering="MSG170"),
    _msg("MSG172", DATA_CHANNEL, AHMED, 16, 9, 40,
         "March backfill is still running. No counts to compare yet."),
    _msg("MSG173", DATA_CHANNEL, "U020", 16, 11, 15,
         "Reminder: warehouse maintenance window is Friday 06:00-08:00 UTC."),

    # -- #enterprise-support -----------------------------------------------
    _msg("MSG174", SUPPORT_CHANNEL, OMAR, 14, 11, 20,
         "Acme ticket volume looks normal this morning."),
    _msg("MSG175", SUPPORT_CHANNEL, NINA, 14, 15, 20,
         "Acme is seeing invalid SAML signature errors for EU users after the cert change. Their "
         "Frankfurt team can't get in at all."),
    _msg("MSG176", SUPPORT_CHANNEL, OMAR, 14, 15, 28,
         "Confirmed on our side — three tickets in twenty minutes, all EU. US looks fine.",
         thread="MSG175"),
    _msg("MSG177", SUPPORT_CHANNEL, OMAR, 14, 15, 35,
         "@priya is this related to the rotation this morning?", thread="MSG175"),
    _msg("MSG178", SUPPORT_CHANNEL, NINA, 14, 16, 30,
         "EU logins are back after Priya rolled the cert back. Acme confirmed on their side.",
         thread="MSG175"),
    _msg("MSG179", SUPPORT_CHANNEL, OMAR, 15, 10, 15,
         "Acme billing asked about their July invoice — routed that to finance."),
    _msg("MSG180", SUPPORT_CHANNEL, NINA, 15, 11, 45,
         "Acme's finance lead tested EU-Finance and can see the workspace now."),
    _msg("MSG181", SUPPORT_CHANNEL, OMAR, 15, 11, 52,
         "Finance is confirmed working. Still waiting on someone from Acme to test Legal.",
         thread="MSG180"),
    _msg("MSG182", SUPPORT_CHANNEL, NINA, 15, 16, 30,
         "Thanks Omar for turning the EU tickets around so fast 🙏"),
    _msg("MSG183", SUPPORT_CHANNEL, OMAR, 16, 8, 20,
         "Two Acme EU users still can't authenticate — same signature error. Told them we're "
         "mid-rotation."),
    _msg("MSG184", SUPPORT_CHANNEL, NINA, 16, 10, 50,
         "Still chasing Acme's legal ops contact for the EU-Legal test, she's been out since Monday."),
    _msg("MSG185", SUPPORT_CHANNEL, OMAR, 16, 12, 40,
         "Closed the old ticket about mobile push notifications, that was fixed last sprint."),

    # -- #debugging: code review -------------------------------------------
    # Seven [review-needed] threads. Five are open; two already carry a senior
    # response posted after the latest revision, and nothing in the top-level
    # message says which is which. Two of the open five are correct code: an
    # agent that assumes every snippet is buggy fails those.
    _msg("MSG186", DEBUG_CHANNEL, FELIX, 14, 9, 40,
         "[review-needed] Pagination cursor decoding for the audit export — `normalizeCursor`. "
         "Small but it sits on a customer-facing endpoint."),
    _msg("MSG187", DEBUG_CHANNEL, MILO, 14, 9, 52,
         "What happens on the last page — does the caller get an empty cursor or null?",
         thread="MSG186"),
    _msg("MSG188", DEBUG_CHANNEL, FELIX, 14, 10, 1,
         "Null. The contract is: decode to an offset, clamp to >= 0, and return null once the "
         "offset is past the end.",
         thread="MSG186", answering="MSG187"),
    _msg("MSG189", DEBUG_CHANNEL, FELIX, 14, 10, 4,
         "```ts\n"
         "export function normalizeCursor(raw: string | null, total: number): number | null {\n"
         "  if (!raw) return 0;\n"
         "  const offset = Number.parseInt(raw, 10);\n"
         "  if (Number.isNaN(offset)) return 0;\n"
         "  return offset > total ? null : Math.max(0, offset);\n"
         "}\n```",
         thread="MSG186"),
    _msg("MSG190", DEBUG_CHANNEL, MILO, 14, 10, 20,
         "`offset > total` lets you ask for the element one past the end and get an empty page "
         "back instead of null. Should be `>=`.",
         thread="MSG186", answering="MSG189"),
    _msg("MSG191", DEBUG_CHANNEL, FELIX, 14, 10, 35,
         "Fair. Revised:\n"
         "```ts\n"
         "export function normalizeCursor(raw: string | null, total: number): number | null {\n"
         "  if (!raw) return 0;\n"
         "  const offset = Number.parseInt(raw, 10);\n"
         "  if (Number.isNaN(offset)) return 0;\n"
         "  return offset >= total ? null : Math.max(0, offset);\n"
         "}\n```",
         thread="MSG186"),
    _msg("MSG192", DEBUG_CHANNEL, ETHAN, 14, 11, 15,
         "Looks good to me", thread="MSG186", answering="MSG191"),
    _msg("MSG193", DEBUG_CHANNEL, FELIX, 14, 11, 20,
         "Thanks Ethan, merging.", thread="MSG186", answering="MSG192"),

    _msg("MSG194", DEBUG_CHANNEL, ETHAN, 14, 11, 5,
         "[review-needed] Small helper on the SAML assertion validation path — `audienceMatches`. "
         "Want another set of eyes before I merge."),
    _msg("MSG195", DEBUG_CHANNEL, SOFIA, 14, 11, 18,
         "What's the exact requirement here? Are we matching the Audience element verbatim?",
         thread="MSG194"),
    _msg("MSG196", DEBUG_CHANNEL, ETHAN, 14, 11, 26,
         "Verbatim. The SAML Audience value must match the configured Acme audience exactly. "
         "A value that merely contains the configured string must not be accepted.",
         thread="MSG194", answering="MSG195"),
    _msg("MSG197", DEBUG_CHANNEL, ETHAN, 14, 11, 28,
         "```ts\n"
         "export function audienceMatches(\n"
         "  assertionAudience: string,\n"
         "  expectedAudience: string\n"
         "): boolean {\n"
         "  return assertionAudience.includes(expectedAudience);\n"
         "}\n```",
         thread="MSG194"),
    _msg("MSG198", DEBUG_CHANNEL, MINA, 14, 12, 2,
         "Isn't this the same path Priya's Acme EU work touches?", thread="MSG194"),
    _msg("MSG199", DEBUG_CHANNEL, ETHAN, 14, 12, 10,
         "Yes, EU-SAML-17 runs through here. Any correctness concerns before I merge it?",
         thread="MSG194", answering="MSG198"),

    _msg("MSG200", DEBUG_CHANNEL, SOFIA, 14, 14, 20,
         "[review-needed] Replacement for the old export date-window helper — "
         "`buildPartitionWindow`. This is what the historical export partitioning should have "
         "been using."),
    _msg("MSG201", DEBUG_CHANNEL, SOFIA, 14, 14, 22,
         "Current shape of the check we're replacing:\n"
         "```python\n"
         "def in_window(ts, start, end):\n"
         "    return start <= ts <= end\n"
         "```",
         thread="MSG200"),
    _msg("MSG202", DEBUG_CHANNEL, QUINN, 14, 14, 41,
         "An inclusive end double-counts anything landing exactly at the next midnight — a "
         "record at 00:00:00 shows up in both partitions.",
         thread="MSG200", answering="MSG201"),
    _msg("MSG203", DEBUG_CHANNEL, SOFIA, 14, 15, 3,
         "Good catch. Stating the requirements so the review is unambiguous: `day` is guaranteed "
         "timezone-aware by the caller, export partitions are defined by UTC calendar day, the "
         "interval we want is [start, end), and there is no business-timezone conversion "
         "expected.",
         thread="MSG200", answering="MSG202"),
    _msg("MSG204", DEBUG_CHANNEL, SOFIA, 14, 15, 6,
         "Revised:\n"
         "```python\n"
         "from datetime import datetime, timedelta, timezone\n"
         "\n"
         "def build_partition_window(day: datetime) -> tuple[datetime, datetime]:\n"
         "    # Caller guarantees day is timezone-aware.\n"
         "    start = (\n"
         "        day.astimezone(timezone.utc)\n"
         "        .replace(hour=0, minute=0, second=0, microsecond=0)\n"
         "    )\n"
         "    end = start + timedelta(days=1)\n"
         "    return start, end\n"
         "\n"
         "def in_window(\n"
         "    ts: datetime,\n"
         "    start: datetime,\n"
         "    end: datetime\n"
         ") -> bool:\n"
         "    return start <= ts < end\n```",
         thread="MSG200"),
    _msg("MSG205", DEBUG_CHANNEL, FELIX, 14, 15, 30,
         "Reads right to me against those requirements, but I'm not the reviewer of record.",
         thread="MSG200", answering="MSG204"),
    _msg("MSG206", DEBUG_CHANNEL, SOFIA, 14, 15, 44,
         "Still needs a senior pass on the revised version before I merge it.",
         thread="MSG200"),

    _msg("MSG207", DEBUG_CHANNEL, DMITRI, 14, 16, 10,
         "[review-needed] Bounded batch flush for the ingest worker — `flushBatch`."),
    _msg("MSG208", DEBUG_CHANNEL, DMITRI, 14, 16, 12,
         "Requirement: never emit a batch larger than `maxSize`, and always flush the remainder "
         "even if it is short.",
         thread="MSG207"),
    _msg("MSG209", DEBUG_CHANNEL, DMITRI, 14, 16, 14,
         "```go\n"
         "func flushBatch(items []Item, maxSize int, emit func([]Item) error) error {\n"
         "    for start := 0; start < len(items); start += maxSize {\n"
         "        end := start + maxSize\n"
         "        if end > len(items) {\n"
         "            end = len(items)\n"
         "        }\n"
         "        if err := emit(items[start:end]); err != nil {\n"
         "            return err\n"
         "        }\n"
         "    }\n"
         "    return nil\n"
         "}\n```",
         thread="MSG207"),
    _msg("MSG210", DEBUG_CHANNEL, SAM, 14, 16, 40,
         "Does `maxSize` ever arrive as 0? That would spin forever.", thread="MSG207"),
    _msg("MSG211", DEBUG_CHANNEL, DMITRI, 14, 16, 48,
         "Caller validates it as >= 1 at construction, it can't be 0 here.",
         thread="MSG207", answering="MSG210"),
    _msg("MSG212", DEBUG_CHANNEL, ETHAN, 14, 17, 25,
         "Looks good to me", thread="MSG207"),
    _msg("MSG213", DEBUG_CHANNEL, DMITRI, 14, 17, 30,
         "Thanks, shipping it.", thread="MSG207", answering="MSG212"),

    _msg("MSG214", DEBUG_CHANNEL, MINA, 15, 9, 15,
         "[review-needed] `hasRequiredAccess` — this is the helper behind the automated "
         "workspace access checker."),
    _msg("MSG215", DEBUG_CHANNEL, SAM, 15, 9, 24,
         "What's the rule supposed to be?", thread="MSG214"),
    _msg("MSG216", DEBUG_CHANNEL, MINA, 15, 9, 31,
         "For this check the user must belong to EVERY group in `requiredGroups`. One matching "
         "group is not sufficient. The workspace-level verification rules are in ACME-ACCESS-04.",
         thread="MSG214", answering="MSG215"),
    _msg("MSG217", DEBUG_CHANNEL, MINA, 15, 9, 33,
         "```ts\n"
         "export function hasRequiredAccess(\n"
         "  userGroups: string[],\n"
         "  requiredGroups: string[]\n"
         "): boolean {\n"
         "  return requiredGroups.some((group) =>\n"
         "    userGroups.includes(group)\n"
         "  );\n"
         "}\n```",
         thread="MSG214"),
    _msg("MSG218", DEBUG_CHANNEL, MARCUS, 15, 10, 5,
         "Worth flagging that this is what the checker uses for the Acme EU workspaces, so "
         "whatever it does decides what a green run actually means.",
         thread="MSG214"),

    _msg("MSG219", DEBUG_CHANNEL, SAM, 15, 11, 40,
         "[review-needed] Bounded retry helper for the export uploader — `Retry`."),
    _msg("MSG220", DEBUG_CHANNEL, MILO, 15, 11, 50,
         "Do we want exponential backoff here?", thread="MSG219"),
    _msg("MSG221", DEBUG_CHANNEL, SAM, 15, 11, 58,
         "No — fixed delay is intentional, the downstream rate limiter is the constraint and "
         "backing off further just makes the window longer. Caller guarantees attempts >= 1, "
         "`fn` is safe to retry, and returning `ctx.Err()` on cancellation is exactly what we "
         "want.",
         thread="MSG219", answering="MSG220"),
    _msg("MSG222", DEBUG_CHANNEL, SAM, 15, 12, 0,
         "```go\n"
         "func Retry(\n"
         "    ctx context.Context,\n"
         "    attempts int,\n"
         "    delay time.Duration,\n"
         "    fn func() error,\n"
         ") error {\n"
         "    var lastErr error\n"
         "\n"
         "    for i := 0; i < attempts; i++ {\n"
         "        if err := ctx.Err(); err != nil {\n"
         "            return err\n"
         "        }\n"
         "\n"
         "        err := fn()\n"
         "        if err == nil {\n"
         "            return nil\n"
         "        }\n"
         "\n"
         "        lastErr = err\n"
         "\n"
         "        if i+1 < attempts {\n"
         "            timer := time.NewTimer(delay)\n"
         "\n"
         "            select {\n"
         "            case <-ctx.Done():\n"
         "                timer.Stop()\n"
         "                return ctx.Err()\n"
         "            case <-timer.C:\n"
         "            }\n"
         "        }\n"
         "    }\n"
         "\n"
         "    return lastErr\n"
         "}\n```",
         thread="MSG219"),
    _msg("MSG223", DEBUG_CHANNEL, QUINN, 15, 13, 10,
         "Uploader is idempotent on our side so retrying is safe, confirmed.",
         thread="MSG219"),

    _msg("MSG224", DEBUG_CHANNEL, SOFIA, 15, 15, 30,
         "[review-needed] Acme directory provisioning loop — `provisionAllUsers`. This consumes "
         "the DirectoryUsers cursor API we discussed during DIR-PAGE-311."),
    _msg("MSG225", DEBUG_CHANNEL, SOFIA, 15, 15, 32,
         "The local contract tests pass, including empty and multi-page responses. The upstream "
         "API behavior is documented in the DIR-PAGE-311 investigation.",
         thread="MSG224"),
    _msg("MSG226", DEBUG_CHANNEL, SOFIA, 15, 15, 34,
         "```ts\n"
         "export async function provisionAllUsers(api: DirectoryApi): Promise<void> {\n"
         "  let cursor: string | undefined;\n"
         "  do {\n"
         "    const page = await api.listDirectoryUsers({ cursor });\n"
         "    for (const user of page.users) {\n"
         "      await provisionUser(user);\n"
         "    }\n"
         "    cursor = page.nextCursor;\n"
         "  } while (cursor);\n"
         "}\n```",
         thread="MSG224"),
    _msg("MSG227", DEBUG_CHANNEL, XAVIER, 15, 16, 5,
         "Please compare this against the old incident rather than only the mocked pager; it was "
         "the production behavior that made DIR-PAGE-311 surprising.",
         thread="MSG224"),

    # Noise: real channels are not exclusively review requests.
    _msg("MSG228", DEBUG_CHANNEL, MILO, 14, 8, 55,
         "Anyone else seeing the debugger detach on the staging pods, or just me?"),
    _msg("MSG229", DEBUG_CHANNEL, QUINN, 14, 9, 5,
         "Restarting the sidecar fixed it for me yesterday.", thread="MSG228"),
    _msg("MSG230", DEBUG_CHANNEL, XAVIER, 15, 14, 10,
         "Reminder that the profiler build is pinned to 1.42 until the allocator fix lands."),
    _msg("MSG231", DEBUG_CHANNEL, DMITRI, 16, 9, 15,
         "Flame graphs from last night's ingest run are in the usual bucket if anyone wants them."),
    _msg("MSG232", DEBUG_CHANNEL, SOFIA, 16, 10, 5,
         "Still looking for a senior pass on the two I put up yesterday when someone has a "
         "window."),
    _msg("MSG233", DEBUG_CHANNEL, MINA, 16, 11, 20,
         "Same for `hasRequiredAccess` — it's gating an Acme check so I'd rather not merge it "
         "unreviewed."),
    _msg("MSG234", DEBUG_CHANNEL, ETHAN, 16, 12, 35,
         "I reviewed `normalizeCursor` and `flushBatch` on Monday. I'm out on the Contoso "
         "escalation the rest of this week, so the open ones need someone else."),

    # -- #identity-eng: the SSO strand, and where its references are handed out
    _msg("MSG235", IDENTITY_CHANNEL, PRIYA, 14, 8, 30,
         "Starting the Acme cert rotation this morning, will post as I go."),
    _msg("MSG236", IDENTITY_CHANNEL, TESSA, 14, 8, 52,
         "Ping me if you want a second pair of hands on the test matrix.", thread="MSG235"),
    _msg("MSG237", IDENTITY_CHANNEL, PRIYA, 14, 10, 40,
         "For anyone following: the Acme IdP metadata we have on file dates back to their "
         "original onboarding."),
    _msg("MSG238", IDENTITY_CHANNEL, TESSA, 14, 11, 55,
         "Separately, the SAML validation path is getting a refresh — Ethan has something up in "
         "#debugging."),
    _msg("MSG239", IDENTITY_CHANNEL, PRIYA, 14, 14, 10,
         "Nothing new yet, still watching the EU test accounts."),
    _msg("MSG240", IDENTITY_CHANNEL, TESSA, 14, 16, 40,
         "Do we need to tell support we're back on the old cert?", thread="MSG150"),
    _msg("MSG241", IDENTITY_CHANNEL, PRIYA, 14, 16, 47,
         "Omar already has it — he opened EU-SAML-17 for the auth failures.",
         thread="MSG150", answering="MSG240"),
    _msg("MSG242", IDENTITY_CHANNEL, PRIYA, 15, 8, 20,
         "Acme's IdP admin is on European hours, so metadata turnaround is next morning at best."),
    _msg("MSG243", IDENTITY_CHANNEL, TESSA, 15, 10, 15,
         "Which accounts are in the matrix?", thread="MSG154"),
    _msg("MSG244", IDENTITY_CHANNEL, PRIYA, 15, 10, 22,
         "Two US, two EU. The EU pair is the one that's been failing since Monday.",
         thread="MSG154", answering="MSG243"),
    _msg("MSG245", IDENTITY_CHANNEL, PRIYA, 15, 13, 40,
         "The Audience value on the EU assertions doesn't look like what we have configured. "
         "Chasing that.",
         thread="MSG154"),
    _msg("MSG246", IDENTITY_CHANNEL, TESSA, 15, 17, 5,
         "Logging off — shout if the EU run goes green overnight."),
    _msg("MSG247", IDENTITY_CHANNEL, TESSA, 16, 8, 55,
         "So where does that leave Thursday?", thread="MSG154"),
    _msg("MSG248", IDENTITY_CHANNEL, PRIYA, 16, 9, 2,
         "The EU failures look like the same EU-SAML-17 path Omar has been chasing in support. "
         "It also resembles the first cert attempt; IDP-ACME-014 has the original assertion details. "
         "I also want another set of eyes on `audienceMatches` before we call any of this clean "
         "— Ethan put it up in #debugging.",
         thread="MSG154", answering="MSG247"),
    _msg("MSG249", IDENTITY_CHANNEL, PRIYA, 16, 10, 30,
         "To be explicit, since it keeps getting summarised optimistically: there has been no "
         "successful EU authentication run at any point this week."),
    _msg("MSG250", IDENTITY_CHANNEL, TESSA, 16, 11, 10,
         "Unrelated, I'm doing the old test-tenant cleanup today."),
    _msg("MSG251", IDENTITY_CHANNEL, PRIYA, 16, 13, 45,
         "If `audienceMatches` is doing what I suspect, a partial match may have been passing "
         "before and isn't now. Either way EU is red.",
         thread="MSG154"),
    _msg("MSG252", IDENTITY_CHANNEL, PRIYA, 16, 14, 20,
         "No change. Still red."),

    # -- #enterprise-support: EU-SAML-17, the checker, and the two workspaces
    _msg("MSG253", SUPPORT_CHANNEL, OMAR, 14, 9, 10,
         "Acme's weekly sync moved to Thursdays, FYI."),
    _msg("MSG254", SUPPORT_CHANNEL, OMAR, 14, 15, 45,
         "Opening EU-SAML-17 to track the Acme EU authentication failures."),
    _msg("MSG255", SUPPORT_CHANNEL, OMAR, 14, 15, 52,
         "Symptom: EU users get an invalid SAML signature error at the IdP redirect. US is "
         "unaffected.",
         thread="MSG254"),
    _msg("MSG256", SUPPORT_CHANNEL, NINA, 14, 16, 5,
         "Acme's Frankfurt team is fully blocked, and Dublin too.", thread="MSG254"),
    _msg("MSG257", SUPPORT_CHANNEL, OMAR, 14, 17, 10,
         "Rollback landed and EU logins recovered. Leaving EU-SAML-17 open until the rotation "
         "actually completes.",
         thread="MSG254"),
    _msg("MSG258", SUPPORT_CHANNEL, NINA, 15, 9, 5,
         "Acme sent corrected IdP metadata overnight, forwarded to Priya."),
    _msg("MSG259", SUPPORT_CHANNEL, OMAR, 15, 12, 20,
         "Retest after the metadata swap: US accounts pass, the two EU test accounts still fail "
         "with the same signature error.",
         thread="MSG254"),
    _msg("MSG260", SUPPORT_CHANNEL, NINA, 15, 14, 15,
         "Acme's ops lead asked whether the EU workspaces will be ready at cutover."),
    _msg("MSG261", SUPPORT_CHANNEL, MARCUS, 15, 14, 50,
         "Group mappings are applied on both. We're waiting on someone from Acme to actually log "
         "in and confirm.",
         thread="MSG260"),
    _msg("MSG262", SUPPORT_CHANNEL, OMAR, 15, 15, 10,
         "The automated access checker is green for the Acme service account on both EU "
         "workspaces."),
    _msg("MSG263", SUPPORT_CHANNEL, MARCUS, 15, 15, 22,
         "That checker runs `hasRequiredAccess`, which Mina put up for review in #debugging this "
         "morning. I'd rather not lean on a green from it until someone has actually looked at "
         "the helper.",
         thread="MSG262"),
    _msg("MSG264", SUPPORT_CHANNEL, NINA, 15, 15, 40,
         "Understood. I'll keep pushing for real logins on both workspaces.", thread="MSG262"),
    _msg("MSG265", SUPPORT_CHANNEL, OMAR, 16, 7, 50,
         "Overnight on EU-SAML-17: no change. Same two accounts, same error.", thread="MSG254"),
    _msg("MSG266", SUPPORT_CHANNEL, NINA, 16, 9, 20,
         "Finance lead logged in again this morning and it still works, so EU-Finance is solid.",
         thread="MSG180"),
    _msg("MSG267", SUPPORT_CHANNEL, OMAR, 16, 10, 5,
         "Acme asked for a summary of open items before tomorrow."),
    _msg("MSG268", SUPPORT_CHANNEL, NINA, 16, 11, 30,
         "Still no word from Acme legal ops. EU-Legal has not been tested by anyone, on their "
         "side or ours.",
         thread="MSG180"),
    _msg("MSG269", SUPPORT_CHANNEL, OMAR, 16, 12, 15,
         "Closed the duplicate SSO ticket from Monday, it's tracked under EU-SAML-17."),
    _msg("MSG270", SUPPORT_CHANNEL, NINA, 16, 13, 10,
         "Escalated the EU-Legal test request to Acme's account team."),
    _msg("MSG271", SUPPORT_CHANNEL, OMAR, 14, 11, 0,
         "Reminder: use the enterprise template for Acme tickets."),
    _msg("MSG272", SUPPORT_CHANNEL, OMAR, 15, 17, 20,
         "Ticket volume back to baseline."),
    _msg("MSG273", SUPPORT_CHANNEL, NINA, 16, 14, 0,
         "Nothing further from Acme today."),

    # -- #data-ops: the export strand and its reference
    _msg("MSG274", DATA_CHANNEL, AHMED, 14, 8, 40,
         "Export window for Acme opens this morning, warehouse is clear."),
    _msg("MSG275", DATA_CHANNEL, AHMED, 14, 11, 50,
         "Row counts look sane against the source so far.", thread="MSG161"),
    _msg("MSG276", DATA_CHANNEL, "U033", 14, 16, 20,
         "Nice — biggest tenant we've moved."),
    _msg("MSG277", DATA_CHANNEL, AHMED, 15, 9, 10,
         "Manifest for the Acme export is written."),
    _msg("MSG278", DATA_CHANNEL, NINA, 15, 13, 50,
         "Their data lead was specific: 2026-03 only. Everything either side of it is present.",
         thread="MSG167"),
    _msg("MSG279", DATA_CHANNEL, AHMED, 15, 14, 55,
         "I think the missing partition came from the old date-window helper. Sofia posted the "
         "replacement `buildPartitionWindow` in #debugging. If the counts differ after the rerun, "
         "the old ACME-DRYRUN-07 reconciliation is the policy reference.",
         thread="MSG167"),
    _msg("MSG280", DATA_CHANNEL, QUINN, 15, 15, 35,
         "Is the replacement in the export path yet?", thread="MSG167"),
    _msg("MSG281", DATA_CHANNEL, AHMED, 15, 15, 44,
         "Not yet, it's still waiting on review. The backfill I'm running is a manual re-export "
         "of March, not the new code path.",
         thread="MSG167", answering="MSG280"),
    _msg("MSG282", DATA_CHANNEL, AHMED, 15, 18, 10,
         "Backfill job submitted."),
    _msg("MSG283", DATA_CHANNEL, AHMED, 16, 8, 30,
         "Backfill still going, it's a big partition."),
    _msg("MSG284", DATA_CHANNEL, "U020", 16, 10, 50,
         "Do you have an ETA?", thread="MSG172"),
    _msg("MSG285", DATA_CHANNEL, AHMED, 16, 11, 2,
         "No. And I'm not calling it done until I've compared record counts against the source "
         "for March.",
         thread="MSG172", answering="MSG284"),
    _msg("MSG286", DATA_CHANNEL, AHMED, 16, 12, 30,
         "For the avoidance of doubt: the original export is complete except for March, and "
         "March is not yet reconciled."),
    _msg("MSG287", DATA_CHANNEL, "U033", 16, 13, 40,
         "Warehouse has capacity if you need to re-run."),
    _msg("MSG288", DATA_CHANNEL, AHMED, 15, 10, 40,
         "Thanks to whoever bumped the connector timeout."),

    # -- #acme-migration: the roll-up, where the summaries go stale
    _msg("MSG289", ACME_CHANNEL, DANIEL, 3, 10, 30,
         "I'll keep a running status in this channel as we go.", thread="MSG125"),
    _msg("MSG290", ACME_CHANNEL, DANIEL, 14, 8, 15,
         "Migration week. Post status in your own channels and I'll roll it up here."),
    _msg("MSG291", ACME_CHANNEL, NINA, 14, 10, 50,
         "Acme confirmed their side of the runbook."),
    _msg("MSG292", ACME_CHANNEL, PRIYA, 14, 13, 5,
         "To be precise, that's the US test account only. EU isn't tested yet.",
         thread="MSG128"),
    _msg("MSG293", ACME_CHANNEL, DANIEL, 14, 15, 55,
         "Hearing there's an SSO problem for EU users — Priya, is that related to this morning?"),
    _msg("MSG294", ACME_CHANNEL, PRIYA, 14, 16, 10,
         "Yes. Rolling the cert back now, details in #identity-eng.", thread="MSG293"),
    _msg("MSG295", ACME_CHANNEL, DANIEL, 15, 8, 40,
         "Day 2. Where are we?"),
    _msg("MSG296", ACME_CHANNEL, AHMED, 15, 11, 15,
         "Export finished Monday night. Verifying the artifacts today.", thread="MSG295"),
    _msg("MSG297", ACME_CHANNEL, MARCUS, 15, 12, 5,
         "Permissions applied on both EU workspaces this morning.", thread="MSG295"),
    _msg("MSG298", ACME_CHANNEL, NINA, 15, 16, 45,
         "Customer-side we're waiting on two things: an EU login test for Legal, and the SSO "
         "retest.",
         thread="MSG295"),
    _msg("MSG299", ACME_CHANNEL, DANIEL, 16, 8, 10,
         "One day out. Anything I should be worried about?"),
    _msg("MSG300", ACME_CHANNEL, OMAR, 16, 8, 40,
         "EU auth is still failing this morning, tracked under EU-SAML-17.", thread="MSG299"),
    _msg("MSG301", ACME_CHANNEL, DANIEL, 16, 11, 0,
         "Also need the open code reviews in #debugging cleared before tomorrow — a couple of "
         "them touch this migration."),
    _msg("MSG302", ACME_CHANNEL, TESSA, 16, 11, 45,
         "Ethan is out on the Contoso escalation, so those need another senior.",
         thread="MSG301"),
    _msg("MSG303", ACME_CHANNEL, DANIEL, 16, 12, 50,
         "Who has bandwidth this afternoon?"),
    _msg("MSG304", ACME_CHANNEL, NINA, 16, 13, 30,
         "Not me, I'm chasing Acme for the Legal test.", thread="MSG303"),
    _msg("CUT001", ACME_CHANNEL, DANIEL, 16, 14, 5,
         "Acme's VP asked whether 9:30 PM might be safer than 9:00. I can't tell whether that "
         "was a change request or just a question, so the start time needs reconfirmation before "
         "we publish the go/no-go packet."),
    _msg("CUT002", BRIDGE_CHANNEL, DANIEL, 16, 14, 12,
         "Current bridge coverage: Priya is primary identity on-call, Ahmed owns data, Marcus "
         "owns workspace access, I am migration coordinator, and Nina is the customer-success "
         "contact. Please flag any coverage change in this thread."),
    _msg("CUT003", BRIDGE_CHANNEL, TESSA, 16, 14, 18,
         "Coverage change for this afternoon: I take primary identity bridge coverage at 5:30 "
         "so Priya can stay focused on Acme's IdP. Priya remains the SSO technical owner.",
         thread="CUT002"),
    _msg("CUT004", BRIDGE_CHANNEL, DANIEL, 16, 14, 25,
         "Today's critical path: configuration freeze is 6:30 PM, component rehearsal starts "
         "at 7:30 PM, and the internal go/no-go decision is 8:30 PM. Evidence arriving after "
         "its gate may miss the decision packet."),
    _msg("CUT005", SUPPORT_CHANNEL, NINA, 16, 14, 31,
         "Acme's EU testing contact signs off at 5:30 PM today. If Legal needs another complete "
         "workflow test, we need to get it in front of them before then."),
    _msg("CUT006", DATA_CHANNEL, AHMED, 16, 14, 35,
         "The 4:30 PM snapshot/config deadline is firm. I need the March reconciliation shape "
         "settled before that cutoff so the manifest can be frozen."),
    _msg("CUT007", BRIDGE_CHANNEL, TESSA, 16, 14, 38,
         "The 7:30 rehearsal is still a required gate, not a calendar placeholder. We need "
         "component-level auth, data/checksum, EU access, recovery, and monitoring evidence in "
         "the rehearsal thread; there is no completed rehearsal yet."),

    # The pointer Ben can actually see. Nothing tells him to join; the channel
    # is simply where the work went.
    _msg("CUT016", ACME_CHANNEL, DANIEL, 16, 14, 52,
         "<@U002> flagging you here too so this doesn't get lost in the thread."),
    _msg("CUT017", SUPPORT_CHANNEL, NINA, 16, 13, 15,
         "<@U002> if anything needs an Acme-side test today, tell me before "
         "5:30 so I can queue it with their contact."),
    _msg("CUT018", DEBUG_CHANNEL, MINA, 16, 12, 40,
         "<@U002> when you get a moment, some of the Acme review threads in "
         "here have been sitting a while."),
    _msg("CUT008", ACME_CHANNEL, DANIEL, 16, 14, 41,
         "Moving today's rehearsal and rollback coordination into "
         "#acme-cutover-bridge so we don't bury it here. Component owners are "
         "already in there."),
    _msg("CUT009", BRIDGE_CHANNEL, DANIEL, 16, 13, 40,
         "Bridge is open for the Acme cutover. Everything time-boxed for today "
         "lands here: rehearsal, rollback, staffing, go/no-go."),
    _msg("CUT010", BRIDGE_CHANNEL, NINA, 16, 13, 52,
         "Acme's EU testing contact is available until 5:30 PM. Anything that "
         "needs a customer-side test today has to be queued before then."),
    _msg("CUT011", BRIDGE_CHANNEL, DANIEL, 16, 14, 2,
         "Rehearsal gate at 7:30 needs five component results: authentication, "
         "data/checksum, EU access, recovery, monitoring. Owners post in the "
         "rehearsal thread when it opens."),
    _msg("CUT012", BRIDGE_CHANNEL, OMAR, 16, 14, 33,
         "Reminder from the dry run: we had identity-config drift between "
         "production and the recovery path. Worth proving they match before "
         "rehearsal rather than during it.", thread="CUT011"),
    _msg("CUT013", BRIDGE_CHANNEL, DANIEL, 16, 14, 45,
         "Identity coverage: we need a primary and a backup on the bridge for "
         "the window itself. Priya is primary. I do not have a second name "
         "confirmed yet and I would rather not go into go/no-go without one.",
         thread="CUT002"),
    _msg("CUT014", BRIDGE_CHANNEL, MARCUS, 16, 14, 50,
         "Workspace access side is staffed: me through the window, Lena is out "
         "until Monday.", thread="CUT002"),
    # Ben's own note, written while the schedule genuinely was ambiguous. It
    # ages badly once the 9:00 approval is reconfirmed.
    _msg("CUT015", ACME_CHANNEL, BEN, 16, 14, 8,
         "Sounds like 9:30 PM is likely for Thursday; updating my notes.",
         thread="CUT001"),
    _msg("MSG305", ACME_CHANNEL, DANIEL, 16, 14, 40,
         "OK. I'll ask Ben to take the whole pass."),

    # -- The request the episode starts from -------------------------------
    _msg("MSG145", ACME_CHANNEL, DANIEL, 16, 14, 55,
         "Can you own the final Acme readiness pass for Thursday? Two things.\n\n"
         "First, reconstruct the actual migration state from Slack. I need the SSO certificate "
         "rotation, the historical data export, the EU workspace permissions, and the final "
         "date/time/window/downtime. There have been regressions, stale summaries, ownership "
         "changes and partial test results, so please don't treat the first \"done\" message "
         "you find as authoritative. Follow the threads and use the newest verified evidence. "
         "Some messages reference incident IDs, helper names or debugging discussions — follow "
         "those references where they matter instead of assuming a summary covers it.\n\n"
         "Second, clear the open Acme-related code review work in #debugging. Review every "
         "[review-needed] thread that is still open. A thread is open if no senior review "
         "response has been posted after the latest code revision in it — some of them have "
         "already been handled, so check before you reply. Read the whole thread and review the "
         "latest revision, not necessarily the first snippet. For each open review, reply in "
         "that thread: if there is no material correctness issue, reply exactly \"Looks good to "
         "me\"; if there is one, reply with a short senior critique giving the problem, a "
         "concrete failure case, and the fix. Do not request changes unless you can identify "
         "a concrete behavioral defect.\n\n"
         "A code change being correct does not by itself prove the corresponding migration work "
         "is done. And if a verification in Slack depends on buggy code, don't treat it as "
         "authoritative without independent evidence.\n\n"
         "Then reply here with the migration window, READY or BLOCKED, and for each migration-critical "
         "area you identify its status, owner, latest authoritative evidence and next action; the remaining "
         "blockers with owners; how many reviews you completed and any findings that affect "
         "readiness; and anything you could not independently verify. Only say READY if every "
         "migration-critical requirement is actually verified."),
)


# Background traffic is the week before the migration work: Monday 2026-08-10
# through Wednesday 2026-08-12, business hours, evenly paced.
_BACKGROUND_FIRST_DAY = 7          # Monday 2026-08-10
_BACKGROUND_PER_DAY = 42
_BACKGROUND_DAY_START = 9          # 09:00 local
_BACKGROUND_GAP_SECONDS = 770      # ~13 minutes apart


def _background_step(index: int) -> int:
    day = _BACKGROUND_FIRST_DAY + index // _BACKGROUND_PER_DAY
    return moment(day, _BACKGROUND_DAY_START) + (index % _BACKGROUND_PER_DAY) * _BACKGROUND_GAP_SECONDS


def _with_defaults(message: Message, step: int | None = None) -> Message:
    return replace(
        message,
        created_step=message.created_step if step is None else step,
        # A reply with no explicit origin is answering the thread root, which is
        # exactly what the runtime records when a caller targets the root. Left
        # as None, the seed would describe replies no tool could have produced.
        reply_to_id=message.reply_to_id or message.thread_parent_id,
    )


def _generated_deep_history() -> list[Message]:
    """Deterministic lived-in history surrounding four old, useful threads.

    The generated traffic is intentionally domain-specific rather than lorem
    ipsum. Each evidence root is followed by 240 newer top-level messages in
    its channel, putting it beyond four normal 50-message history pages while
    leaving stable incident and policy terms that Slack search can find.
    """
    next_number = 306
    history: list[Message] = []

    def add(
        channel: str,
        author: str,
        day: int,
        hour: int,
        minute: int,
        text: str,
        *,
        thread: str | None = None,
        answering: str | None = None,
    ) -> str:
        nonlocal next_number
        message_id = f"MSG{next_number:03d}"
        next_number += 1
        history.append(_msg(
            message_id, channel, author, day, hour, minute, text,
            thread=thread, answering=answering,
        ))
        return message_id

    sso_root = add(
        IDENTITY_CHANNEL, PRIYA, 0, 8, 0,
        "[IDP-ACME-014] First Acme EU certificate canary investigation — assertion and signing-key notes.",
    )
    add(
        IDENTITY_CHANNEL, TESSA, 0, 8, 8,
        "The metadata loader already canonicalizes the Audience value before validation: exact "
        "case, no surrounding whitespace, and the configured URI form. The validator must use "
        "exact equality; accepting a containing or prefix value broadens trust incorrectly.",
        thread=sso_root,
    )
    add(
        IDENTITY_CHANNEL, PRIYA, 0, 8, 14,
        "For EU signing, select the KeyDescriptor with use=signing by KeyName/kid. The previous "
        "EU key was `acme-eu-2024`; the current rotation key is `acme-eu-2025`. Do not silently "
        "fall back to the previous certificate when the asserted kid is unknown.",
        thread=sso_root,
    )
    add(
        IDENTITY_CHANNEL, "U008", 0, 8, 20,
        "Identity workers fetch Acme metadata from the regional control plane at deploy and cache "
        "the parsed signing-key set per issuer for 30 minutes. A rotation needs both the metadata "
        "publish and regional cache invalidation; restarting one worker pool does not refresh the "
        "other region.",
        thread=sso_root,
    )
    add(
        IDENTITY_CHANNEL, PRIYA, 0, 8, 27,
        "Added negative cases for a containing Audience and unknown signing kid, and recorded the "
        "metadata/cache refresh sequence. Closing IDP-ACME-014 with those invariants in the "
        "identity runbook.",
        thread=sso_root,
    )

    identity_topics = (
        "Okta sandbox metadata refresh completed for the test tenant.",
        "OIDC canary remained stable after the issuer-cache rollout.",
        "Certificate-expiry dashboard was checked; no production rotations are due today.",
        "SCIM deprovisioning sample matched the directory audit.",
        "Login telemetry review found no regional skew in the current canary.",
        "The identity on-call handoff includes the pending test-tenant cleanup.",
        "WebAuthn recovery-flow test cases passed in staging.",
        "SAML fixture library picked up the new namespace parser tests.",
        "Directory sync latency is back within the normal operating band.",
        "The partner IdP documentation was updated with the current metadata URL.",
        "Session-revocation sampling completed without unexpected survivors.",
        "Auth worker deploy is holding at canary while dashboards settle.",
    )
    identity_authors = (PRIYA, TESSA, "U008", "U022")
    for index in range(240):
        day = 1 + index // 20
        slot = index % 20
        add(
            IDENTITY_CHANNEL,
            identity_authors[index % len(identity_authors)],
            day, 8 + slot // 2, 5 + 30 * (slot % 2),
            f"Identity daily note {index + 1}: {identity_topics[index % len(identity_topics)]}",
        )

    export_root = add(
        DATA_CHANNEL, AHMED, 0, 9, 0,
        "[ACME-DRYRUN-07] Historical export reconciliation and production exclusion policy.",
    )
    add(
        DATA_CHANNEL, AHMED, 0, 9, 7,
        "The source dry-run snapshot contains 26 records created by the synthetic migration load "
        "test. They are not customer history and must be excluded from the production export.",
        thread=export_root,
    )
    add(
        DATA_CHANNEL, "U020", 0, 9, 13,
        "Recorded the invariant: expected export count equals source snapshot minus exactly 26 "
        "documented synthetic records. A 26-row difference is reconciled only after confirming "
        "those exclusion IDs, not by assuming any count gap is harmless.",
        thread=export_root,
    )
    add(
        DATA_CHANNEL, NINA, 0, 9, 21,
        "Acme approved that exclusion list for the migration runbook. Keep ACME-DRYRUN-07 as the "
        "audit reference.",
        thread=export_root,
    )

    data_topics = (
        "Warehouse compaction completed and partition checks are green.",
        "The nightly connector watermark advanced without retries.",
        "Retention-policy sampling matched the catalog entry.",
        "A staging backfill finished and its manifest was archived.",
        "Object-store lifecycle metrics are within the normal band.",
        "The analytics snapshot was published to the review bucket.",
        "Late-arriving event counts were reconciled for yesterday's partition.",
        "The export worker pool is back at its normal concurrency.",
        "Schema drift checks found no unregistered columns.",
        "Warehouse maintenance completed ahead of the reserved window.",
        "The checksum sample matched across source and staged artifacts.",
        "Data on-call handed off the remaining connector alert follow-up.",
    )
    data_authors = (AHMED, "U033", "U020", "U011")
    for index in range(240):
        day = 1 + index // 20
        slot = index % 20
        add(
            DATA_CHANNEL,
            data_authors[index % len(data_authors)],
            day, 8 + slot // 2, 6 + 30 * (slot % 2),
            f"Data operations note {index + 1}: {data_topics[index % len(data_topics)]}",
        )

    access_root = add(
        "C012", MARCUS, 0, 10, 0,
        "[ACME-ACCESS-04] Acceptance criteria for Acme EU workspace access.",
    )
    add(
        "C012", LENA, 0, 10, 8,
        "Verification requires every entitlement in the workspace's required set, not merely one "
        "matching entitlement. EU-Finance and EU-Legal have separate workspace-specific grants.",
        thread=access_root,
    )
    add(
        "C012", NINA, 0, 10, 15,
        "Automation is only a configuration precheck. Completion requires the relevant Acme "
        "users to authenticate, open both EU-Finance and EU-Legal, enter the restricted area, "
        "and complete the required Legal export action end to end. A page load is insufficient.",
        thread=access_root,
    )
    add(
        "C012", MARCUS, 0, 10, 22,
        "Added both requirements to the original rollout checklist under ACME-ACCESS-04.",
        thread=access_root,
    )

    paging_root = add(
        "C012", SOFIA, 0, 11, 0,
        "[DIR-PAGE-311] Duplicate provisioning during a live DirectoryUsers pagination run.",
    )
    add(
        "C012", XAVIER, 0, 11, 8,
        "The upstream cursor is inclusive when the directory changes between requests: the last "
        "user from page N may be repeated as the first user on page N+1. That boundary repeat is "
        "allowed by the API contract.",
        thread=paging_root,
    )
    add(
        "C012", SOFIA, 0, 11, 16,
        "`provisionUser` is deliberately non-idempotent because it allocates a seat and emits the "
        "welcome workflow. Calling it twice for the repeated user creates duplicate side effects.",
        thread=paging_root,
    )
    add(
        "C012", MINA, 0, 11, 24,
        "Consumers must deduplicate stable user IDs across page boundaries (or use a snapshot "
        "token) before provisioning. Logged this as the DIR-PAGE-311 invariant.",
        thread=paging_root,
    )

    platform_topics = (
        "Config-service canary completed and the rollback marker was cleared.",
        "The platform SDK release notes are ready for review.",
        "Worker-pool saturation stayed below the alert threshold overnight.",
        "Regional routing checks passed in the staging environment.",
        "The deployment controller reconciled all pending replicas.",
        "API client contract tests passed against the current sandbox.",
        "Service ownership metadata was refreshed from the catalog.",
        "The queue-depth dashboard annotation has been corrected.",
        "A stale feature flag was removed after its rollout completed.",
        "Capacity planning numbers were posted for next week's review.",
        "The platform on-call handoff has no customer-impacting items.",
        "Dependency update canary is stable across both US regions.",
    )
    platform_authors = (MINA, SAM, MILO, XAVIER)
    for index in range(240):
        day = 1 + index // 20
        slot = index % 20
        add(
            "C012",
            platform_authors[index % len(platform_authors)],
            day, 8 + slot // 2, 7 + 30 * (slot % 2),
            f"Platform engineering note {index + 1}: {platform_topics[index % len(platform_topics)]}",
        )

    long_running_channels = (
        (
            ACME_CHANNEL,
            (DANIEL, NINA, PRIYA, AHMED, MARCUS),
            (
                "Customer migration runbook owners reviewed their next-week handoffs.",
                "The enterprise cutover calendar was reconciled with support coverage.",
                "A completed tenant move was archived from the active migration board.",
                "Customer communications copied the approved maintenance wording.",
                "The migration coordinator closed an obsolete readiness follow-up.",
                "Workspace and identity owners confirmed the standard escalation path.",
            ),
            "Migration coordination note",
        ),
        (
            SUPPORT_CHANNEL,
            (OMAR, NINA, "U018", "U035"),
            (
                "Enterprise queue review found no unassigned priority cases.",
                "Customer verification notes were attached to the account timeline.",
                "Support handed a regional login question to the identity rotation.",
                "The weekly enterprise escalation review completed without carryover.",
                "Account contacts confirmed their normal testing availability.",
                "A duplicate customer ticket was linked to its canonical incident.",
            ),
            "Enterprise support note",
        ),
        (
            DEBUG_CHANNEL,
            (ETHAN, SOFIA, MINA, QUINN, MILO),
            (
                "The staging debugger smoke test passed on the current worker image.",
                "A routine helper review was merged after its unit tests completed.",
                "The profiler attachment issue was reproduced only on an old sidecar.",
                "A flaky integration fixture was quarantined with an owner assigned.",
                "The current SDK canary produced no new serialization warnings.",
                "Review rotation coverage was updated for the next engineering shift.",
            ),
            "Debugging activity",
        ),
    )
    for channel, authors, topics, label in long_running_channels:
        for index in range(210):
            day = 1 + index // 18
            slot = index % 18
            add(
                channel,
                authors[index % len(authors)],
                day,
                8 + slot // 2,
                9 + 30 * (slot % 2),
                f"{label} {index + 1}: {topics[index % len(topics)]}",
            )

    assert next_number == 1673
    return history


SLACK_MESSAGES = [
    _with_defaults(message, _background_step(index))
    for index, message in enumerate(_BACKGROUND_LOG)
] + [_with_defaults(message) for message in _ACME_LOG] + _generated_deep_history()


# These records never enter the initial `messages` table. The simulator's
# event engine materializes an event's rows only after its prerequisite action
# or observation has occurred. A non-MSG namespace prevents collision with the
# ordinary runtime allocator while preserving stable IDs for evaluation.
SLACK_LATENT_MESSAGES: tuple[LatentMessage, ...] = (
    LatentMessage(
        "LAT001", "saml_revision_2", DEBUG_CHANNEL, ETHAN,
        "Revised after Ben's review. The metadata loader supplies a canonical Audience, so this "
        "now uses exact matching:\n```ts\nexport function audienceMatches(assertionAudience: string, "
        "expectedAudience: string): boolean {\n  return assertionAudience === expectedAudience;\n}\n```",
        "MSG194", "MSG197", 0,
    ),
    LatentMessage(
        "LAT002", "saml_deployed", DEBUG_CHANNEL, "U030",
        "CI passed for the revised `audienceMatches` implementation.", "MSG194", "LAT001", 0,
    ),
    LatentMessage(
        "LAT003", "saml_deployed", IDENTITY_CHANNEL, PRIYA,
        "The exact-match validator is deployed to the Acme authentication workers.", None, None, 1,
    ),
    LatentMessage(
        "LAT004", "saml_deployed", IDENTITY_CHANNEL, PRIYA,
        "Post-deploy smoke test: US-1 PASS; US-2 PASS; EU-1 PASS; EU-2 FAIL. EU-2 is no longer "
        "failing Audience validation; signature verification fails for key ID `acme-eu-2025`.",
        None, None, 2,
    ),
    LatentMessage(
        "LAT005", "saml_deployed", IDENTITY_CHANNEL, PRIYA,
        "The audience defect is fixed, but this is a different failure. Reopening EU-SAML-17 and "
        "comparing EU-2 with the original IDP-ACME-014 rotation notes before deciding whether the "
        "key path is recoverable today.", None, None, 3,
    ),
    LatentMessage(
        "LAT006", "export_backfill_results", DATA_CHANNEL, AHMED,
        "March corrective backfill finished: 12,481,991 records exported.", None, None, 0,
    ),
    LatentMessage(
        "LAT007", "export_backfill_results", DATA_CHANNEL, "U033",
        "Reconciliation bot: source snapshot 12,482,017; export 12,481,991; difference 26. "
        "This is not verified until the difference is explained.", None, None, 1,
    ),
    LatentMessage(
        "LAT008", "export_backfill_results", DATA_CHANNEL, AHMED,
        "That shape may match the old dry-run exclusion. ACME-DRYRUN-07 is the policy reference; "
        "please verify the documented IDs before calling this complete.", None, None, 2,
    ),
    LatentMessage(
        "LAT009", "export_reconciled", DATA_CHANNEL, "U033",
        "Reconciliation complete:\nsource records: 12,482,017\ndocumented exclusions: 26\n"
        "expected export: 12,481,991\nactual export: 12,481,991",
        None, None, 0,
    ),
    LatentMessage(
        "LAT010", "export_reconciled", DATA_CHANNEL, AHMED,
        "Confirmed: all 26 differences are the synthetic load-test records documented in "
        "ACME-DRYRUN-07. March and the historical export are VERIFIED COMPLETE; no blocker remains.",
        None, None, 1,
    ),
    LatentMessage(
        "LAT011", "export_reconciled", DEBUG_CHANNEL, SOFIA,
        "One follow-up revision after the reconciliation: I added the documented-exclusion count "
        "to the manifest assertion. The half-open UTC partition behavior is unchanged. This is "
        "correct against ACME-DRYRUN-07, but the thread is reopened for one final senior look.\n"
        "```python\nassert actual == source - documented_exclusions\n```",
        "MSG200", "MSG204", 2,
    ),
    LatentMessage(
        "LAT012", "permissions_revision_2", DEBUG_CHANNEL, MINA,
        "Revised after Ben's review:\n```ts\nexport function hasRequiredAccess(userGroups: string[], "
        "requiredGroups: string[]): boolean {\n  return requiredGroups.every((group) => "
        "userGroups.includes(group));\n}\n```",
        "MSG214", "MSG217", 0,
    ),
    LatentMessage(
        "LAT013", "permissions_checker_rerun", SUPPORT_CHANNEL, MARCUS,
        "Corrected checker rerun: EU-Finance PASS; EU-Legal FAIL; missing entitlement "
        "`acme-eu-legal-export`.", None, None, 0,
    ),
    LatentMessage(
        "LAT014", "permissions_checker_rerun", SUPPORT_CHANNEL, MARCUS,
        "I found the omitted EU-Legal entitlement and am applying it now. I still need direct "
        "verification before marking permissions complete.", None, None, 1,
    ),
    LatentMessage(
        "LAT015", "permissions_partial_verification", SUPPORT_CHANNEL, NINA,
        "First customer retest after Marcus's entitlement update: Acme Legal can authenticate and "
        "open EU-Legal now, but they have not opened the restricted export area or run the export "
        "workflow. This is only a partial check.", None, None, 0,
    ),
    LatentMessage(
        "LAT016", "permissions_verified", ACME_CHANNEL, MARCUS,
        "EU workspace access is verified end to end. I remain the owner; no permissions blocker "
        "remains.", None, None, 1,
    ),
    LatentMessage(
        "LAT017", "export_reopen_closed", DEBUG_CHANNEL, SOFIA,
        "Thanks — the final senior pass is recorded and the reconciliation assertion is merging.",
        "MSG200", "LAT011", 0,
    ),
    LatentMessage(
        "LAT018", "sso_key_investigated", IDENTITY_CHANNEL, PRIYA,
        "Compared EU-2 with IDP-ACME-014. Its assertion correctly names `acme-eu-2025`, but that "
        "regional worker pool is still receiving metadata without the current key after refresh. "
        "EU-1 has the current key, so this is isolated to the EU-2 metadata/signing-key path.",
        None, None, 0,
    ),
    LatentMessage(
        "LAT019", "sso_key_investigated", IDENTITY_CHANNEL, PRIYA,
        "The control-plane refresh for EU-2 is still returning the previous key set. I cannot "
        "safely bypass kid selection or fall back to `acme-eu-2024`. SSO remains unresolved; I "
        "own the Acme IdP escalation and another full smoke test after corrected metadata lands.",
        None, None, 1,
    ),
    LatentMessage(
        "LAT020", "permissions_verified", SUPPORT_CHANNEL, NINA,
        "Definitive Acme workflow test complete: the test users authenticated, opened EU-Finance "
        "and EU-Legal, entered the restricted export area, and completed the required Legal export "
        "action successfully.", None, None, 0,
    ),
    LatentMessage(
        "LAT021", "timing_confirmed", ACME_CHANNEL, NINA,
        "Confirmed directly with Acme: 9:30 PM was only their VP asking a question, not a change "
        "request. The approved start remains Thursday, August 20 at 9:00 PM PT, with a 90-minute "
        "window and about 15 minutes of customer-visible downtime.", "CUT001", None, 0,
    ),
    LatentMessage(
        "LAT022", "timing_confirmed", BRIDGE_CHANNEL, DANIEL,
        "One dependency that was missing from my original list: before go/no-go we must verify the "
        "Acme rollback worker and its pinned recovery configuration. Sam Okafor owns that check; "
        "a green deploy alone is not evidence that rollback invocation works.", None, None, 1,
    ),
    LatentMessage(
        "LAT023", "timing_confirmed", BRIDGE_CHANNEL, TESSA,
        "Bridge coverage update is now effective: I am primary identity bridge coverage from "
        "5:30 onward while Priya remains the SSO technical owner. Ahmed covers data, Marcus covers "
        "workspace access, Daniel coordinates, and Nina remains the Acme contact.",
        "CUT002", "CUT003", 2,
    ),
    LatentMessage(
        "LAT024", "rollback_initially_verified", BRIDGE_CHANNEL, SAM,
        "Rollback worker owner check: I own it. The recovery image and Acme tenant configuration "
        "are present, and the worker accepted the dry-run invocation.", "LAT022", None, 0,
    ),
    LatentMessage(
        "LAT025", "rollback_initially_verified", BRIDGE_CHANNEL, SAM,
        "Dry-run rollback invocation completed without changing customer state; recovery routing "
        "and the monitoring callback both returned PASS. This is sufficient to enter rehearsal.",
        "LAT022", "LAT024", 1,
    ),
    LatentMessage(
        "LAT026", "rollback_initially_verified", BRIDGE_CHANNEL, DANIEL,
        "Recording rollback as provisionally verified by Sam's invocation, subject to the final "
        "rehearsal exercising the same pinned configuration.", "LAT022", "LAT025", 2,
    ),
    LatentMessage(
        "LAT027", "rehearsal_started", BRIDGE_CHANNEL, DANIEL,
        "Final migration rehearsal is running. Component owners are posting actual results in "
        "this thread; do not reduce it to a single green status.", None, None, 0,
    ),
    LatentMessage(
        "LAT028", "rehearsal_started", BRIDGE_CHANNEL, AHMED,
        "Data/checksum step PASS: the reconciled March manifest and historical-export checksum "
        "match the frozen rehearsal input.", "LAT027", None, 1,
    ),
    LatentMessage(
        "LAT029", "rehearsal_started", BRIDGE_CHANNEL, MARCUS,
        "EU access step PASS: Finance and Legal both completed the required restricted workflow "
        "using the customer-verified entitlement set.", "LAT027", None, 2,
    ),
    LatentMessage(
        "LAT030", "rehearsal_started", BRIDGE_CHANNEL, PRIYA,
        "Authentication step: US-1, US-2, and EU-1 PASS; EU-2 still FAILS on the known "
        "`acme-eu-2025` signature/key path. Audience validation is passing.",
        "LAT027", None, 3,
    ),
    LatentMessage(
        "LAT031", "rehearsal_started", BRIDGE_CHANNEL, SAM,
        "Rollback invocation FAIL: rehearsal selected a stale, older Acme recovery-config pin than the "
        "dry run. No customer state changed, but my earlier provisional green is no longer enough; "
        "the pin must be corrected and the invocation rerun.", "LAT027", None, 4,
    ),
    LatentMessage(
        "LAT032", "rehearsal_started", BRIDGE_CHANNEL, "U030",
        "Monitoring step PASS: alert delivery, worker health, and migration dashboards all emitted "
        "the expected rehearsal signals.", "LAT027", None, 5,
    ),
    LatentMessage(
        "LAT033", "rehearsal_completed", BRIDGE_CHANNEL, SAM,
        "Corrected the recovery-config pin and reran the rollback invocation against the frozen "
        "Acme rehearsal input: PASS. Recovery routing and monitoring callback also PASS.",
        "LAT027", "LAT031", 0,
    ),
    LatentMessage(
        "LAT034", "rehearsal_completed", BRIDGE_CHANNEL, "U030",
        "Post-rerun monitoring is clean. No unintended customer-state mutation occurred during "
        "either rollback exercise.", "LAT027", "LAT033", 1,
    ),
    LatentMessage(
        "LAT035", "rehearsal_completed", BRIDGE_CHANNEL, DANIEL,
        "Final rehearsal evidence is complete: data/checksum, EU access, corrected rollback, and "
        "monitoring are verified. Authentication accurately reproduced the remaining EU-2 key "
        "failure, so the rehearsal is closed but the migration decision remains blocked on SSO.",
        "LAT027", "LAT034", 2,
    ),
)

# Observation-only events have no future message payload. They record that the
# agent actually received the deep prerequisite evidence, so a lucky review
# guess cannot substitute for locating the referenced historical discussion.
OBSERVATION_EVENT_IDS = (
    "sso_context_observed",
    "sso_failure_observed",
    "permissions_policy_observed",
    "permissions_partial_observed",
    "pagination_invariant_observed",
    "coverage_change_observed",
)
SCENARIO_EVENT_IDS = tuple(sorted(
    {message.event_id for message in SLACK_LATENT_MESSAGES} | set(OBSERVATION_EVENT_IDS)
))

# ---------------------------------------------------------------------------
# The Acme scenario's transition rules
# ---------------------------------------------------------------------------
#
# These used to be hand-written branches inside the transition engine, which
# meant the environment only worked for this one task. They are data now: the
# engine evaluates whatever rules the loaded scenario supplies and knows
# nothing about SAML, exports or rehearsals.
#
# The two review threads run the same two-step shape. The first reply has to
# identify the real defect, and only then does an approval mean anything --
# which is what `requires_activated` says. An agent that opens with "looks
# good to me" matches no rule and the world does not move.

_LGTM = "Looks good to me"

SCENARIO_RULES: tuple[ScenarioRule, ...] = (
    ScenarioRule(
        rule_id="r010_saml_critique", event_id="saml_revision_2",
        trigger="reply_keywords", thread_id="MSG194",
        # Two ideas, each stated many ways: the comparison that is wrong, and
        # the comparison that is required. Stems, so "containment" and
        # "contains" are the same evidence.
        keyword_groups=(("substring", "includes", "contain", "partial", "prefix",
                         "loose", "merely"),
                        ("exact", "equal", "verbatim", "===", "identity",
                         "identical", "strict")),
    ),
    ScenarioRule(
        rule_id="r011_saml_approved", event_id="saml_deployed",
        trigger="reply_exact", thread_id="MSG194", exact_body=_LGTM,
        requires_activated=("saml_revision_2",),
    ),
    ScenarioRule(
        rule_id="r020_permissions_critique", event_id="permissions_revision_2",
        trigger="reply_keywords", thread_id="MSG214",
        keyword_groups=(("some", "any", "one", "partial"), ("every", "all", "each", "both")),
    ),
    ScenarioRule(
        rule_id="r021_permissions_approved", event_id="permissions_checker_rerun",
        trigger="reply_exact", thread_id="MSG214", exact_body=_LGTM,
        requires_activated=("permissions_revision_2",),
    ),
    # The export helper is correct, so approving it is the first step. The
    # thread reopens later, and closing it again is a separate event.
    ScenarioRule(
        rule_id="r030_export_approved", event_id="export_backfill_results",
        trigger="reply_exact", thread_id="MSG200", exact_body=_LGTM,
    ),
    ScenarioRule(
        rule_id="r031_export_reclosed", event_id="export_reopen_closed",
        trigger="reply_exact", thread_id="MSG200", exact_body=_LGTM,
        requires_activated=("export_reconciled",),
    ),
    ScenarioRule(
        rule_id="r040_timing_confirmed", event_id="timing_confirmed",
        trigger="reply_keywords", thread_id="CUT001",
        # Daniel raised it in CUT001 and Nina owns the Acme relationship, so
        # either is a defensible person to resolve it with. What stays required
        # is that the agent noticed the 9:30 contradiction at all.
        recipient_ids=(DANIEL, NINA),
        tools=("reply_to_thread", "send_dm_message"),
        keyword_groups=(("9:30", "930", "nine thirty"),
                        ("confirm", "question", "request", "approv", "actual",
                         "change", "still", "plan", "sign", "supersede",
                         "stand", "final")),
    ),
    ScenarioRule(
        rule_id="r050_rollback_checked", event_id="rollback_initially_verified",
        trigger="reply_keywords", thread_id="LAT022",
        # LAT022 introduces the dependency, but the whole cutover bridge is
        # where it would be chased, and Sam owns the check. A run that asked
        # him for exactly this evidence one thread over used to lose the entire
        # rollback and rehearsal chain to the thread id.
        # LAT024 answers LAT022, which Daniel only posts once the timing
        # question is settled -- so the dependency has to be stated rather than
        # left to the thread id to enforce.
        requires_activated=("timing_confirmed",),
        channel_id=BRIDGE_CHANNEL, recipient_ids=(SAM,),
        tools=("reply_to_thread", "post_message", "send_dm_message"),
        keyword_groups=(("rollback", "recovery"),
                        ("verif", "test", "run", "status", "invo", "evidence",
                         "confirm")),
    ),
    ScenarioRule(
        rule_id="r060_rehearsal_closed", event_id="rehearsal_completed",
        trigger="reply_keywords", thread_id="LAT027",
        # The prerequisite has to be declared, not implied by the thread id.
        # LAT027 does not exist until the rehearsal starts, so while this rule
        # was thread-only the ordering held by accident; once it also listens
        # to the channel, an early ask would try to publish replies to a parent
        # that is not visible yet.
        requires_activated=("rehearsal_started",),
        channel_id=BRIDGE_CHANNEL, recipient_ids=(SAM, DANIEL),
        tools=("reply_to_thread", "post_message", "send_dm_message"),
        keyword_groups=(("rollback", "recovery"), ("pin", "config"),
                        ("rerun", "retest", "fix", "correct", "redo", "again")),
    ),
    # Observation rules. These fire on evidence the agent actually received,
    # so locating the referenced history is what advances the world -- not
    # guessing what it probably said.
    ScenarioRule(
        rule_id="r100_sso_context", event_id="sso_context_observed",
        trigger="observed",
        observed_ids=("MSG306", "MSG307", "MSG308", "MSG309", "MSG310"),
    ),
    ScenarioRule(
        rule_id="r101_sso_failure", event_id="sso_failure_observed",
        trigger="observed", observed_ids=("LAT004", "LAT005"),
    ),
    ScenarioRule(
        rule_id="r102_permissions_policy", event_id="permissions_policy_observed",
        trigger="observed", observed_ids=("MSG795", "MSG796", "MSG797", "MSG798"),
    ),
    ScenarioRule(
        rule_id="r103_permissions_partial", event_id="permissions_partial_observed",
        trigger="observed", observed_ids=("LAT015",),
    ),
    ScenarioRule(
        rule_id="r104_pagination_invariant", event_id="pagination_invariant_observed",
        trigger="observed", observed_ids=("MSG799", "MSG800", "MSG801", "MSG802"),
    ),
    ScenarioRule(
        rule_id="r105_coverage_change", event_id="coverage_change_observed",
        trigger="observed", observed_ids=("LAT023",),
    ),
    ScenarioRule(
        rule_id="r110_export_reconciled", event_id="export_reconciled",
        trigger="observed", observed_ids=("MSG551", "MSG552", "MSG553", "MSG554"),
        requires_activated=("export_backfill_results",),
    ),
    ScenarioRule(
        rule_id="r111_permissions_partial_verification",
        event_id="permissions_partial_verification",
        trigger="observed", observed_ids=("LAT013", "LAT014"),
        requires_activated=("permissions_checker_rerun",),
    ),
    # Dependency closures: nothing to do, they resolve once their inputs hold.
    ScenarioRule(
        rule_id="r200_permissions_verified", event_id="permissions_verified",
        trigger="all_of",
        requires_activated=("permissions_partial_observed", "permissions_policy_observed"),
    ),
    ScenarioRule(
        rule_id="r201_sso_investigated", event_id="sso_key_investigated",
        trigger="all_of",
        requires_activated=("sso_context_observed", "sso_failure_observed"),
    ),
    ScenarioRule(
        rule_id="r210_rehearsal_starts", event_id="rehearsal_started",
        trigger="all_of",
        requires_activated=(
            "sso_key_investigated", "export_reopen_closed", "permissions_verified",
            "timing_confirmed", "coverage_change_observed", "rollback_initially_verified",
        ),
    ),
)

# Inbox entries and one membership that arrive with their events. Declaring
# them keeps a run reproducible: the same work always produces the same inbox,
# rather than one that depends on who happened to be following what.
SCENARIO_NOTIFICATIONS: tuple[LatentNotification, ...] = (
    LatentNotification("NTF-LAT001", "saml_revision_2", BEN, "mention",
                       "LAT001", DEBUG_CHANNEL),
    LatentNotification("NTF-LAT002", "saml_deployed", BEN, "thread_reply",
                       "LAT005", IDENTITY_CHANNEL),
    LatentNotification("NTF-LAT003", "export_backfill_results", BEN, "thread_reply",
                       "LAT007", DATA_CHANNEL),
    LatentNotification("NTF-LAT004", "export_reconciled", BEN, "thread_reply",
                       "LAT010", DATA_CHANNEL),
    LatentNotification("NTF-LAT005", "permissions_revision_2", BEN, "mention",
                       "LAT012", DEBUG_CHANNEL),
    LatentNotification("NTF-LAT006", "permissions_partial_observed", BEN, "mention",
                       "LAT015", SUPPORT_CHANNEL),
    LatentNotification("NTF-LAT007", "permissions_verified", BEN, "mention",
                       "LAT020", SUPPORT_CHANNEL),
    LatentNotification("NTF-LAT008", "timing_confirmed", BEN, "thread_reply",
                       "LAT021", ACME_CHANNEL),
    LatentNotification("NTF-LAT009", "rehearsal_completed", BEN, "thread_reply",
                       "LAT035", BRIDGE_CHANNEL),
)

# The identity backup the bridge has been trying to name. He starts absent, and
# a later membership read shows him present.
SCENARIO_MEMBERSHIPS: tuple[LatentMembership, ...] = (
    LatentMembership("MBR-LAT001", "coverage_change_observed", BRIDGE_CHANNEL,
                     FELIX_IDENTITY, "member"),
)

#: The rehearsal happens at 7:30 PM on the benchmark's own calendar rather than
#: one second after whichever action completed its prerequisites.
_SCHEDULED_STEPS: dict[str, int] = {"rehearsal_started": moment(16, 19, 30)}

ACME_SCENARIO = Scenario(
    events=tuple(
        ScenarioEvent(event_id, _SCHEDULED_STEPS.get(event_id))
        for event_id in SCENARIO_EVENT_IDS
    ),
    latent_messages=SLACK_LATENT_MESSAGES,
    latent_notifications=SCENARIO_NOTIFICATIONS,
    latent_memberships=SCENARIO_MEMBERSHIPS,
    rules=SCENARIO_RULES,
    observation_only=OBSERVATION_EVENT_IDS,
)

_STEP_BY_ID = {message.message_id: message.created_step for message in SLACK_MESSAGES}


def _first_step_in(conversation_id: str) -> int:
    return min(
        (m.created_step for m in SLACK_MESSAGES if m.conversation_id == conversation_id),
        default=1,
    )


def _chat(chat_id: str, kind: str, name: str | None) -> Chat:
    """A chat exists before anything is said in it."""
    return Chat(chat_id, kind, name, _first_step_in(chat_id) - 1)


SLACK_CHATS = [
    # Everyone has a chat with themselves, the way Slack provides one. It is
    # where people park notes mid-incident, so it exists from the start rather
    # than costing a turn to discover and create.
    _chat("D007", "dm", None),
    _chat("D001", "dm", None),
    _chat("D002", "dm", None),
    _chat("D003", "dm", None),
    _chat("D004", "dm", None),
    _chat("D005", "dm", None),
    _chat("D006", "dm", None),
    _chat("G001", "group", "Incident Operations"),
    _chat("G002", "group", "Orion Legal Review"),
    _chat("G003", "group", "Northwind Recovery"),
    _chat("G004", "group", "Checkout On-call Rotation"),
    _chat("G005", "group", "Platform Standup"),
    _chat("G006", "group", "Weekend Coverage"),
]

# Runtime chat IDs are minted as D/G{chat count + 1}, so no seeded chat may
# carry a numeric suffix above the seeded chat count or the next created chat
# collides with it.
assert all(
    int(chat.chat_id[1:]) <= len(SLACK_CHATS) for chat in SLACK_CHATS
), "a seeded chat ID would collide with the next runtime-minted chat ID"

SLACK_CHAT_PARTICIPANTS = [
    ChatParticipant("D007", BEN),
    ChatParticipant("D001", "U001"), ChatParticipant("D001", BEN),
    ChatParticipant("D002", BEN), ChatParticipant("D002", "U003"),
    ChatParticipant("D003", "U008"), ChatParticipant("D003", "U010"),
    ChatParticipant("D004", BEN), ChatParticipant("D004", "U004"),
    ChatParticipant("D005", BEN), ChatParticipant("D005", "U006"),
    ChatParticipant("D006", "U016"), ChatParticipant("D006", "U019"),
    ChatParticipant("G001", "U001"), ChatParticipant("G001", BEN),
    ChatParticipant("G001", "U003"), ChatParticipant("G001", "U004"),
    ChatParticipant("G002", "U009"), ChatParticipant("G002", "U010"),
    ChatParticipant("G002", "U012"), ChatParticipant("G002", "U038"),
    ChatParticipant("G003", BEN), ChatParticipant("G003", "U004"), ChatParticipant("G003", "U012"),
    ChatParticipant("G004", BEN), ChatParticipant("G004", "U006"), ChatParticipant("G004", "U007"),
    ChatParticipant("G004", "U014"), ChatParticipant("G004", "U028"),
    ChatParticipant("G005", BEN), ChatParticipant("G005", "U013"),
    ChatParticipant("G005", "U019"), ChatParticipant("G005", "U039"),
    ChatParticipant("G006", "U016"), ChatParticipant("G006", "U027"), ChatParticipant("G006", "U030"),
]

SLACK_SEEDED_MENTIONS = [
    MessageMention("MSG039", "person", "U012", 0),
    MessageMention("MSG039", "person", BEN, 1),
    MessageMention("MSG040", "user_group", "S004", 0),
    MessageMention("MSG177", "person", "U042", 0),
    MessageMention("CUT016", "person", BEN, 0),
    MessageMention("CUT017", "person", BEN, 0),
    MessageMention("CUT018", "person", BEN, 0),
]


def _reactions(*rows: tuple[str, str, str]) -> list[Reaction]:
    """Number reactions REA001.. in order.

    The runtime mints IDs as REA{count + 1}, and no tool removes a reaction, so
    a contiguous seeded block can never collide with one the agent creates. A
    reaction lands at the instant of the message it is attached to, which keeps
    every seeded timestamp a function of the message log alone.
    """
    return [
        Reaction(f"REA{index:03d}", message_id, user_id, emoji, _STEP_BY_ID[message_id])
        for index, (message_id, user_id, emoji) in enumerate(rows, start=1)
    ]


# Reactions other people left before the episode began. None of them is
# (open action, Ben, eyes): the reaction milestone stays entirely unearned at
# the start, and the verifier excludes this seeded set from its extra-reaction
# penalty so pre-existing activity is never charged to the agent.
SLACK_REACTIONS = _reactions(
    # Someone acknowledging Ben's note about 9:30 -- a harmless signal that
    # ages into a misleading one once 9:00 is reconfirmed.
    ("CUT015", TESSA, "+1"),
    ("MSG041", "U029", "tada"),
    ("MSG041", "U021", "+1"),
    ("MSG045", "U016", "eyes"),
    ("MSG045", "U013", "+1"),
    ("MSG049", "U016", "white_check_mark"),
    ("MSG052", "U028", "chart_with_upwards_trend"),
    ("MSG055", "U023", "heart"),
    ("MSG055", "U037", "+1"),
    ("MSG058", "U020", "eyes"),
    ("MSG070", "U035", "coffee"),
    ("MSG001", "U003", "eyes"),
    ("MSG001", "U006", "fire"),
    ("MSG001", "U016", "eyes"),
    ("MSG003", "U018", "eyes"),
    ("MSG004", "U007", "+1"),
    ("MSG009", "U006", "+1"),
    ("MSG016", "U007", "white_check_mark"),
    ("MSG016", BEN, "+1"),
    ("MSG020", "U018", "white_check_mark"),
    ("MSG020", BEN, "eyes"),
    ("MSG021", "U006", "white_check_mark"),
    ("MSG025", "U007", "thinking_face"),
    ("MSG031", "U024", "+1"),
    ("MSG036", "U001", "eyes"),
    ("MSG100", "U032", "+1"),
    ("MSG103", "U030", "white_check_mark"),
    # Acme migration. People cheered the claims that later turned out to be
    # premature, which is part of what makes the stale summaries feel credible.
    ("MSG128", "U046", "+1"),
    ("MSG128", "U048", "tada"),
    ("MSG135", "U041", "white_check_mark"),
    ("MSG135", BEN, "+1"),
    ("MSG146", "U041", "tada"),
    ("MSG157", "U048", "eyes"),
    ("MSG157", "U047", "eyes"),
    ("MSG164", "U041", "+1"),
    ("MSG169", "U041", "eyes"),
    ("MSG169", "U046", "eyes"),
    ("MSG181", "U046", "+1"),
    ("MSG160", "U042", "tada"),
)


def _read_through(conversation_id: str, last_read_message_id: str | None) -> ConversationRead:
    """Ben's read cursor, expressed as the last message he has seen.

    Naming the message rather than a step keeps the cursor meaningful when
    traffic is inserted around it: everything after that message stays unread,
    which is what makes the workspace look lived-in instead of freshly opened.
    """
    step = 0 if last_read_message_id is None else _STEP_BY_ID[last_read_message_id]
    return ConversationRead(conversation_id, BEN, step)


# ---------------------------------------------------------------------------
# Pins, saved items and follows
# ---------------------------------------------------------------------------
#
# All three are things a real workspace accumulates and none of them are
# authoritative. A pin is what somebody thought was worth pinning at the time,
# a saved item is what Ben meant to come back to, and a followed thread is what
# he happened to be in. Two of the three pins below are stale, most of the
# saved items are dead ends, and the threads he follows are not the threads
# that matter today.

SLACK_PINS = [
    # Stale: the kickoff window was superseded by the 9:00 PM approval.
    Pin(ACME_CHANNEL, "MSG125", DANIEL, _STEP_BY_ID["MSG125"] + 120),
    # Partially stale: Lena is named, and she handed EU access to Marcus.
    Pin(ACME_CHANNEL, "MSG129", DANIEL, _STEP_BY_ID["MSG129"] + 90),
    # Still current: the window length and downtime never changed.
    Pin(ACME_CHANNEL, "MSG135", NINA, _STEP_BY_ID["MSG135"] + 240),
]

SLACK_SAVED_ITEMS = [
    SavedItem(BEN, "MSG551", _STEP_BY_ID["MSG551"] + 600),   # July dry run
    SavedItem(BEN, "MSG799", _STEP_BY_ID["MSG799"] + 300),   # directory pagination
    SavedItem(BEN, "MSG125", _STEP_BY_ID["MSG125"] + 400),   # old cutover checklist
    SavedItem(BEN, "MSG306", _STEP_BY_ID["MSG306"] + 500),   # first Acme cert rotation
    SavedItem(BEN, "MSG031", _STEP_BY_ID["MSG031"] + 200),   # unrelated platform note
]

# Followed early enough to still be receiving replies, or late enough that the
# conversation had already finished -- which is why this list is not a to-do
# list. Only the July dry-run thread still surfaces anything.
SLACK_THREAD_FOLLOWS = [
    ThreadFollow(BEN, "MSG551", _STEP_BY_ID["MSG551"] + 60),     # old dry-run thread
    ThreadFollow(BEN, "MSG167", 1352700),                        # current export thread
    ThreadFollow(BEN, "MSG186", 1250500),                        # a review already closed
    ThreadFollow(BEN, "MSG041", 639600),                         # all-hands, irrelevant
]

SLACK_READ_CURSORS = [
    _read_through("C001", "MSG027"),
    _read_through("C002", "MSG021"),
    _read_through("C003", "MSG013"),
    _read_through("C004", "MSG018"),
    _read_through("C005", "MSG005"),
    _read_through("C006", "MSG008"),
    _read_through("C007", "MSG024"),
    _read_through("C008", "MSG032"),
    _read_through("C010", "MSG037"),
    _read_through("C012", "MSG048"),
    _read_through("C014", "MSG051"),
    _read_through("C016", "MSG100"),
    _read_through("C018", "MSG070"),
    _read_through("D001", None),
    _read_through("D002", "MSG029"),
    _read_through("D004", "MSG067"),
    _read_through("D005", None),
    _read_through("D007", "MSG124"),
    _read_through("G001", None),
    _read_through("G003", None),
    _read_through("G004", "MSG065"),
    _read_through("G005", "MSG063"),
    _read_through("C019", "MSG128"),
    _read_through("C020", "MSG146"),
    _read_through("C021", "MSG161"),
    _read_through("C022", "MSG174"),
]


# ---------------------------------------------------------------------------
# Inbox
# ---------------------------------------------------------------------------
#
# Two weeks of people mentioning each other and sending DMs would have left
# notifications behind, so the initial state contains them. They are derived
# from the messages above at import time rather than typed out, which keeps
# them from drifting when the history changes, and they are ordinary seed data
# once built: the service inserts them like reactions or read cursors and does
# not compute anything at seed time.


def _build_notifications() -> tuple[Notification, ...]:
    """Who the seeded history would have notified, and what they have read."""
    mentions: dict[str, list[MessageMention]] = {}
    for mention in SLACK_SEEDED_MENTIONS:
        mentions.setdefault(mention.message_id, []).append(mention)

    chat_members: dict[str, set[str]] = {}
    for participant in SLACK_CHAT_PARTICIPANTS:
        chat_members.setdefault(participant.chat_id, set()).add(participant.user_id)

    channel_members: dict[str, set[str]] = {}
    for membership in build_slack_memberships():
        channel_members.setdefault(membership.channel_id, set()).add(membership.user_id)

    group_members: dict[str, set[str]] = {}
    for member in SLACK_USER_GROUP_MEMBERS:
        group_members.setdefault(member.user_group_id, set()).add(member.user_id)

    cursors = {
        (cursor.conversation_id, cursor.user_id): cursor.last_read_step
        for cursor in SLACK_READ_CURSORS
    }
    authors = {message.message_id: message.author_id for message in SLACK_MESSAGES}
    _thread_of = {
        message.message_id: message.thread_parent_id or message.message_id
        for message in SLACK_MESSAGES
    }
    conversations = {message.message_id: message.conversation_id for message in SLACK_MESSAGES}

    built: list[Notification] = []
    for message in sorted(SLACK_MESSAGES, key=lambda m: (m.created_step, m.message_id)):
        container = message.conversation_id
        recipients: dict[str, str] = {}
        if container in chat_members:
            for user_id in chat_members[container]:
                recipients[user_id] = "direct_message"
        for mention in mentions.get(message.message_id, []):
            if mention.mention_type == "person":
                recipients[mention.target_id] = "mention"
            elif mention.mention_type == "special":
                for user_id in channel_members.get(container, set()):
                    recipients.setdefault(user_id, "mention")
            elif mention.mention_type == "user_group":
                for user_id in group_members.get(mention.target_id, set()):
                    recipients.setdefault(user_id, "mention")
        recipients.pop(message.author_id, None)
        for user_id, kind in sorted(recipients.items()):
            read_through = cursors.get((container, user_id))
            built.append(
                Notification(
                    notification_id="",
                    user_id=user_id,
                    kind=kind,
                    message_id=message.message_id,
                    conversation_id=container,
                    created_step=message.created_step,
                    read_step=(
                        read_through
                        if read_through is not None and message.created_step <= read_through
                        else None
                    ),
                )
            )
    follows = {
        (follow.user_id, follow.thread_id): follow.followed_step
        for follow in SLACK_THREAD_FOLLOWS
    }
    for message in SLACK_MESSAGES:
        if message.thread_parent_id is None:
            continue
        for (user_id, thread_id), followed_step in follows.items():
            if thread_id != message.thread_parent_id:
                continue
            if user_id == message.author_id or message.created_step <= followed_step:
                continue
            built.append(
                Notification(
                    notification_id="",
                    user_id=user_id,
                    kind="thread_reply",
                    message_id=message.message_id,
                    conversation_id=message.conversation_id,
                    created_step=message.created_step,
                    read_step=None,
                )
            )

    for reaction in SLACK_REACTIONS:
        author = authors.get(reaction.message_id)
        if author is None or author == reaction.user_id:
            continue
        built.append(
            Notification(
                notification_id="",
                user_id=author,
                kind="reaction",
                message_id=reaction.message_id,
                conversation_id=conversations[reaction.message_id],
                created_step=reaction.created_step,
                read_step=None,
            )
        )

    # A real inbox does not hold one entry per message in a busy group chat; it
    # holds the conversation. Collapse direct-message runs to the newest one
    # per person per conversation and leave everything addressed at somebody --
    # mentions, thread replies, reactions -- alone.
    newest: dict[tuple[str, str, str], Notification] = {}
    kept: list[Notification] = []
    for item in built:
        if item.kind not in {"direct_message", "thread_reply"}:
            kept.append(item)
            continue
        key = (item.user_id, item.kind, item.conversation_id if item.kind == "direct_message"
               else str(item.message_id))
        if item.kind == "thread_reply":
            key = (item.user_id, item.kind, _thread_of.get(item.message_id, item.message_id))
        if key not in newest or item.created_step > newest[key].created_step:
            newest[key] = item
    # An inbox entry exists because something still wants your attention. A
    # conversation you have already read through does not.
    kept.extend(item for item in newest.values() if item.read_step is None)
    kept.sort(key=lambda n: (n.created_step, n.user_id, n.message_id))
    return tuple(
        replace(item, notification_id=f"NTF-{index:06d}")
        for index, item in enumerate(kept, start=1)
    )



# Channels with an explicit roster. Everything else is a public channel that
# the whole workspace belongs to. Ben's memberships are exactly the channels a
# platform incident coordinator would be in: he is deliberately absent from
# #data-quality, #checkout-web, #design-review, #security-operations and
# #exec-briefing, so the workspace holds conversations he can find only through
# search, and one private channel he cannot see at all.
_EXPLICIT_MEMBERS: dict[str, set[str]] = {
    "C006": {"U001", BEN, "U003"},
    "C008": {BEN, "U006", "U009", "U011", "U012", "U024"},
    "C009": {"U001", "U008", "U022", "U036"},
    "C010": {BEN, "U004", "U012", "U034", "U035"},
    "C011": {"U006", "U007", "U009", "U011", "U020", "U033"},
    "C013": {"U006", "U014", "U021", "U023", "U024", "U028"},
    "C015": {"U009", "U021", "U023", "U037"},
    "C017": {"U001", "U009", "U040"},
    "C019": {BEN, "U041", "U042", "U043", "U044", "U045", "U046", "U047"},
    "C020": {BEN, "U008", "U022", "U041", "U042", "U048"},
    "C021": {BEN, "U011", "U020", "U033", "U041", "U045", "U046"},
    "C022": {BEN, "U004", "U018", "U035", "U041", "U044", "U046", "U047"},
    "C023": {BEN, "U013", "U017", "U019", "U024", "U030", "U032", "U039",
             "U042", "U044", "U045", "U049", "U050"},
    # The cutover bridge. Ben has to find it and join; Felix Moreau, the
    # identity backup, is the coverage gap the bridge discussion is about.
    "C024": {DANIEL, NINA, AHMED, MARCUS, OMAR, PRIYA, TESSA},
}


def _validate_threads() -> None:
    """Threads are one level deep, and every reply answers its own thread.

    The runtime enforces this by resolving a reply target to its thread root.
    The seed has to satisfy the same rule by construction, or it would describe
    a workspace no sequence of tool calls could have produced.
    """
    by_id = {m.message_id: m for m in SLACK_MESSAGES}
    for message in SLACK_MESSAGES:
        parent_id = message.thread_parent_id
        if parent_id is not None:
            parent = by_id[parent_id]
            if parent.thread_parent_id is not None:
                raise ValueError(f"{message.message_id} replies to a reply: threads are flat")
            if parent.conversation_id != message.conversation_id:
                raise ValueError(f"{message.message_id} replies across conversations")
            if parent.created_step >= message.created_step:
                raise ValueError(f"{message.message_id} precedes the thread it is in")
        target_id = message.reply_to_id
        if parent_id is None:
            if target_id is not None:
                raise ValueError(f"{message.message_id} answers a message but is not in a thread")
            continue
        if target_id is None:
            raise ValueError(f"{message.message_id} is a reply with no origin")
        target = by_id[target_id]
        if (target.thread_parent_id or target.message_id) != parent_id:
            raise ValueError(f"{message.message_id} answers a message in another thread")
        if target.created_step >= message.created_step:
            raise ValueError(f"{message.message_id} answers a message from its own future")


_validate_threads()


def conversations_visible_to(user_id: str) -> set[str]:
    """Conversation IDs `user_id` can read, as a property of the seed.

    This is the same rule the running service enforces -- public channels are
    readable by anyone, private channels only by their members, chats only by
    their participants -- restated over the seed dataclasses so the verifier can
    apply it without a database. `test_seed_visibility_matches_the_live_service`
    pins the two statements of the rule together.
    """
    memberships = {m.channel_id for m in build_slack_memberships() if m.user_id == user_id}
    visible = {
        channel.channel_id
        for channel in SLACK_CHANNELS
        if not channel.is_private or channel.channel_id in memberships
    }
    visible |= {p.chat_id for p in SLACK_CHAT_PARTICIPANTS if p.user_id == user_id}
    return visible


def build_slack_memberships() -> list[Membership]:
    """Channel memberships with realistic public/private participation."""
    memberships = []
    for channel in SLACK_CHANNELS:
        channel_id = channel.channel_id
        for user in SLACK_USERS:
            user_id = user.user_id
            members = _EXPLICIT_MEMBERS.get(channel_id)
            if members is not None and user_id not in members:
                continue
            if channel.is_private and members is None:
                continue
            memberships.append(
                Membership(
                    membership_id=f"M{channel_id[-3:]}{user_id[-3:]}",
                    channel_id=channel_id,
                    user_id=user_id,
                    role="owner" if user_id == channel.owner_id else "member",
                )
            )
    return memberships


SLACK_NOTIFICATIONS: tuple[Notification, ...] = _build_notifications()
