#!/usr/bin/env python3
"""Self-contained contract checks that do not mutate the graded workspace."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent.parent
#: The two checks below read the task's source files. Inside the container this
#: suite runs from /tests and the repository is not present, so they are
#: authoring-time guards: they run from a checkout and skip in the environment,
#: rather than failing the verifier over a file that was never shipped.
IN_REPOSITORY = (ROOT / "task.toml").is_file()

from slack_sim.environment import SlackIncidentEnvironment
from slack_sim.identity import LOGGED_IN_USER
from slack_sim.service import TOOL_NAMES, export_state

import dynamic_scenario as scenario


def test_the_harness_hands_the_agent_the_workspace() -> None:
    """The agent must arrive holding the tools, not hunting for them.

    A run against an earlier revision spent four shell commands looking for a
    task description on disk, concluded it had no way to read Slack, and
    refused -- correctly, on the information it had. The environment was fine;
    nothing had handed the agent an interface. Harbor registers the server
    declared here with whichever agent runs the task, so the fix belongs in the
    task's configuration rather than in prose the prompt has to carry.
    """
    import tomllib

    config = tomllib.loads((ROOT / "task.toml").read_text(encoding="utf-8"))
    servers = config["environment"]["mcp_servers"]
    slack = next((server for server in servers if server["name"] == "slack"), None)
    assert slack is not None, "task.toml declares no slack MCP server for the agent"
    assert slack["transport"] == "stdio", (
        "the environment runs with no network; a socket-side subprocess needs none"
    )

    command = Path(slack["command"])
    wrapper = ROOT / "environment" / "bin" / command.name
    assert wrapper.is_file(), f"{slack['command']} has no wrapper at {wrapper}"

    dockerfile = (ROOT / "environment" / "Dockerfile").read_text(encoding="utf-8")
    agent_stage = dockerfile.split("FROM base AS agent", 1)[-1]
    assert f"bin/{command.name} {slack['command']}" in agent_stage, (
        f"{command.name} is never installed into the agent image"
    )
    assert "mcp_server.py" in agent_stage, "the agent image ships no MCP server to run"


def test_the_instruction_does_not_smuggle_in_the_interface() -> None:
    """Daniel writes to a colleague, not to a tool user.

    Anything about clients, sockets or command names in the prompt is the task
    author reaching around the harness. If a future agent cannot find the
    workspace, the repair is a server it is handed, not a paragraph it is told.
    """
    import re

    instruction = (ROOT / "instruction.md").read_text(encoding="utf-8")
    leaked = [
        marker
        for marker in ("list_tools", "mcp", "/usr/local/bin", "agent.sock", "--input-payload")
        if marker in instruction.lower()
    ]
    assert not leaked, f"the prompt is describing the harness: {leaked}"
    # Anywhere, not just at the start of a line: an example smuggled into the
    # middle of a sentence teaches exactly as much as one in a code block.
    demonstrated = {
        name for name in re.findall(r"slack\s+`?([a-z_]+)", instruction, re.IGNORECASE)
        if name in set(TOOL_NAMES)
    }
    assert not demonstrated, (
        f"the prompt is demonstrating tools the harness already hands over: "
        f"{sorted(demonstrated)}"
    )


# ---------------------------------------------------------------------------
# What the agent's container is allowed to contain
# ---------------------------------------------------------------------------

#: Exactly the modules of the world package that may reach the agent's image.
#: That image is built by naming files, so this set and the Dockerfile's COPY
#: line are two statements of one decision, and the test below fails when they
#: drift apart.
#:
#: Every other module is withheld for a specific reason. `seed` and `service`
#: would let the agent build a private copy of the workspace and read private
#: channels it was refused; `migration_truth` is the answer key itself, and
#: `migration_reward`, `scenario` and `tracker` each disclose what the grader
#: rewards. The remaining protections -- admin.sock at 0600, /opt/grading at
#: 0700 -- are checked in test_integrity_violations; this one guards the file
#: list those protections assume.
AGENT_READABLE_MODULES = frozenset({
    "__init__.py",           # package marker
    "protocol.py",           # the wire format, which the agent must speak
    "tool_definitions.py",   # the schemas its MCP server advertises
    "agent_cli.py",          # the CLI the oracle solution drives
    "mcp_server.py",         # what Harbor registers for the agent's harness
    # Inert on its own: it speaks to admin.sock, which is 0600 root, and the
    # `slack-admin` wrapper here is 0750. Shipped so the verifier -- which runs
    # in this container as root -- has a client without a second copy.
    "admin_cli.py",
})


def _agent_readable_copies(dockerfile: str) -> set[str]:
    """Basenames the agent stage puts anywhere the agent can open.

    A whole-directory copy shows up here as `slack_sim`, which belongs to no
    allowlist and so fails loudly -- deliberately, because the cheapest way to
    leak the package is one token: `COPY environment/slack_sim /opt/slack_sim`.
    """
    stage = dockerfile.split("FROM base AS agent", 1)[-1]
    copied: set[str] = set()
    for line in stage.replace("\\\n", " ").splitlines():
        line = line.strip()
        if not line.startswith("COPY "):
            continue
        tokens = [t for t in line[len("COPY "):].split() if not t.startswith("--")]
        if len(tokens) < 2:
            continue
        destination, sources = tokens[-1], tokens[:-1]
        # /opt/grading is root-owned and 0700. The verifier runs in this same
        # container as root, so the answer key being present there is the
        # design, not a leak.
        if destination.startswith("/opt/grading"):
            continue
        copied.update(
            PurePosixPath(source).name for source in sources if "slack_sim" in source
        )
    return copied


def test_the_agent_image_carries_no_part_of_the_world_it_was_refused() -> None:
    if not IN_REPOSITORY:
        return
    dockerfile = (ROOT / "environment" / "Dockerfile").read_text(encoding="utf-8")
    copied = _agent_readable_copies(dockerfile)

    assert copied == set(AGENT_READABLE_MODULES), (
        "the agent image's file list drifted from the allowlist -- "
        f"gained {sorted(copied - AGENT_READABLE_MODULES)}, "
        f"lost {sorted(AGENT_READABLE_MODULES - copied)}"
    )

    package = ROOT / "environment" / "slack_sim"
    absent = sorted(name for name in AGENT_READABLE_MODULES if not (package / name).is_file())
    assert not absent, f"the allowlist names modules that no longer exist: {absent}"


def test_the_agent_image_guard_fails_on_the_edits_it_exists_to_catch() -> None:
    """An allowlist that cannot fail is decoration. Both leaks below are a few
    characters in one line, and neither changes any behaviour a test would
    otherwise notice: the suite stays green while the agent holds the answer."""
    if not IN_REPOSITORY:
        return
    dockerfile = (ROOT / "environment" / "Dockerfile").read_text(encoding="utf-8")

    wholesale = dockerfile.replace(
        "RUN mkdir -p /opt/slack_sim",
        "RUN mkdir -p /opt/slack_sim\nCOPY environment/slack_sim /opt/slack_sim",
    )
    assert wholesale != dockerfile, "the fixture no longer matches the Dockerfile"
    assert "slack_sim" in _agent_readable_copies(wholesale) - set(AGENT_READABLE_MODULES), (
        "copying the whole package into the agent image went unnoticed"
    )

    one_file = dockerfile.replace(
        "environment/slack_sim/mcp_server.py",
        "environment/slack_sim/mcp_server.py environment/slack_sim/migration_truth.py",
    )
    assert one_file != dockerfile, "the fixture no longer matches the Dockerfile"
    assert "migration_truth.py" in _agent_readable_copies(one_file), (
        "adding the answer key to the agent's COPY line went unnoticed"
    )


def test_the_reference_solution_reads_each_thread_in_the_channel_it_is_in() -> None:
    """Pin `solve.sh`'s channel variables against the seed.

    Nothing executes the reference solution -- the suites replay a scripted
    sequence in `dynamic_scenario.drive_terminal` instead -- so the script and
    the episode the tests call "the oracle" can drift apart in silence. They
    did: the script read CUT002, LAT022 and LAT027 out of `$ACME` while the
    seed puts all three in the cutover bridge, a channel Ben is not a member
    of. `thread_ts` returned empty, `set -e` killed the script partway, and
    `harbor run -a oracle` paid 0.49 for a task whose entire calibration rests
    on the reference scoring exactly 1.0. Every suite stayed green throughout.

    A static check rather than a run, because executing the script needs a live
    socket and this has to work from a checkout. It catches the mistake that
    was actually made: asking the wrong channel for a thread, and reading a
    channel without joining it first.
    """
    import re

    script = (ROOT / "solution" / "solve.sh").read_text(encoding="utf-8")
    with tempfile.TemporaryDirectory(prefix="slack-solution-") as directory:
        database = scenario.seeded(Path(directory) / "seed")
        # Memberships come from the seed and message locations from the played
        # episode, and the difference is the whole point: half the threads the
        # solution reads are latent and have no channel until something
        # activates them, while the episode itself joins the bridge -- so
        # reading memberships back out of the finished state would report every
        # channel as already joined and quietly pass.
        opening = export_state(database)
        state = scenario.drive_terminal(database)
    by_name = {channel["name"]: channel["channel_id"] for channel in state["channels"]}
    channel_of = {message["message_id"]: message["channel_id"] for message in state["messages"]}

    # `VAR=$(channel_id <name>)`, plus the one that has to be searched for
    # because it is absent from list_channels until the solution joins it.
    variables = {
        name: by_name[channel]
        for name, channel in re.findall(r"(\w+)=\$\(channel_id (\S+)\)", script)
        if channel in by_name
    }
    variables.update({
        name: by_name[channel]
        for name, channel in re.findall(
            r'(\w+)=\$\(slack search_channels.*?\.name == "([^"]+)"', script, re.S
        )
        if channel in by_name
    })
    assert len(variables) >= 5, f"channel variables did not resolve: {sorted(variables)}"

    lookups = re.findall(r'thread_ts "\$(\w+)" (\w+)', script)
    assert len(lookups) >= 4, f"expected the script to look threads up by id: {lookups}"
    for variable, message_id in lookups:
        assert variables.get(variable) == channel_of.get(message_id), (
            f"solve.sh reads {message_id} from ${variable} "
            f"({variables.get(variable)}), but it is in {channel_of.get(message_id)}"
        )

    # Reading a channel the acting user is not in is refused, so anything
    # beyond the seeded membership has to be joined first -- the step whose
    # absence caused the failure this test exists for.
    joined = {
        row["channel_id"] for row in opening["memberships"]
        if row["user_id"] == LOGGED_IN_USER.user_id
    }
    for variable, channel_id in sorted(variables.items()):
        if channel_id not in joined:
            assert f'join_channel --channel-id "${variable}"' in script, (
                f"solve.sh reads ${variable} ({channel_id}), which "
                f"{LOGGED_IN_USER.display_name} is not a member of, without joining it"
            )


def test_the_agent_package_imports_nothing_it_would_have_to_install() -> None:
    """`agent/` has to run wherever this task runs, without an install step.

    That is what lets `python3 -m agent` work from a bare checkout and what
    keeps the package honest about being a client: the moment it needs a wheel,
    it stops being something you can point at a container and run. The one
    exception is confined to one file -- `agent/harbor_agent.py` is the
    translation between Harbor's agent protocol and this package's own loop, so
    it necessarily imports Harbor, and it runs on the machine that starts the
    run rather than in the graded image.
    """
    import ast

    stdlib = set(sys.stdlib_module_names)
    local = {"agent", "slack_sim", "verifiers"}
    exempt = {"harbor_agent.py": {"harbor"}}
    offenders: list[str] = []
    for path in sorted((ROOT / "agent").rglob("*.py")):
        allowed = local | exempt.get(path.name, set())
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                names = [node.module]
            for name in names:
                if name.split(".")[0] not in stdlib | allowed:
                    offenders.append(f"{path.name}: {name}")
    assert not offenders, f"agent/ imports third-party packages: {sorted(set(offenders))}"

    # And it must stay out of the image, for the same reason the solution does.
    ignored = (ROOT / ".dockerignore").read_text(encoding="utf-8").split()
    assert "agent" in ignored, f"agent/ is not excluded from the build context: {ignored}"


def test_every_check_in_this_module_is_actually_called() -> None:
    """This suite dispatches by name rather than by discovery, so a check that
    nobody adds to `main` passes by never running. That is a worse failure than
    having no check at all, because the suite reports green either way -- and
    it caught both guards above during authoring, which is the only reason this
    one exists.
    """
    import inspect

    body = inspect.getsource(main)
    orphans = sorted(
        name for name, value in globals().items()
        if name.startswith("test_") and callable(value) and f"{name}()" not in body
    )
    assert not orphans, f"declared in this module but never run by main: {orphans}"


def main() -> None:
    # This suite is one long transcript rather than a collection of test
    # functions, so the standalone checks have to be called by name. Anything
    # added below must be listed here or it silently never runs.
    if IN_REPOSITORY:
        test_the_harness_hands_the_agent_the_workspace()
        test_the_instruction_does_not_smuggle_in_the_interface()
        test_the_agent_image_carries_no_part_of_the_world_it_was_refused()
        test_the_agent_image_guard_fails_on_the_edits_it_exists_to_catch()
        test_the_reference_solution_reads_each_thread_in_the_channel_it_is_in()
        test_the_agent_package_imports_nothing_it_would_have_to_install()
        test_every_check_in_this_module_is_actually_called()

    with tempfile.TemporaryDirectory(prefix="slack-contract-") as temp_dir:
        root = Path(temp_dir)
        environment = SlackIncidentEnvironment(
            db_path=root / "slack.db",
            snapshot_path=root / "seed.sql",
        )
        initial = environment.reset(
            session_cookie="episode-test",
            user_instruction="Inspect the incident workspace.",
            metadata={"split": "contract-test"},
        )
        assert initial["episode"] == {"seed": 0, "state": "ready", "max_turns": 100}
        assert initial["session"]["session_cookie"] == "episode-test"
        assert initial["session"]["actor_id"] == LOGGED_IN_USER.user_id
        assert initial["observation"]["actor_id"] == LOGGED_IN_USER.user_id
        assert initial["prompt"]["user"] == "Inspect the incident workspace."
        assert initial["prompt"]["prompt_id"] == "slack_operator"
        initial_virtual_time = export_state(environment.db_path)["virtual_time"]

        prompt_ids = {prompt["prompt_id"] for prompt in environment.prompts()}
        assert prompt_ids == {
            "slack_operator",
            "incident_coordinator",
            "communications_coordinator",
            "workspace_admin",
        }
        rendered = environment.render_prompt(
            "incident_coordinator",
            "Reconcile the active incident.",
            {"severity": "SEV-1", "region": "us-east"},
        )
        assert rendered["prompt_id"] == "incident_coordinator"
        assert "severity: SEV-1" in rendered["system"]
        assert rendered["tools"]

        definitions = environment.tool_definitions()
        tool_names = {tool["name"] for tool in definitions}
        assert tool_names == set(TOOL_NAMES)
        assert all("description" in tool and "input_schema" in tool for tool in definitions)

        transition = environment.step("episode-test", "search_messages", {"query": "Acme"})
        assert transition["ok"] is True
        assert transition["turn"] == 1
        assert transition["result"]["matches"]

        # A new adapter instance over the same sandbox sees the same episode,
        # workspace mutations, turn counter, and conversation history.
        resumed = SlackIncidentEnvironment(root / "slack.db", root / "seed.sql")
        state = resumed.state("episode-test")
        assert state["session"]["turn"] == 1
        assert state["session"]["prompt_id"] == "slack_operator"
        assert state["session"]["prompt_context"] == {}
        assert state["session"]["metadata"] == {"split": "contract-test"}
        assert [event["role"] for event in state["history"]] == [
            "system",
            "user",
            "assistant",
            "tool",
        ]
        assert [event["ts"] for event in state["history"]] == sorted(
            event["ts"] for event in state["history"]
        )

        # Relationship discovery exposes profiles, reporting lines, group
        # memberships, and only the chats visible to this actor.
        users = resumed.step("episode-test", "list_users", {})
        assert any(u["user_id"] == "U007" and u["manager_id"] == "U006" for u in users["result"]["users"])
        groups = resumed.step("episode-test", "list_user_groups", {})
        checkout = next(g for g in groups["result"]["user_groups"] if g["handle"] == "checkout-oncall")
        assert {m["user_id"] for m in checkout["members"]} == {
            LOGGED_IN_USER.user_id,
            "U006",
            "U007",
        }
        chats = resumed.step("episode-test", "list_chats", {})
        assert {c["chat_id"] for c in chats["result"]["chats"]} == {
            "D001", "D002", "D004", "D005", "D007", "G001", "G003", "G004", "G005",
        }, "list_chats must show every chat the actor is in, including the self-DM"
        # The workspace also holds chats between other people. Leaking them here
        # would be the same disclosure bug the private-channel tests guard.
        assert not {"D003", "D006", "G002", "G006"} & {
            c["chat_id"] for c in chats["result"]["chats"]
        }, "a chat the actor does not participate in was enumerated"
        # A DM nobody has opened yet still reports its unread traffic.
        unread = {c["chat_id"]: c["unread_count"] for c in chats["result"]["chats"]}
        assert unread["D001"] == 2, f"unread DM count is wrong: {unread}"

        # Edit tools enforce author, participant, owner, self, and admin permissions.
        edited = resumed.step(
            "episode-test", "edit_message", {"message_id": "MSG026", "body": "revised draft"}
        )
        assert edited["ok"] and edited["result"]["message"]["edited_ts"]
        renamed_group = resumed.step(
            "episode-test", "edit_group_chat_name", {"group_id": "G003", "new_name": "ACME Follow-up"}
        )
        assert renamed_group["result"]["chat"]["name"] == "ACME Follow-up"
        display = resumed.step(
            "episode-test",
            "edit_display_name",
            {"user_id": LOGGED_IN_USER.user_id, "new_display_name": "Ben O."},
        )
        assert display["result"]["user"]["display_name"] == "Ben O."

        created = resumed.step(
            "episode-test", "create_channel", {"name": "incident-drill", "is_private": True}
        )
        channel_id = created["result"]["id"]
        renamed_channel = resumed.step(
            "episode-test", "edit_channel_name", {"channel_id": channel_id, "new_name": "incident-drill-2"}
        )
        assert renamed_channel["result"]["channel"]["name"] == "incident-drill-2"
        invited = resumed.step(
            "episode-test", "invite_to_channel", {"channel_id": channel_id, "user_ids": ["U003", "U008"]}
        )
        assert invited["result"]["added_user_ids"] == ["U003", "U008"]

        people_tag = resumed.step(
            "episode-test",
            "tag_people",
            {"channel_id": channel_id, "user_ids": ["U003", "U008"], "body": "join the drill"},
        )
        assert people_tag["result"]["message"]["body"] == "<@U003> <@U008> join the drill"
        assert [m["target_id"] for m in people_tag["result"]["mentions"]] == ["U003", "U008"]
        here_tag = resumed.step(
            "episode-test", "tag_here", {"channel_id": channel_id, "body": "drill begins now"}
        )
        assert here_tag["result"]["message"]["body"] == "<!here> drill begins now"
        everyone_tag = resumed.step(
            "episode-test", "tag_everyone", {"channel_id": channel_id, "body": "drill complete"}
        )
        assert everyone_tag["ok"] is True  # Ben owns the newly created channel.
        denied = resumed.step(
            "episode-test", "tag_everyone", {"channel_id": "C008", "body": "unauthorized broadcast"}
        )
        assert denied["ok"] is False and denied["error"]["code"] == "permission_denied"
        group_tag = resumed.step(
            "episode-test",
            "tag_user_group",
            {"channel_id": channel_id, "user_group_id": "S002", "body": "review the drill"},
        )
        assert group_tag["result"]["mentions"] == [
            {"type": "user_group", "target_id": "S002", "ordinal": 0}
        ]

        added = resumed.step(
            "episode-test", "add_group_chat_participants", {"group_id": "G003", "user_ids": ["U003"]}
        )
        assert added["result"]["added_user_ids"] == ["U003"]
        group_history = resumed.step(
            "episode-test", "get_channel_messages", {"channel_id": "G003"}
        )
        assert group_history["result"]["conversation"]["chat_id"] == "G003"

        unread = resumed.step(
            "episode-test", "get_channel_messages", {"channel_id": "C003"}
        )
        assert unread["result"]["unread_count"] > 0
        marked = resumed.step(
            "episode-test",
            "mark_conversation_read",
            {"conversation_id": "C003"},
        )
        assert marked["result"]["last_read_ts"] >= unread["result"]["messages"][0]["ts"]
        assert resumed.step(
            "episode-test", "get_channel_messages", {"channel_id": "C003"}
        )["result"]["unread_count"] == 0

        persisted_mentions = export_state(resumed.db_path)["mentions"]
        assert any(
            mention["message_id"] == people_tag["result"]["id"]
            and mention["mention_type"] == "person"
            for mention in persisted_mentions
        )

        reset = resumed.reset(session_cookie="episode-next", seed=0)
        assert reset["session"]["turn"] == 0
        assert export_state(resumed.db_path)["virtual_time"] == initial_virtual_time
        try:
            resumed.state("episode-test")
        except KeyError:
            pass
        else:
            raise AssertionError("reset must discard the previous episode session")

    print("environment contract: ok")


if __name__ == "__main__":
    main()
