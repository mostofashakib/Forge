# Two images from one context, so the agent's container never contains the
# world's source or its seed. The agent could otherwise re-seed a private copy
# of the workspace and read the answer the verifier is going to grade against.
FROM python:3.12-slim AS base
RUN apt-get update \
    && apt-get install -y --no-install-recommends bash procps util-linux \
    && rm -rf /var/lib/apt/lists/*
ENV PYTHONPATH=/opt
ENV PYTHONDONTWRITEBYTECODE=1

# --------------------------------------------------------------------------
# world: owns the workspace database and serves the tracker tools.
# --------------------------------------------------------------------------
FROM base AS world
RUN useradd --system --no-create-home --home-dir /var/lib/tasks \
        --shell /usr/sbin/nologin tasksd
COPY environment/task_sim /opt/task_sim
COPY environment/rl_env.py /opt/rl_env.py
COPY environment/bin/tasksd-entrypoint environment/bin/tasks-admin /usr/local/bin/
RUN chmod 0755 /usr/local/bin/tasksd-entrypoint /usr/local/bin/tasks-admin \
    && mkdir -p /var/lib/tasks \
    && python3 -m task_sim.service seed \
         --db /var/lib/tasks/tasks.db \
         --snapshot /var/lib/tasks/tasks_seed_snapshot.sql \
    && chown -R tasksd:tasksd /var/lib/tasks \
    && chmod 0700 /var/lib/tasks
WORKDIR /var/lib/tasks
ENTRYPOINT ["/usr/local/bin/tasksd-entrypoint"]

# --------------------------------------------------------------------------
# agent: the client half only. No seed, no service, no database.
# --------------------------------------------------------------------------
FROM base AS agent
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl jq nodejs npm \
    && npm install --global @anthropic-ai/claude-code@2.1.258 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -s /bin/bash agent
# Without an explicit shell, useradd gives the agent Debian's default /bin/sh,
# which is dash. Two things break: bashisms an agent writes in a Bash tool call
# fail ("[[: not found"), and Claude Code's shell snapshotting has no bash-shaped
# shell to capture, so it leaves $CLAUDE_CONFIG_DIR/shell-snapshots empty.
ENV SHELL=/bin/bash
# Exactly the modules a client needs: the wire format, the tool schemas, and the
# two front ends that speak them -- the MCP server Harbor registers for the
# agent's harness, and the CLI the oracle solution drives. task_sim.seed,
# .service, .server, .tracker and the database are deliberately absent here,
# which is why this is a file list and not a directory copy.
RUN mkdir -p /opt/task_sim
COPY environment/task_sim/__init__.py environment/task_sim/protocol.py \
     environment/task_sim/tool_definitions.py environment/task_sim/agent_cli.py \
     environment/task_sim/mcp_server.py environment/task_sim/admin_cli.py \
     /opt/task_sim/
# The verifier runs in this container as root and needs the seed to derive
# ground truth. It ships root-only: present for the grader, unreadable to the
# agent, the same way admin.sock is. PYTHONPATH order in tests/test.sh puts this
# ahead of the client-only package above.
COPY environment/task_sim /opt/grading/task_sim
# The verifier stack is grading code: root-only, in the same place as the answer
# key it reads, and absent from anything the agent can open.
COPY verifiers /opt/grading/verifiers
COPY environment/bin/tasks /usr/local/bin/tasks
COPY environment/bin/tasks-mcp /usr/local/bin/tasks-mcp
COPY environment/bin/tasks-admin /usr/local/bin/tasks-admin
# tasks and tasks-mcp are for the agent; tasks-admin is root-only here as
# defence in depth, on top of admin.sock's own 0600 permission.
RUN chmod 0755 /usr/local/bin/tasks /usr/local/bin/tasks-mcp \
    && chmod 0750 /usr/local/bin/tasks-admin \
    && chown -R root:root /opt/grading \
    && chmod -R go-rwx /opt/grading \
    && chmod 0700 /opt/grading \
    && mkdir -p /app \
    && chown agent:agent /app
WORKDIR /app
CMD ["sh", "-c", "sleep infinity"]
