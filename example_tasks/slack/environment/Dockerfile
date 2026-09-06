# Two images from one context, so the agent's container never contains the
# world's source or its seed. The agent could otherwise re-seed a private copy
# of the workspace and read private channels it is not a member of.
FROM python:3.12-slim AS base
RUN apt-get update \
    && apt-get install -y --no-install-recommends bash procps util-linux \
    && rm -rf /var/lib/apt/lists/*
ENV PYTHONPATH=/opt
ENV PYTHONDONTWRITEBYTECODE=1

# --------------------------------------------------------------------------
# world: owns the workspace database and serves the Slack tools.
# --------------------------------------------------------------------------
FROM base AS world
RUN useradd --system --no-create-home --home-dir /var/lib/slack \
        --shell /usr/sbin/nologin slackd
COPY environment/slack_sim /opt/slack_sim
COPY environment/rl_env.py /opt/rl_env.py
COPY environment/bin/slackd-entrypoint environment/bin/slack-admin /usr/local/bin/
RUN chmod 0755 /usr/local/bin/slackd-entrypoint /usr/local/bin/slack-admin \
    && mkdir -p /var/lib/slack \
    && python3 -m slack_sim.service seed \
         --db /var/lib/slack/slack.db \
         --snapshot /var/lib/slack/slack_seed_snapshot.sql \
    && chown -R slackd:slackd /var/lib/slack \
    && chmod 0700 /var/lib/slack
WORKDIR /var/lib/slack
ENTRYPOINT ["/usr/local/bin/slackd-entrypoint"]

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
# Exactly the modules a client needs: the wire format, the tool schemas, and
# the two front ends that speak them -- the MCP server Harbor registers for the
# agent's harness, and the CLI the oracle solution drives. slack_sim.seed,
# .service, .models and the database are deliberately absent from this image.
RUN mkdir -p /opt/slack_sim
COPY environment/slack_sim/__init__.py environment/slack_sim/protocol.py \
     environment/slack_sim/tool_definitions.py environment/slack_sim/agent_cli.py \
     environment/slack_sim/mcp_server.py environment/slack_sim/admin_cli.py /opt/slack_sim/
# The verifier runs in this container as root and needs the seed to derive
# ground truth. It is shipped root-only: present for the grader, unreadable to
# the agent, the same way admin.sock is. PYTHONPATH order in tests/test.sh puts
# this ahead of the client-only package above.
COPY environment/slack_sim /opt/grading/slack_sim
# The verifier stack is grading code: root-only, in the same place as the answer
# key it reads, and absent from anything the agent can open.
COPY verifiers /opt/grading/verifiers
COPY environment/bin/slack /usr/local/bin/slack
COPY environment/bin/slack-mcp /usr/local/bin/slack-mcp
COPY environment/bin/slack-admin /usr/local/bin/slack-admin
# slack and slack-mcp are for the agent; slack-admin is root-only here as
# defence in depth, on top of admin.sock's own 0600 permission.
RUN chmod 0755 /usr/local/bin/slack /usr/local/bin/slack-mcp \
    && chmod 0750 /usr/local/bin/slack-admin \
    && chown -R root:root /opt/grading \
    && chmod -R go-rwx /opt/grading \
    && chmod 0700 /opt/grading \
    && mkdir -p /app \
    && chown agent:agent /app
WORKDIR /app
CMD ["sh", "-c", "sleep infinity"]
