"""A ward handover the agent cannot complete alone.

The smallest environment that shows why simulated people matter. The agent's
task is to get a patient discharged, and the discharge is not the agent's to
make: a supervisor has to approve it, and the supervisor will not approve
anything undocumented. So the agent has to ask, wait, and answer a question
before the task can close.

Run it:

    from examples.clinical_handoff.env import build_ward_env
    env = build_ward_env()
    obs, info = env.reset(seed=1)
    print(info["personas"])
"""
from __future__ import annotations

import copy

from forge.contracts import InitialStateProvider, ToolParam
from forge.contracts.persona import PersonaPopulation
from forge.personas.archetypes import archetype
from forge.runtime.env import ForgeEnv
from forge.runtime.env_builder import EnvBuilder
from forge.runtime.transition import TransitionResult


class WardInitialState(InitialStateProvider):
    """One patient, awaiting a discharge nobody has approved."""

    def reset(self, ctx, *, seed: int | None, options: dict) -> dict:
        return {
            "patients": {
                "pt_0000": {
                    "id": "pt_0000",
                    "name": "R. Mbeki",
                    "status": "awaiting_discharge",
                    "discharge_approved": 0,
                }
            },
            "messages": {},
        }


def send_message(state, action, ctx):
    """Anyone in the ward — agent or persona — writes to the shared board."""
    new_state = copy.deepcopy(state)
    message_id = ctx.id_generator.next("msg")
    new_state["messages"][message_id] = {
        "id": message_id,
        "author": ctx.actor_id,
        "to": action.get("to", ""),
        "body": action.get("body", ""),
    }
    return TransitionResult(
        state=new_state,
        events=[
            {
                "type": "message_sent",
                "entity_id": message_id,
                "recipient": action.get("to", ""),
            }
        ],
    )


def approve_discharge(state, action, ctx):
    """Only the supervisor is permitted this action — see the roster below."""
    new_state = copy.deepcopy(state)
    patient = new_state["patients"]["pt_0000"]
    patient["discharge_approved"] = 1
    patient["status"] = "discharged"
    return TransitionResult(
        state=new_state,
        events=[{"type": "discharge_approved", "entity_id": "pt_0000"}],
    )


def discharge_was_approved(state, trajectory, task) -> bool:
    return state["patients"]["pt_0000"]["discharge_approved"] == 1


def ward_population() -> PersonaPopulation:
    """Two people, with deliberately different reasons for existing.

    Both start from a domain-neutral disposition and are dressed for this ward.
    That separation is the intended usage: the library knows what a gatekeeper
    is like, and only this environment knows that its gatekeeper is a shift
    supervisor named Alan who signs off discharges.

    The supervisor is the gate — the agent cannot approve a discharge itself,
    so the task is unreachable without asking someone. The nurse is noise with
    a purpose: she volunteers information on her own initiative, which is what
    teaches an agent to read messages it did not ask for.
    """
    return PersonaPopulation(
        enabled=True,
        driver="scripted",
        max_actions_per_step=1,
        roster=[
            archetype(
                "gatekeeper",
                persona_id="supervisor",
                name="Alan Whitmore",
                role="shift supervisor",
                backstory=(
                    "Answerable for anything that goes wrong on the shift, and "
                    "so unwilling to approve a discharge he cannot justify to "
                    "the review afterwards."
                ),
                goals=["Keep the ward auditable", "Discharge nobody undocumented"],
                allowed_actions=["send_message", "approve_discharge"],
                wake_on=["send_message"],
                latency_steps=2,
            ),
            archetype(
                "meticulous_checker",
                persona_id="nurse",
                name="Priya Raman",
                role="charge nurse",
                backstory=(
                    "Has caught three medication errors this month and checks "
                    "everything twice. Knows the ward's practical details "
                    "better than anyone."
                ),
                goals=["Keep patients safe", "Make sure the chart matches reality"],
                allowed_actions=["send_message"],
                wake_on=["send_message"],
                activity=20,
            ),
        ],
    )


def build_ward_env(max_steps: int = 20, with_personas: bool = True) -> ForgeEnv:
    builder = (
        EnvBuilder("clinical_handoff", domain="clinical", max_steps=max_steps)
        .with_initial_state(WardInitialState())
        .with_transition(
            "send_message",
            send_message,
            description="Message someone on the ward",
            params=[
                ToolParam(name="to", type="string", description="who to message"),
                ToolParam(name="body", type="string", description="what to say"),
            ],
        )
        .with_transition(
            "approve_discharge",
            approve_discharge,
            description="Approve the patient's discharge",
        )
        .with_verifier("discharge_patient", discharge_was_approved)
        .with_default_task(
            {
                "id": "discharge_patient",
                "objective": (
                    "Get R. Mbeki discharged. You cannot approve a discharge "
                    "yourself — the shift supervisor has to, and will not "
                    "approve anything undocumented."
                ),
            }
        )
    )
    if with_personas:
        builder = builder.with_personas(ward_population())
    return builder.build()
