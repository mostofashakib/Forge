import copy
from forge.runtime.context import RuntimeContext
from forge.runtime.snapshot import InvalidActionError
from forge.runtime.transition import TransitionResult


def apply_escalate_thread(state: dict, action: dict, ctx: RuntimeContext) -> TransitionResult:
    thread_id = action["thread_id"]

    if thread_id not in state["threads"]:
        raise InvalidActionError(f"Thread '{thread_id}' not found", code="ENTITY_NOT_FOUND")

    new_state = copy.deepcopy(state)
    new_state["threads"][thread_id]["escalated"] = True
    for email_id in new_state["threads"][thread_id]["email_ids"]:
        new_state["emails"][email_id]["escalated"] = True

    return TransitionResult(
        state=new_state,
        events=[{
            "type": "thread_escalated",
            "entity_id": thread_id,
            "payload": {},
            "timestamp": ctx.clock.now().isoformat(),
        }],
    )
