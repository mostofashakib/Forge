import copy
from forge.runtime.context import RuntimeContext
from forge.runtime.snapshot import InvalidActionError
from forge.runtime.transition import TransitionResult


def apply_archive_email(state: dict, action: dict, ctx: RuntimeContext) -> TransitionResult:
    email_id = action["email_id"]

    if email_id not in state["emails"]:
        raise InvalidActionError(f"Email '{email_id}' not found", code="ENTITY_NOT_FOUND")

    new_state = copy.deepcopy(state)
    new_state["emails"][email_id]["archived"] = True

    return TransitionResult(
        state=new_state,
        events=[{
            "type": "email_archived",
            "entity_id": email_id,
            "payload": {},
            "timestamp": ctx.clock.now().isoformat(),
        }],
    )
