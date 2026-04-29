import copy
from forge.runtime.context import RuntimeContext
from forge.runtime.snapshot import InvalidActionError
from forge.runtime.transition import TransitionResult


def apply_apply_label(state: dict, action: dict, ctx: RuntimeContext) -> TransitionResult:
    email_id = action["email_id"]
    label = action["label"]

    if email_id not in state["emails"]:
        raise InvalidActionError(f"Email '{email_id}' not found", code="ENTITY_NOT_FOUND")

    new_state = copy.deepcopy(state)
    labels = new_state["emails"][email_id]["labels"]
    if label not in labels:
        labels.append(label)

    return TransitionResult(
        state=new_state,
        events=[{
            "type": "label_applied",
            "entity_id": email_id,
            "payload": {"label": label},
            "timestamp": ctx.clock.now().isoformat(),
        }],
    )
