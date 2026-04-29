import copy
from forge.runtime.context import RuntimeContext
from forge.runtime.snapshot import InvalidActionError
from forge.runtime.transition import TransitionResult


def apply_reply_email(state: dict, action: dict, ctx: RuntimeContext) -> TransitionResult:
    thread_id = action["thread_id"]
    body = action["body"]

    if thread_id not in state["threads"]:
        raise InvalidActionError(f"Thread '{thread_id}' not found", code="ENTITY_NOT_FOUND")

    thread = state["threads"][thread_id]
    original_email_id = thread["email_ids"][0]
    original_email = state["emails"][original_email_id]

    new_email_id = ctx.id_generator.next("email")
    new_email = {
        "id": new_email_id,
        "from_": state["users"][ctx.actor_id]["email"],
        "to": original_email["from_"],
        "subject": f"Re: {original_email['subject']}",
        "body": body,
        "labels": ["sent"],
        "archived": False,
        "thread_id": thread_id,
        "read": True,
        "created_at": ctx.clock.now().isoformat(),
        "escalated": False,
    }

    new_state = copy.deepcopy(state)
    new_state["emails"][new_email_id] = new_email
    new_state["threads"][thread_id]["email_ids"].append(new_email_id)

    return TransitionResult(
        state=new_state,
        events=[{
            "type": "email_replied",
            "entity_id": thread_id,
            "payload": {"email_id": new_email_id},
            "timestamp": ctx.clock.now().isoformat(),
        }],
    )
