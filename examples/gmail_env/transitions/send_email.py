import copy
from forge.runtime.context import RuntimeContext
from forge.runtime.transition import TransitionResult


def apply_send_email(state: dict, action: dict, ctx: RuntimeContext) -> TransitionResult:
    to = action["to"]
    subject = action["subject"]
    body = action["body"]
    now = ctx.clock.now().isoformat()

    thread_id = ctx.id_generator.next("thread")
    email_id = ctx.id_generator.next("email")

    new_email = {
        "id": email_id,
        "from_": state["users"][ctx.actor_id]["email"],
        "to": to,
        "subject": subject,
        "body": body,
        "labels": ["sent"],
        "archived": False,
        "thread_id": thread_id,
        "read": True,
        "created_at": now,
        "escalated": False,
    }
    new_thread = {
        "id": thread_id,
        "email_ids": [email_id],
        "escalated": False,
    }

    new_state = copy.deepcopy(state)
    new_state["emails"][email_id] = new_email
    new_state["threads"][thread_id] = new_thread

    return TransitionResult(
        state=new_state,
        events=[{
            "type": "email_sent",
            "entity_id": thread_id,
            "payload": {"email_id": email_id, "to": to},
            "timestamp": now,
        }],
    )
