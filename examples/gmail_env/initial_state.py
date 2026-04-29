from forge.runtime.context import RuntimeContext


class GmailInitialStateFactory:
    def create(self, ctx: RuntimeContext, options: dict) -> dict:
        # Advance the clock by a seed-dependent offset so each seed produces a
        # unique timestamp (and therefore a unique state hash).
        ctx.clock.advance(ctx.seed % 86400)

        user_id = ctx.id_generator.next("user")
        ctx.actor_id = user_id

        thread_id = ctx.id_generator.next("thread")
        email_id = ctx.id_generator.next("email")
        now = ctx.clock.now().isoformat()

        scenario = options.get("scenario", "refund_request")

        if scenario == "newsletter":
            subject = "Your weekly digest"
            body = "Here are this week's top stories."
            sender = "newsletter@digest.com"
            labels = ["inbox"]
        elif scenario == "billing_complaint":
            subject = "Billing issue - urgent"
            body = "I have been charged incorrectly. This is urgent."
            sender = "customer@example.com"
            labels = ["inbox"]
        else:
            subject = "Refund request"
            body = "I was charged twice for my order."
            sender = "customer@example.com"
            labels = ["inbox"]

        return {
            "users": {
                user_id: {
                    "id": user_id,
                    "email": "agent@example.com",
                    "role": "support_agent",
                }
            },
            "emails": {
                email_id: {
                    "id": email_id,
                    "from_": sender,
                    "to": "support@example.com",
                    "subject": subject,
                    "body": body,
                    "labels": labels,
                    "archived": False,
                    "thread_id": thread_id,
                    "read": False,
                    "created_at": now,
                    "escalated": False,
                }
            },
            "threads": {
                thread_id: {
                    "id": thread_id,
                    "email_ids": [email_id],
                    "escalated": False,
                }
            },
            "labels": {
                "inbox": {"id": "inbox", "name": "Inbox"},
                "sent": {"id": "sent", "name": "Sent"},
                "urgent": {"id": "urgent", "name": "Urgent"},
                "newsletter": {"id": "newsletter", "name": "Newsletter"},
            },
            "actor_id": user_id,
        }
