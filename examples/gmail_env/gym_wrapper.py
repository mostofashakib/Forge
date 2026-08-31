from forge.runtime.env import ForgeEnv
from forge.runtime.reward import RewardEngine
from forge.runtime.snapshot import EnvironmentSpec
from forge.runtime.transition import TransitionEngine
from forge.runtime.verifier import VerifierEngine
from examples.gmail_env.initial_state import GmailInitialStateFactory
from examples.gmail_env.rewards.base import GmailRubric
from examples.gmail_env.transitions.apply_label import ApplyLabelHandler
from examples.gmail_env.transitions.archive_email import ArchiveEmailHandler
from examples.gmail_env.transitions.escalate_thread import EscalateThreadHandler
from examples.gmail_env.transitions.mark_read import MarkReadHandler
from examples.gmail_env.transitions.reply_email import ReplyEmailHandler
from examples.gmail_env.transitions.send_email import SendEmailHandler
from examples.gmail_env.verifiers.archive_newsletter import ArchiveNewsletterVerifier
from examples.gmail_env.verifiers.escalate_billing_complaint import EscalateBillingComplaintVerifier
from examples.gmail_env.verifiers.label_urgent_request import LabelUrgentRequestVerifier
from examples.gmail_env.verifiers.reply_to_customer import ReplyToCustomerVerifier


def build_gmail_env(max_steps: int = 20) -> ForgeEnv:
    spec = EnvironmentSpec(name="gmail_env", domain="email", max_steps=max_steps)

    te = TransitionEngine()
    te.register("reply_email", ReplyEmailHandler())
    te.register("send_email", SendEmailHandler())
    te.register("archive_email", ArchiveEmailHandler())
    te.register("apply_label", ApplyLabelHandler())
    te.register("mark_read", MarkReadHandler())
    te.register("escalate_thread", EscalateThreadHandler())

    ve = VerifierEngine()
    ve.register("reply_to_customer", ReplyToCustomerVerifier())
    ve.register("label_urgent_request", LabelUrgentRequestVerifier())
    ve.register("archive_newsletter", ArchiveNewsletterVerifier())
    ve.register("escalate_billing_complaint", EscalateBillingComplaintVerifier())

    re = RewardEngine()
    re.set_default(GmailRubric())

    return ForgeEnv(
        env_spec=spec,
        initial_state_provider=GmailInitialStateFactory(),
        transition_engine=te,
        verifier_engine=ve,
        reward_engine=re,
    )
