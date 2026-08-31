from forge.runtime.env import ForgeEnv
from forge.runtime.reward import FunctionRubric, RewardEngine
from forge.runtime.snapshot import EnvironmentSpec
from forge.runtime.transition import FunctionTransitionHandler, TransitionEngine
from forge.runtime.verifier import FunctionVerifier, VerifierEngine
from examples.gmail_env.initial_state import GmailInitialStateFactory
from examples.gmail_env.rewards.base import compute_gmail_reward
from examples.gmail_env.transitions.apply_label import apply_apply_label
from examples.gmail_env.transitions.archive_email import apply_archive_email
from examples.gmail_env.transitions.escalate_thread import apply_escalate_thread
from examples.gmail_env.transitions.mark_read import apply_mark_read
from examples.gmail_env.transitions.reply_email import apply_reply_email
from examples.gmail_env.transitions.send_email import apply_send_email
from examples.gmail_env.verifiers.archive_newsletter import verify_archive_newsletter
from examples.gmail_env.verifiers.escalate_billing_complaint import verify_escalate_billing_complaint
from examples.gmail_env.verifiers.label_urgent_request import verify_label_urgent_request
from examples.gmail_env.verifiers.reply_to_customer import verify_reply_to_customer


def build_gmail_env(max_steps: int = 20) -> ForgeEnv:
    spec = EnvironmentSpec(name="gmail_env", domain="email", max_steps=max_steps)

    te = TransitionEngine()
    te.register("reply_email", FunctionTransitionHandler(apply_reply_email))
    te.register("send_email", FunctionTransitionHandler(apply_send_email))
    te.register("archive_email", FunctionTransitionHandler(apply_archive_email))
    te.register("apply_label", FunctionTransitionHandler(apply_apply_label))
    te.register("mark_read", FunctionTransitionHandler(apply_mark_read))
    te.register("escalate_thread", FunctionTransitionHandler(apply_escalate_thread))

    ve = VerifierEngine()
    ve.register("reply_to_customer", FunctionVerifier(verify_reply_to_customer))
    ve.register("label_urgent_request", FunctionVerifier(verify_label_urgent_request))
    ve.register("archive_newsletter", FunctionVerifier(verify_archive_newsletter))
    ve.register("escalate_billing_complaint", FunctionVerifier(verify_escalate_billing_complaint))

    re = RewardEngine()
    re.set_default(FunctionRubric(compute_gmail_reward))

    return ForgeEnv(
        env_spec=spec,
        initial_state_provider=GmailInitialStateFactory(),
        transition_engine=te,
        verifier_engine=ve,
        reward_engine=re,
    )
