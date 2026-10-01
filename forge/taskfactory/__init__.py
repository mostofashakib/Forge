"""Create versioned batches of executable synthetic tasks for an environment.

Separate from environment building. Four steps: taxonomy, writer,
validation (static check, golden solution pass^k, cross-family LLM review),
and registry.
"""
