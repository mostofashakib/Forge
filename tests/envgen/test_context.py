from forge.envgen.context import EnvGenContext
from forge.envgen.schemas import GeneratedApp, GeneratedFile, FileContent
from forge.extraction.schemas import CompilerInput


def test_env_gen_context_defaults():
    ctx = EnvGenContext(
        env_name="test_env",
        description="A ticket system",
        compiler_input=CompilerInput(
            project_name="test_env", domain="tickets",
            entities=[], actions=[], tasks=[],
        ),
    )
    assert ctx.policy_requirements == ""
    assert ctx.reward_requirements == ""


def test_generated_app_serializes():
    app = GeneratedApp(files=[FileContent(path="main.py", content="# app")])
    assert app.files[0].path == "main.py"


def test_generated_file_serializes():
    f = GeneratedFile(content="def compute_reward(): pass")
    assert "compute_reward" in f.content
