from __future__ import annotations
from pathlib import Path
import jinja2

_TEMPLATES_DIR = Path(__file__).parent.parent.parent / "templates"


def _pascal_case(s: str) -> str:
    return "".join(w.title() for w in s.split("_"))


def _py_type(field) -> str:
    """Convert a FieldDef or ActionParam to a Python type annotation string."""
    t = getattr(field, "type", "str")
    vals = getattr(field, "values", None)
    nullable = getattr(field, "nullable", False)
    if t == "enum" and vals:
        inner = ", ".join(repr(v) for v in vals)
        base = f"Literal[{inner}]"
    elif t == "integer":
        base = "int"
    elif t == "boolean":
        base = "bool"
    elif t == "list":
        base = "list"
    elif t == "dict":
        base = "dict"
    else:
        base = "str"
    return f"{base} | None" if nullable else base


def _default_repr(field) -> str | None:
    default = getattr(field, "default", None)
    t = getattr(field, "type", "str")
    if default is None and not getattr(field, "nullable", False):
        if t == "list":
            return "Field(default_factory=list)"
        return None
    if default is None:
        return "None"
    if isinstance(default, str):
        return repr(default)
    return str(default).lower() if isinstance(default, bool) else str(default)


class BaseGenerator:
    def __init__(self) -> None:
        loader = jinja2.FileSystemLoader(str(_TEMPLATES_DIR))
        self._jinja = jinja2.Environment(
            loader=loader,
            keep_trailing_newline=True,
            trim_blocks=True,
            lstrip_blocks=True,
        )
        self._jinja.filters["pascal_case"] = _pascal_case
        self._jinja.filters["py_type"] = _py_type
        self._jinja.filters["default_repr"] = _default_repr
        self._jinja.filters["plural"] = lambda s: s + "s"

    def render(self, template_name: str, **ctx: object) -> str:
        return self._jinja.get_template(template_name).render(**ctx)
