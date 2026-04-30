from __future__ import annotations
from pydantic import BaseModel


class FileContent(BaseModel):
    path: str
    content: str


class GeneratedApp(BaseModel):
    files: list[FileContent]


class GeneratedFile(BaseModel):
    content: str
