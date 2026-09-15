"""Deliberately broken code. `07_mypy_in_practice.py` runs mypy on this file
and prints the real output. Every error below is a bug pattern from real
LLM-pipeline code."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    score: float


def find_chunk(chunk_id: str) -> Chunk | None:
    return None


# BUG 1 — Optional used without a check. The single largest category.
def bug_optional() -> str:
    c = find_chunk("c1")
    return c.doc_id


# BUG 2 — argument order swapped after a refactor. Both are str, so tests
# with symmetric fixtures pass.
def search(query: str, deployment: str) -> list[Chunk]:
    return []


def bug_arg_order() -> list[Chunk]:
    query = "units billed"
    deployment = "gpt-4o-mini"
    return search(deployment, query)


# BUG 3 — a path that returns None from a function annotated -> Chunk.
def bug_missing_return(chunk_id: str) -> Chunk:
    if chunk_id.startswith("c"):
        return Chunk(chunk_id, "doc", 1.0)


# BUG 4 — comparing a Literal against a value outside its set. The branch is
# dead and no test will ever cover it.
from typing import Literal

FinishReason = Literal["stop", "length"]


def bug_impossible_branch(reason: FinishReason) -> str:
    if reason == "content_filter":
        return "filtered"
    return "ok"


# BUG 5 — list invariance. list[SpecialChunk] is not list[Chunk].
@dataclass
class ScoredChunk(Chunk):
    rank: int = 0


def rank_all(chunks: list[Chunk]) -> None:
    chunks.append(Chunk("x", "y", 0.0))


def bug_invariance(special: list[ScoredChunk]) -> None:
    rank_all(special)


# BUG 6 — dict access typed as the wrong thing.
def bug_wrong_type(scores: dict[str, float]) -> int:
    return scores["c1"]


# BUG 7 — awaiting something that is not awaitable.
import asyncio


def not_async() -> int:
    return 1


async def bug_await() -> int:
    return await not_async()
