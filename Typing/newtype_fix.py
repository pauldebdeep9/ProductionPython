"""The same argument-order bug, with NewType applied."""
from __future__ import annotations
from typing import NewType

Query = NewType("Query", str)
DeploymentName = NewType("DeploymentName", str)


def search(query: Query, deployment: DeploymentName) -> list[str]:
    return []


def bug_arg_order_now_caught() -> list[str]:
    q = Query("units billed")
    d = DeploymentName("gpt-4o-mini")
    return search(d, q)   # <- swapped
