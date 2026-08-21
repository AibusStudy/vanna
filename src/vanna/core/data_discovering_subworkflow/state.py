""" data discovery state 관리 """

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional


NodeStatus = Literal["success", "failed", "retry", "finish", "skipped"]
WorkflowStatus = Literal["success", "failed", "skipped", "fallback"]
