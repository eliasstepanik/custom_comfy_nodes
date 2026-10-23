"""Pure-python core of the CCN tilemap generator.

Modules (stdlib + numpy + PIL only; no torch, no comfy imports):
sanitize, llm_client, plan_prompt, plan_schema, plan_cache, expand,
pixelize, transitions, atlas.

Nodes under ``nodes/`` import submodules via ``from ..tilemap_core import x``;
inside this package only ``from . import x`` is used, so tests can import
``tilemap_core`` as a top-level package. This file imports nothing and has no
import-time side effects.
"""
