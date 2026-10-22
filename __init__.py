# Monorepo pack: every node group lives in nodes/<name>.py (one file per group).
# Each module defines NODE_CLASS_MAPPINGS and optionally NODE_DISPLAY_NAME_MAPPINGS.
# Modules are auto-discovered below; names starting with "_" are skipped.
import importlib
import os
import pkgutil

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

_nodes_dir = os.path.join(os.path.dirname(__file__), "nodes")

for _info in sorted(pkgutil.iter_modules([_nodes_dir]), key=lambda i: i.name):
    if _info.name.startswith("_"):
        continue
    try:
        _mod = importlib.import_module(f".nodes.{_info.name}", __name__)
    except Exception as _e:  # ponytail: one broken module must not take down the pack
        print(f"[custom_comfy_nodes] failed to load nodes.{_info.name}: {_e}")
        continue
    for _key, _cls in getattr(_mod, "NODE_CLASS_MAPPINGS", {}).items():
        if _key in NODE_CLASS_MAPPINGS:
            print(f"[custom_comfy_nodes] duplicate node '{_key}' in nodes.{_info.name}, keeping first")
            continue
        NODE_CLASS_MAPPINGS[_key] = _cls
        if _key in getattr(_mod, "NODE_DISPLAY_NAME_MAPPINGS", {}):
            NODE_DISPLAY_NAME_MAPPINGS[_key] = _mod.NODE_DISPLAY_NAME_MAPPINGS[_key]

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
