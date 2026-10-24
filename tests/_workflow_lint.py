"""Graph linter functions for the shipped ComfyUI workflows (stdlib only, no torch / ComfyUI import).

Used by ``test_workflow_graph``. Checks node classes, links, slot types and widget values against ``SPEC``
(hand-copied from the live ``/object_info`` and the ``INPUT_TYPES`` / ``RETURN_TYPES`` in ``nodes/*.py``), and
that an API file and its UI twin (same stem) describe the same graph.
``CCN_WORKFLOW_ROOT`` overrides the directory that holds ``workflows/`` and ``example_workflows/``.
"""

import os

ROOT = os.environ.get("CCN_WORKFLOW_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# class -> (required inputs, optional inputs, outputs); "name:TYPE", COMBO/INT/FLOAT/BOOLEAN/STRING are widget types.
_N = "view:COMBO tile_size:COMBO max_tiles:INT model:COMBO model_override:STRING refresh_plan:BOOLEAN"
_S = "palette_size:INT seam_blend:FLOAT key_tolerance:INT transitions:BOOLEAN"
SPEC = {k: tuple(v.split() for v in vs) for k, vs in {
    "CCN_TilemapPlan": (f"prompt:STRING {_N}", "plan_mode:COMBO existing_plan:STRING",
                        "plan_json:STRING summary:STRING model_used:STRING"),
    "CCN_TilemapExpand": ("plan_json:STRING style_prefix:STRING seed:INT cell_px:INT", "model:STRING",
                          "prompt:STRING width:INT height:INT seed:INT job_json:STRING"),
    "CCN_TilemapReviewLoop": (
        "model:MODEL clip:CLIP vae:VAE job_json:STRING review_mode:COMBO review_model:COMBO max_rounds:INT "
        f"max_regen_per_round:INT max_attempts:INT pass_score:INT layout_source:COMBO {_S} steps:INT cfg:FLOAT "
        "sampler:COMBO scheduler:COMBO guidance:FLOAT negative:STRING on_review_error:COMBO",
        "images:IMAGE base_atlas:IMAGE base_alpha:MASK", "images:IMAGE job_json:STRING scene:IMAGE review_log:STRING"),
    "CCN_TilemapAssemble": (f"images:IMAGE job_json:STRING {_S.split(' ', 1)[0]} columns:INT spacing:INT "
                            f"{_S.split(' ', 1)[1]}", "base_atlas:IMAGE base_alpha:MASK",
                            "atlas:IMAGE alpha:MASK tiles:IMAGE metadata:STRING"),
    "CCN_SaveTilemap": ("atlas:IMAGE alpha:MASK metadata:STRING filename_prefix:STRING", "", ""),
    "CCN_ImageGridLabeled": ("images:IMAGE columns:INT padding:INT background:COMBO show_index:BOOLEAN", "", "grid:IMAGE"),
    "CheckpointLoaderSimple": ("ckpt_name:COMBO", "", "MODEL:MODEL CLIP:CLIP VAE:VAE"),
    "LoraLoader": ("model:MODEL clip:CLIP lora_name:COMBO strength_model:FLOAT strength_clip:FLOAT", "",
                   "MODEL:MODEL CLIP:CLIP"),
    "CLIPTextEncode": ("text:STRING clip:CLIP", "", "CONDITIONING:CONDITIONING"),
    "FluxGuidance": ("conditioning:CONDITIONING guidance:FLOAT", "", "CONDITIONING:CONDITIONING"),
    "EmptySD3LatentImage": ("width:INT height:INT batch_size:INT", "", "LATENT:LATENT"),
    "KSampler": ("model:MODEL seed:INT steps:INT cfg:FLOAT sampler_name:COMBO scheduler:COMBO positive:CONDITIONING "
                 "negative:CONDITIONING latent_image:LATENT denoise:FLOAT", "", "LATENT:LATENT"),
    "VAEDecode": ("samples:LATENT vae:VAE", "", "IMAGE:IMAGE"),
    "PreviewImage": ("images:IMAGE", "", ""),
    "PreviewAny": ("source:*", "", ""),
}.items()}
ENUMS = {("CCN_TilemapPlan", "view"): ["top-down", "three-quarter", "side-view"],
         ("CCN_TilemapPlan", "tile_size"): [8, 16, 24, 32, 48, 64]}
OUTPUT_NODES = ("CCN_SaveTilemap", "PreviewImage", "PreviewAny")


def _pairs(items):
    return [tuple(s.split(":")) for s in items]


def _ins(cls):
    req, opt, _ = SPEC[cls]
    return dict(_pairs(req) + _pairs(opt)), {n for n, _ in _pairs(req)}


def _outs(cls):
    return [t for _, t in _pairs(SPEC[cls][2])]


def _value_ok(cls, name, typ, v):
    if typ == "INT":
        ok = isinstance(v, int) and not isinstance(v, bool)
    elif typ == "FLOAT":
        ok = isinstance(v, (int, float)) and not isinstance(v, bool)
    elif typ == "BOOLEAN":
        ok = isinstance(v, bool)
    else:
        ok = isinstance(v, str) or (cls, name) in ENUMS
    return ok and v in ENUMS.get((cls, name), [v])


def _int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def check_api(g):
    """Return a list of error strings for an API-format graph."""
    errs = []
    if not isinstance(g, dict) or not g:
        return ["graph is not a non-empty object"]
    deps = {}
    for nid, n in g.items():
        if not (isinstance(n, dict) and isinstance(n.get("class_type"), str) and isinstance(n.get("inputs"), dict)):
            errs.append(f"node {nid}: needs class_type and inputs object")
            continue
        cls, inputs = n["class_type"], n["inputs"]
        known = cls in SPEC
        if not known:
            errs.append(f"node {nid}: unknown class_type {cls}")
        else:
            types, required = _ins(cls)
            errs += [f"node {nid} ({cls}): missing required input {m}" for m in sorted(required - set(inputs))]
            errs += [f"node {nid} ({cls}): unknown input {m}" for m in sorted(set(inputs) - set(types))]
        deps[nid] = set()
        for name, v in inputs.items():
            if not isinstance(v, list):
                typ = _ins(cls)[0].get(name) if known else None
                if typ and not _value_ok(cls, name, typ, v):
                    errs.append(f"node {nid} ({cls}).{name}: bad literal {v!r} for {typ}")
                continue
            if len(v) != 2 or not isinstance(v[0], str) or not _int(v[1]) or v[1] < 0:
                errs.append(f"node {nid}.{name}: malformed link {v!r}")
            elif v[0] not in g:
                errs.append(f"node {nid}.{name}: link to missing node {v[0]}")
            else:
                deps[nid].add(v[0])
                src = g[v[0]].get("class_type")
                if src not in SPEC:
                    continue
                outs = _outs(src)
                want = _ins(cls)[0].get(name) if known else None
                if v[1] >= len(outs):
                    errs.append(f"node {nid}.{name}: node {v[0]} ({src}) has no output {v[1]}")
                elif want is not None and want != "*" and outs[v[1]] != want:
                    errs.append(f"node {nid}.{name}: type {outs[v[1]]} -> {want} ({v[0]}:{v[1]})")
    if errs:
        return errs
    sinks = [i for i, n in g.items() if n["class_type"] in OUTPUT_NODES]
    if not sinks:
        errs.append("no output node (CCN_SaveTilemap/PreviewImage/PreviewAny)")
    seen, stack = set(), list(sinks)
    while stack:
        i = stack.pop()
        if i not in seen:
            seen.add(i)
            stack += deps[i]
    errs += [f"node {i}: dangling, reaches no output node" for i in g if i not in seen and sinks]
    state = {}

    def cyc(i):
        state[i] = 1
        for d in deps[i]:
            if state.get(d) == 1 or (d not in state and cyc(d)):
                return True
        state[i] = 2
        return False

    if any(i not in state and cyc(i) for i in g):
        errs.append("graph has a cycle")
    return errs


def check_ui(u):
    """Return a list of error strings for a LiteGraph-format workflow."""
    errs = []
    nodes, links = u.get("nodes"), u.get("links")
    if not (isinstance(nodes, list) and isinstance(links, list)):
        return ["nodes/links must be lists"]
    byid = {}
    for n in nodes:
        if n.get("id") in byid:
            errs.append(f"duplicate node id {n.get('id')}")
        byid[n.get("id")] = n
    lk = {}
    for l in links:
        if not (isinstance(l, list) and len(l) == 6):
            errs.append(f"malformed link {l!r}")
        elif l[0] in lk:
            errs.append(f"duplicate link id {l[0]}")
        else:
            lk[l[0]] = l
    if errs:
        return errs
    if u.get("last_node_id", -1) < max(byid, default=0):
        errs.append("last_node_id < max node id")
    if u.get("last_link_id", -1) < max(lk, default=0):
        errs.append("last_link_id < max link id")
    for n in nodes:
        nid, cls = n["id"], n.get("type")
        if cls not in SPEC:
            errs.append(f"node {nid}: unknown type {cls}")
            continue
        types, required = _ins(cls)
        ins, outs = n.get("inputs", []), n.get("outputs", [])
        names = [i.get("name") for i in ins]
        if len(set(names)) != len(names) or not required <= set(names) or not set(names) <= set(types):
            errs.append(f"node {nid} ({cls}): input names {names} do not fit the spec")
        for i in ins:
            want = types.get(i.get("name"))
            if i.get("type") != want:
                errs.append(f"node {nid}.{i.get('name')}: slot type {i.get('type')} != spec {want}")
        if [(o.get("name"), o.get("type")) for o in outs] != [(a, b) for a, b in _pairs(SPEC[cls][2])]:
            errs.append(f"node {nid} ({cls}): outputs differ from the spec")
        if n.get("properties", {}).get("Node name for S&R") != cls:
            errs.append(f"node {nid}: 'Node name for S&R' != {cls}")
        widgets = [i["name"] for i in ins if "widget" in i]
        want_n = len(widgets) + widgets.count("seed")
        if len(n.get("widgets_values", [])) != want_n:
            errs.append(f"node {nid} ({cls}): {len(n.get('widgets_values', []))} widgets_values, expected {want_n}")
        for k, i in enumerate(ins):
            lid = i.get("link")
            if lid is not None and (lid not in lk or lk[lid][3:5] != [nid, k]):
                errs.append(f"node {nid} input {k}: link {lid} missing or not pointing back")
        for k, o in enumerate(outs):
            have = sorted(o.get("links") or [])
            if have != sorted(l[0] for l in lk.values() if l[1] == nid and l[2] == k):
                errs.append(f"node {nid} output {k}: links list {have} disagrees with links table")
    for lid, (_, a, s, b, t, typ) in lk.items():
        if a not in byid or b not in byid:
            errs.append(f"link {lid}: missing endpoint node")
            continue
        outs, ins = byid[a].get("outputs", []), byid[b].get("inputs", [])
        if not (_int(s) and 0 <= s < len(outs) and _int(t) and 0 <= t < len(ins)):
            errs.append(f"link {lid}: slot out of range")
        elif outs[s].get("type") != typ or ins[t].get("type") not in (typ, "*"):
            errs.append(f"link {lid}: type {typ} vs {outs[s].get('type')} -> {ins[t].get('type')}")
        elif ins[t].get("link") != lid:
            errs.append(f"link {lid}: target input does not reference it")
    return errs


def api_edges(g):
    return {(v[0], v[1], nid, name) for nid, n in g.items() for name, v in n["inputs"].items() if isinstance(v, list)}


def ui_edges(u):
    byid = {n["id"]: n for n in u["nodes"]}
    return {(str(a), s, str(b), byid[b]["inputs"][t]["name"]) for _, a, s, b, t, _ in u["links"]}


def check_pair(g, u):
    """API graph and UI twin must have the same edges and the same non-linked widget values."""
    errs = []
    ae, ue = api_edges(g), ui_edges(u)
    errs += [f"only in API: {e}" for e in sorted(ae - ue)] + [f"only in UI: {e}" for e in sorted(ue - ae)]
    if {str(n["id"]) for n in u["nodes"]} != set(g):
        errs.append("node id sets differ")
    for n in u["nodes"]:
        api = g.get(str(n["id"]), {})
        if api.get("class_type") != n["type"]:
            errs.append(f"node {n['id']}: class differs")
            continue
        vals = iter(n["widgets_values"])
        for i in n["inputs"]:
            if "widget" not in i:
                continue
            v = next(vals)
            if i["name"] == "seed":
                next(vals)
            if i.get("link") is None and api["inputs"].get(i["name"]) != v:
                errs.append(f"node {n['id']}.{i['name']}: UI {v!r} != API {api['inputs'].get(i['name'])!r}")
    return errs


def check_review(g):
    """The review loop sits between VAEDecode and Assemble and is fed by the right loaders."""
    def of(cls):
        return [i for i, n in g.items() if n["class_type"] == cls]

    def src(nid, name):
        v = g[nid]["inputs"].get(name)
        return (g[v[0]]["class_type"], v[0], v[1]) if isinstance(v, list) and v[0] in g else None

    loops, asm = of("CCN_TilemapReviewLoop"), of("CCN_TilemapAssemble")
    if len(loops) != 1 or len(asm) != 1:
        return [f"need exactly one review loop and one assemble, got {len(loops)}/{len(asm)}"]
    r, a = loops[0], asm[0]
    lora2 = [i for i in of("LoraLoader") if (src(i, "model") or ("",))[0] == "LoraLoader"]
    ck, exp = of("CheckpointLoaderSimple"), of("CCN_TilemapExpand")
    want = [
        (r, "images", "VAEDecode", None, 0), (r, "job_json", "CCN_TilemapExpand", None, 4),
        (r, "model", "LoraLoader", lora2[0] if lora2 else None, 0), (r, "clip", "LoraLoader", lora2[0] if lora2 else None, 1),
        (r, "vae", "CheckpointLoaderSimple", ck[0] if ck else None, 2),
        (a, "images", "CCN_TilemapReviewLoop", r, 0), (a, "job_json", "CCN_TilemapReviewLoop", r, 1),
    ]
    errs = []
    for nid, name, cls, sid, slot in want:
        s = src(nid, name)
        if not s or s[0] != cls or s[2] != slot or (sid and s[1] != sid):
            errs.append(f"node {nid}.{name} should come from {cls}[{sid or '*'}] slot {slot}, got {s}")
    if len(lora2) != 1 or len(ck) != 1 or len(exp) != 1:
        errs.append("need one checkpoint, one Expand and exactly one LoraLoader chained after another")
    if not any(src(i, "images") == ("CCN_TilemapReviewLoop", r, 2) for i in of("PreviewImage")):
        errs.append("loop scene output (slot 2) feeds no PreviewImage")
    return errs
