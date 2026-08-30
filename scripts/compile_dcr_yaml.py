#!/usr/bin/env python3
"""
DCR YAML → XML_DCR_DATA compiler.

DEPRECATED: no longer part of the runtime load path. SecureAirlineAgent now
loads dcr1.yaml/dcr2.yaml directly via
thesis_dpm_secure_langgraph.constraints.AgentDCRConstraints.parse_from_yaml,
which compiles straight to an in-memory DataDcrGraph -- no XML step, no
checked-in .xml artifact to keep in sync by hand.

Kept only for optional ad hoc XML export (e.g. viewing a graph in an
external DCR-portal visualization tool). This script is a verified *subset*
of thesis_dpm_secure_langgraph's own dcr_yaml_compiler.py: that version adds
duplicate/unknown-event-id validation and correctly preserves descriptions
on dict-form void events (a case this script silently drops). Prefer calling
`thesis_dpm_secure_langgraph.constraints.write_xml`/`compile_to_xml` instead
of this script for anything beyond casual local use.

Converts the compact DCR YAML format to PM4Py-compatible XML_DCR_DATA files.
Labels and labelMappings are derived automatically (1-to-1 with event IDs).
Expression strings use plain < and > — the compiler handles XML escaping.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
YAML FORMAT REFERENCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

title: <string>

events:
  # Void event — a plain string value is the blocked-message description.
  # Use null (~) for no description.
  <id>: <description string>
  <id>: ~

  # Input data event — type present, no expr key → decision="?"
  <id>:
    type: int | bool
    description: <string>          # optional

  # Computed data event — type + expr → decision=<expr>
  # Write expressions with plain < > operators (not &lt; &gt;).
  <id>:
    type: int | bool
    expr: <expression string>
    description: <string>          # optional

# Unguarded relations — dict of source → list of targets.
# Omit the entire key if no relations of that type exist.
conditions:    { <src>: [<tgt>, ...] }
responses:     { <src>: [<tgt>, ...] }
excludes:      { <src>: [<tgt>, ...] }
includes:      { <src>: [<tgt>, ...] }
milestones:    { <src>: [<tgt>, ...] }
co_responses:  { <src>: [<tgt>, ...] }   # → <coresponces> in XML

# Guarded relations — list of [source, target, guard_expression] triples.
# Guard expressions also use plain < > operators.
guarded_conditions:    [[<src>, <tgt>, <guard>], ...]
guarded_responses:     [[<src>, <tgt>, <guard>], ...]
guarded_includes:      [[<src>, <tgt>, <guard>], ...]
guarded_excludes:      [[<src>, <tgt>, <guard>], ...]
guarded_milestones:    [[<src>, <tgt>, <guard>], ...]
guarded_no_responses:  [[<src>, <tgt>, <guard>], ...]

marking:
  executed: [<id>, ...]            # default: []
  included: all | [<id>, ...]      # default: all events in declaration order
  pending:  [<id>, ...]            # default: []
  values:
    <id>: <int or bool>            # initial event values; default: {}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXPRESSION LANGUAGE (decision= and guard=)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Atoms:        42  true  false  [EventId]
  Arithmetic:   +  -  *
  Comparison:   ==  <  >  <=  >=
  Boolean:      and  or  not  (higher precedence: not > and > or)
  Conditional:  if <cond> then <expr> else <expr>
  Grouping:     ( ... )

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Usage
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  python scripts/compile_dcr_yaml.py path/to/graph.yaml          # → graph.xml
  python scripts/compile_dcr_yaml.py graph.yaml out/graph.xml    # explicit dst
  python scripts/compile_dcr_yaml.py graph.yaml --print          # stdout only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import xml.etree.ElementTree as ET

try:
    import yaml
except ImportError:
    sys.exit("PyYAML is required: pip install pyyaml")


# ── helpers ───────────────────────────────────────────────────────────────────

_UNGUARDED: list[tuple[str, str, str]] = [
    # (yaml_key,       xml_section_tag, xml_child_tag)
    ("conditions",    "conditions",    "condition"),
    ("responses",     "responses",     "response"),
    ("excludes",      "excludes",      "exclude"),
    ("includes",      "includes",      "include"),
    ("milestones",    "milestones",    "milestone"),
    ("co_responses",  "coresponces",   "coresponse"),
]

_GUARDED: list[tuple[str, str, str]] = [
    ("guarded_conditions",   "guardedConditions",   "condition"),
    ("guarded_responses",    "guardedResponses",    "response"),
    ("guarded_includes",     "guardedIncludes",     "include"),
    ("guarded_excludes",     "guardedExcludes",     "exclude"),
    ("guarded_milestones",   "guardedMilestones",   "milestone"),
    ("guarded_no_responses", "guardedNoResponses",  "noresponse"),
]


def _event_kind(edef: object) -> str:
    """Return 'void', 'input', or 'computed' for an event definition."""
    if edef is None or isinstance(edef, str):
        return "void"
    if "expr" in edef:
        return "computed"
    if "type" in edef:
        return "input"
    return "void"


# ── core compiler ─────────────────────────────────────────────────────────────

def compile_yaml(src: dict) -> ET.Element:
    """
    Build an ElementTree rooted at <dcrgraph> from a parsed DCR YAML dict.

    ET handles all XML escaping automatically: plain < > in Python strings
    become &lt; &gt; in the serialised attribute values.
    """
    title = src.get("title", "")
    events_def: dict = src.get("events", {})
    marking_def: dict = src.get("marking", {})
    event_ids = list(events_def.keys())

    root = ET.Element("dcrgraph", title=title)

    # ── specification / resources ─────────────────────────────────────────
    spec = ET.SubElement(root, "specification")
    resources = ET.SubElement(spec, "resources")
    events_el = ET.SubElement(resources, "events")
    labels_el = ET.SubElement(resources, "labels")
    mappings_el = ET.SubElement(resources, "labelMappings")

    for eid, edef in events_def.items():
        kind = _event_kind(edef)
        attribs: dict[str, str] = {"id": eid}

        if kind == "void":
            if isinstance(edef, str) and edef:
                attribs["description"] = edef
        elif kind == "input":
            attribs["dataType"] = edef["type"]
            attribs["decision"] = "?"
            if "description" in edef:
                attribs["description"] = edef["description"]
        else:  # computed
            attribs["dataType"] = edef["type"]
            attribs["decision"] = edef["expr"]
            if "description" in edef:
                attribs["description"] = edef["description"]

        ET.SubElement(events_el, "event", **attribs)
        ET.SubElement(labels_el, "label", id=eid)
        ET.SubElement(mappings_el, "labelMapping", eventId=eid, labelId=eid)

    # ── specification / constraints ───────────────────────────────────────
    constraints = ET.SubElement(spec, "constraints")

    for yaml_key, xml_tag, child_tag in _UNGUARDED:
        rel_dict: dict = src.get(yaml_key) or {}
        section = ET.SubElement(constraints, xml_tag)
        for source, targets in rel_dict.items():
            for target in (targets or []):
                ET.SubElement(section, child_tag,
                              sourceId=str(source), targetId=str(target))

    for yaml_key, xml_tag, child_tag in _GUARDED:
        rel_list: list = src.get(yaml_key) or []
        section = ET.SubElement(constraints, xml_tag)
        for item in rel_list:
            if isinstance(item, (list, tuple)):
                source, target, guard = item
            else:
                source, target, guard = item["source"], item["target"], item["guard"]
            ET.SubElement(section, child_tag,
                          sourceId=str(source), targetId=str(target),
                          guard=str(guard))

    # ── runtime / marking ─────────────────────────────────────────────────
    runtime = ET.SubElement(root, "runtime")
    marking_el = ET.SubElement(runtime, "marking")

    executed_el = ET.SubElement(marking_el, "executed")
    for eid in marking_def.get("executed", []):
        ET.SubElement(executed_el, "event", id=str(eid))

    included_el = ET.SubElement(marking_el, "included")
    raw_included = marking_def.get("included", "all")
    included_ids = event_ids if raw_included == "all" else list(raw_included)
    for eid in included_ids:
        ET.SubElement(included_el, "event", id=str(eid))

    pending_el = ET.SubElement(marking_el, "pendingResponses")
    for eid in marking_def.get("pending", []):
        ET.SubElement(pending_el, "event", id=str(eid))

    values: dict = marking_def.get("values") or {}
    if values:
        values_el = ET.SubElement(marking_el, "eventValues")
        for eid, val in values.items():
            if isinstance(val, bool):
                str_val = "true" if val else "false"
            else:
                str_val = str(val)
            ET.SubElement(values_el, "eventValue", id=str(eid), value=str_val)

    return root


def to_xml_string(root: ET.Element) -> str:
    """Serialise an Element tree to a pretty-printed XML string."""
    ET.indent(root, space="  ")
    body = ET.tostring(root, encoding="unicode")
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + body + "\n"


# ── CLI ───────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Compile a DCR YAML graph definition to XML_DCR_DATA.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("src", help="Input YAML file")
    ap.add_argument(
        "dst", nargs="?",
        help="Output XML file (default: same path with .xml extension)",
    )
    ap.add_argument(
        "--print", dest="print_only", action="store_true",
        help="Print XML to stdout instead of writing a file",
    )
    args = ap.parse_args(argv)

    src_path = Path(args.src)
    if not src_path.exists():
        sys.exit(f"Error: source file not found: {src_path}")

    with open(src_path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    if not isinstance(data, dict):
        sys.exit("Error: YAML root must be a mapping")

    root = compile_yaml(data)
    xml_str = to_xml_string(root)

    if args.print_only:
        print(xml_str, end="")
        return

    dst_path = Path(args.dst) if args.dst else src_path.with_suffix(".xml")
    dst_path.write_text(xml_str, encoding="utf-8")
    print(f"Compiled {src_path} → {dst_path}")


if __name__ == "__main__":
    main()
