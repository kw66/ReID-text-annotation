from __future__ import annotations

import json
from typing import Any


SYSTEM_PROMPT = """You create evidence-grounded person ReID annotations.
Use only visible appearance cues supported by the supplied images. Do not infer a
person's name, occupation, nationality, ethnicity, health, religion, or other
sensitive personal facts. Do not transcribe background text. Never mention image
filenames, filesystem paths, API details, prompts, cameras, datasets, or the
annotation process. If a cue is ambiguous, omit it or put it in uncertain_cues.
Return only the requested JSON object."""


BRANCH_INSTRUCTIONS = {
    "overview": (
        "Describe the stable overall appearance: coarse body build, dominant "
        "clothing layout, major accessories, bags, and carried items."
    ),
    "head_face": (
        "Inspect only reliable head-level cues such as hairstyle, headwear, and "
        "eyewear. Avoid facial identity claims and fine anatomy guesses."
    ),
    "body": (
        "Inspect coarse body shape, shoulder width, limb proportions, posture, "
        "and repeated gait cues. Omit one-frame actions."
    ),
    "structure_item": (
        "Inspect garment structure and local retrieval cues: collar, hood, zipper, "
        "pockets, sleeves, hem, pattern placement, bag type, footwear, and carried items."
    ),
}


def _task_rule(task: str) -> str:
    rules = {
        "traditional": (
            "Summarize stable identity-level appearance across the available RGB views."
        ),
        "anytime": (
            "The images share one person, clothing label, and modality within an anytime ReID group. "
            "Use only cues that are valid for this group."
        ),
        "clothes-changing": (
            "This is a changing-clothes group. Describe only the current clothing label and do not "
            "merge garments from another outfit."
        ),
        "visible-infrared": (
            "This is one modality group from a visible-infrared dataset. Do not transfer RGB-only "
            "color evidence into infrared text."
        ),
    }
    return rules[task]


def evidence_prompt(branch: str, modality: str, task: str) -> str:
    if branch not in BRANCH_INSTRUCTIONS:
        raise ValueError(f"Unknown evidence branch: {branch}")
    modality_rule = (
        "These are infrared images. Do not report color, skin tone, or hair color. "
        "Use only modality-stable shape, structure, reflectance, and item cues."
        if modality == "ir"
        else
        "These are RGB images. Mention color only when it is clear and consistent "
        "across the evidence images."
    )
    return f"""All supplied images show the same annotation group.
{_task_rule(task)}
{modality_rule}

Task: {BRANCH_INSTRUCTIONS[branch]}

Return this JSON shape:
{{
  "branch": "{branch}",
  "cues": ["short evidence-grounded cue", "..."],
  "uncertain_cues": ["ambiguous cue to exclude from final text", "..."]
}}
Keep cues concise. Use empty arrays when no reliable cue is visible."""


def synthesis_prompt(
    *,
    annotation_format: str,
    modality: str,
    task: str,
    evidence: dict[str, Any],
) -> str:
    evidence_text = json.dumps(evidence, ensure_ascii=False, sort_keys=True)
    modality_rule = (
        "The final text must not contain color terms because this group is infrared."
        if modality == "ir"
        else
        "Use color terms only when multiple evidence branches support them."
    )
    if annotation_format == "dense":
        output_rule = """Return exactly:
{"dense_en": "one compact English identity-level description"}
The description should merge stable cues across views, remove conflicts and
redundancy, and avoid camera-specific or momentary details."""
    else:
        output_rule = """Return exactly:
{
  "dense_en": "one compact English group-level description",
  "captions": [
    {"style": "dense_full", "en": "caption"},
    {"style": "person_core", "en": "caption"},
    {"style": "outfit_core", "en": "caption"},
    {"style": "distinctive_mix", "en": "caption"},
    {"style": "retrieval_balanced", "en": "caption"},
    {"style": "random_a", "en": "caption"},
    {"style": "random_b", "en": "caption"},
    {"style": "random_c", "en": "caption"},
    {"style": "random_d", "en": "caption"},
    {"style": "random_e", "en": "caption"}
  ]
}
Create 10 non-contradictory captions: the first five should follow a stable
head-to-toe ordering, while the last five should vary wording and cue order.
Every caption must remain consistent with the same verified evidence."""
    return f"""Synthesize verified ReID evidence into release text.
{_task_rule(task)}
{modality_rule}
Ignore every item in uncertain_cues. Resolve conflicts conservatively and omit
weak claims. Do not add facts that are absent from the cue lists.

Verified branch evidence:
{evidence_text}

{output_rule}"""
