# Annotation protocols

This repository supports two related but distinct annotation units.

## Public RGB: PID-level dense text

The annotation unit is one person identity pooled across all available RGB
images and official splits. The pipeline:

1. Groups images by PID and removes junk identities.
2. Selects readable and complementary images across cameras and splits.
3. Runs four evidence branches: overview, head/face, body, and
   garment-structure/items.
4. Conservatively aligns the cues and writes one dense English description.

The published record intentionally contains only `pid` and `dense_en`.

## RGB/IR changing-clothes: group-level diverse text

The annotation unit is `(pid, clothes_id, modality)`. RGB and infrared images
are never mixed in one visual evidence request. The same four evidence branches
are used, followed by modality-aware alignment:

- RGB descriptions may use stable, clearly visible colors.
- IR descriptions must omit color and use shape, structure, reflectance,
  accessories, footwear, and carried-item cues instead.

Each published group contains one dense description and ten consistent caption
variants. The first five use a stable head-to-toe order; the other five vary
wording and cue order without changing the evidence.

## Safety constraints

The prompts prohibit identity claims and unsupported sensitive attributes.
They also prohibit filenames, local paths, dataset internals, API details,
background text, and annotation-process language from entering release text.
Weak or conflicting observations are omitted rather than guessed.
