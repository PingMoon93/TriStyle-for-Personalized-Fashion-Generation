# TriStyle for Personalized Fashion Generation

This initial public release contains the **TriStyle Text-V1.1** preference component.
It converts legal, same-category user-history headlines into a compact attribute
preference prompt. It does not include datasets, checkpoints, generated images, or
training outputs.

## Text-V1.1

Text-V1.1 uses a frozen dictionary of 68 canonical English attributes in four groups:
colors/patterns, silhouettes/proportions, materials/textures, and construction/details.
The vocabulary was informed by iFashion training-history headlines and manually
cleaned and normalized. iFashion extraction is deterministic and does not use an LLM
or image-derived attributes.

For every target category, each headline contributes a canonical attribute at most
once. Support is then weighted by the original click frequency. The default prompt
keeps the two most-supported attributes in each group, with deterministic tie-breaking:

```text
A photo of {category}, on white background. User preferences: colors and patterns: black; materials and textures: knitted.
```

If no supported attribute or no same-category history exists, the component returns
only the category prompt. The API deliberately accepts no target item, target headline,
or full target outfit.

See [the Text-V1.1 protocol](docs/TEXT_PREFERENCE.md) for the data boundary and usage.

## Install and test

```bash
python -m pip install -e .
python -m unittest tests.test_text
```

The repository does not redistribute iFashion or Polyvore data. Configure paths to
assets obtained under their respective licenses.
