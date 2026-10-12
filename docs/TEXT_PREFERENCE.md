# Text-V1.1 preference summaries

## Data boundary

The component consumes the history permitted by the active split and aggregates only
items in the target category. It does not read the target item's headline, SID, image,
or complete ground-truth outfit. A target item that legitimately occurs in the user's
allowed history is not removed merely because its ID overlaps the prediction target.

The iFashion adapter reuses the existing item-ID/headline match: internal item IDs are
joined through `item_info.npy` to the original headlines. It never rematches text to
images. Dictionary development used training-history headlines only.

## Extraction and aggregation

`configs/text_v1_1_dictionary.json` defines 68 canonical English values and literal
Chinese/English aliases in this fixed order:

1. colors and patterns;
2. silhouettes and proportions;
3. materials and textures;
4. construction and details.

The extractor applies HTML decoding, Unicode NFKC normalization, promotion/brand/SKU
cleanup, explicit-negation handling, and longest-span matching. It records only
attributes explicitly supported by a headline. One headline contributes at most one
vote to the same canonical attribute. Votes are multiplied by stored click frequency,
then sorted by support descending and canonical name ascending. The default retains
two values per group.

## Prompt and tokenizer budget

The prefix is always:

```text
A photo of {category}, on white background.
```

Nonempty groups are appended as preference candidates in the fixed group order. Every
configured tokenizer is checked without truncation. If a prompt exceeds any budget,
whole low-support attributes are removed deterministically; attribute strings are
never cut. The category prompt is always retained.

`HistoryText.summarize(user, category)` returns the final prompt, full support counts,
selected values, removed values, tokenizer lengths, fallback status, and a cache key.
The cache fingerprint includes the version, dictionary, configuration, tokenizer,
split, history, user, category, and frequency map.

## Minimal use

```python
import json
from pathlib import Path
from cf_mmdit.text import AttributeExtractor, HistoryText

attribute_dictionary = json.loads(Path("configs/text_v1_1_dictionary.json").read_text())
extractor = AttributeExtractor(attribute_dictionary)
print(extractor.extract("黑色宽松针织长袖上衣"))
```

`HistoryText` additionally needs a catalog, split-bounded frequency history, headline
mapping, and a budget object implementing `fits(text)` and `lengths(text)`. For SD3,
use `TokenBudget(pipe, config)` to check both CLIP tokenizers and T5 independently.
