"""Text-V1.1: deterministic, history-only attribute preference summaries."""
from collections import Counter, OrderedDict
from pathlib import Path
import html
import json
import re
import unicodedata

from .text_utils import digest, fingerprint, integer_id, trusted_npy

VERSION = 'Text-V1.1'
GROUPS = ('colors and patterns', 'silhouettes and proportions',
          'materials and textures', 'construction and details')


def category_prompt(category_name):
    """Canonical Full category prompt shared by both text-preference modes."""
    return f"A photo of {category_name}, on white background."


def normalized(text):
    return unicodedata.normalize('NFKC', html.unescape(text)).casefold()


def literal_pattern(alias):
    # Latin aliases require Latin/digit boundaries, also when adjacent to Chinese.
    alias = normalized(alias)
    return re.compile(('(?<![a-z0-9])' if re.match('[a-z0-9]', alias) else '') +
                      re.escape(alias) + ('(?![a-z0-9])' if re.search('[a-z0-9]$', alias) else ''))


class AttributeExtractor:
    def __init__(self, dictionary):
        if dictionary['version'] != VERSION: raise ValueError('Unsupported text dictionary version')
        self.rules = []
        for group in GROUPS:
            for attribute, aliases in dictionary['groups'][group].items():
                if not re.fullmatch('[a-z][a-z -]*', attribute): raise ValueError('Attributes must be canonical English')
                for alias in aliases:
                    self.rules.append((group, attribute, literal_pattern(alias)))
        self.ignored = [literal_pattern(x) for x in dictionary.get('ignore_phrases', [])]
        self.blocked = [literal_pattern(x) for x in dictionary.get('blocked_phrases', [])]

    def clean(self, headline):
        text = normalized(headline)
        text = re.sub(r'https?://\S+', ' ', text)
        for pattern in self.ignored + self.blocked: text = pattern.sub(' ', text)
        # Remove SKU/year tokens without joining surrounding words or destroying V领.
        text = re.sub(r'(?<![a-z0-9])[a-z]*\d+[a-z0-9-]*(?![a-z0-9])', ' ', text)
        return re.sub(r'\s+', ' ', text).strip()

    def extract(self, headline):
        text = self.clean(headline); matches = []
        for group, attribute, pattern in self.rules:
            for match in pattern.finditer(text):
                prefix = text[max(0, match.start()-12):match.start()]
                if re.search(r'(?:无|非|不带|没有|仿|仿制|非真|not |no |without |faux )\s*$', prefix): continue
                matches.append((match.start(), match.end(), group, attribute))
        # Longest phrase wins on overlapping spans: 浅蓝色 does not also count 蓝色.
        occupied = set(); found = {g:set() for g in GROUPS}
        for start, end, group, attribute in sorted(matches, key=lambda m: (-(m[1]-m[0]), m[0], m[2], m[3])):
            span = set(range(start,end))
            if span & occupied: continue
            occupied.update(span); found[group].add(attribute)
        return {g:sorted(values) for g,values in found.items()}


def load_headlines(config, histories, catalog):
    """Use an existing ID join only. No images, targets, or outfit rows accepted."""
    wanted = {i for cats in histories.values() for counts in cats.values() for i in counts}
    source = Path(config['headline_path'])
    result = {}; manifest = {'headline_sha256':digest(source), 'format':config['headline_format']}
    if config['headline_format'] == 'internal_json':
        raw = json.loads(source.read_text())
        for i in wanted:
            if i in raw:
                if not isinstance(raw[i],str): raise ValueError('Expected internal item ID -> original headline string')
                result[i] = raw[i]
    elif config['headline_format'] == 'matched_jsonl_original_id':
        mapping_path = Path(config['item_info_path']); info = trusted_npy(mapping_path)
        mapping = {}
        for i in wanted:
            meta = info[integer_id(i)]
            if meta['category'] != catalog['category_names'][catalog['item_categories'][i]]:
                raise ValueError('Existing item/category mapping mismatch')
            original = meta['original iid']
            if original in mapping: raise ValueError('Ambiguous existing original ID mapping')
            mapping[original] = i
        with source.open() as stream:
            for line in stream:
                row = json.loads(line); i = mapping.get(row['item_id'])
                if i is None: continue
                title = row['text']
                if not isinstance(title,str): raise ValueError('Original headline must be a string')
                if row['category_name'] != catalog['category_names'][catalog['item_categories'][i]]:
                    raise ValueError('Matched headline category differs from existing catalog')
                if i in result and result[i] != title: raise ValueError('Conflicting existing headlines for one item')
                result[i] = title
        manifest['item_info_sha256'] = digest(mapping_path)
    else: raise ValueError('Explicit supported headline_format required')
    manifest.update(history_items=len(wanted), available_headlines=sum(bool(t.strip()) for t in result.values()))
    return result, manifest


class TokenBudget:
    def __init__(self, pipe, config):
        self.tokenizers = [pipe.tokenizer, pipe.tokenizer_2, pipe.tokenizer_3]
        if any(t is None for t in self.tokenizers): raise ValueError('Text-V1.1 requires all three SD3 tokenizers')
        self.t5_length = config.get('t5_max_sequence_length',256)
        if not isinstance(self.t5_length,int) or not 1 <= self.t5_length <= 512: raise ValueError('T5 length must be 1..512')
        clip_length = pipe.tokenizer_max_length
        self.limits = [min(clip_length, t.model_max_length) for t in self.tokenizers[:2]] + [min(self.t5_length, self.tokenizers[2].model_max_length)]
        cap = config.get('max_tokens',77)
        if not isinstance(cap,int) or cap<1: raise ValueError('max_tokens must be positive')
        self.limits = [min(cap,n) for n in self.limits]
        self.manifest = {'limits':self.limits,'t5_max_sequence_length':self.t5_length,
                         'tokenizers':[{'class':type(t).__name__, 'vocab_sha256':fingerprint(t.get_vocab()),
                         'special_tokens':str(t.special_tokens_map), 'init':str(t.init_kwargs)} for t in self.tokenizers]}

    def lengths(self, text):
        return [len(t(text,add_special_tokens=True,truncation=False)['input_ids']) for t in self.tokenizers]

    def fits(self, text): return all(n<=limit for n,limit in zip(self.lengths(text),self.limits))


class CategoryOnlyText:
    """Text-V1.1 encoding contract without reading item attributes or history text."""
    def __init__(self, config, catalog, split, budget):
        if config.get('version') != VERSION: raise ValueError('Text condition version must be Text-V1.1')
        if split not in ('train','valid','test'): raise ValueError('Explicit original split required')
        self.config,self.catalog,self.split,self.budget=config,catalog,split,budget
        encoding={k:v for k,v in config.items() if k in ('version','max_tokens','t5_max_sequence_length')}
        self.assets={'version':VERSION,'mode':'category-only','text_preference_enabled':False,
                     'encoding_config_sha256':fingerprint(encoding),'tokenizers':budget.manifest}
        self.manifest={**self.assets,'split':split,
                       'category_names_sha256':fingerprint(catalog['category_names']),
                       'attribute_cache_read':False,'headline_assets_read':False,
                       'history_read_for_text':False}
        self.namespace=fingerprint(self.manifest)

    def summarize(self, user, category):
        value=category_prompt(self.catalog['category_names'][category])
        if not self.budget.fits(value): raise ValueError('Category prompt alone exceeds tokenizer budget; increase budget')
        return {'prompt':value,'support':{},'selected':{},'removed_for_budget':[],
                'token_lengths':self.budget.lengths(value),'category_only':True,
                'cache_key':fingerprint([self.namespace,category,value])}


class HistoryText:
    def __init__(self, config, histories, catalog, split, budget, headlines=None):
        if config.get('version') != VERSION: raise ValueError('Text condition version must be Text-V1.1')
        if split not in ('train','valid','test'): raise ValueError('Explicit original split required')
        self.config,self.histories,self.catalog,self.split,self.budget = config,histories,catalog,split,budget
        path = Path(config['dictionary']); dictionary = json.loads(path.read_text())
        self.extractor = AttributeExtractor(dictionary)
        self.top_k = config.get('top_k_per_group',2)
        if isinstance(self.top_k,int): self.top_k = {g:self.top_k for g in GROUPS}
        if set(self.top_k)!=set(GROUPS) or any(not isinstance(k,int) or k<0 for k in self.top_k.values()):
            raise ValueError('top_k_per_group requires nonnegative integer counts for all groups')
        if headlines is None: self.headlines, source = load_headlines(config,histories,catalog)
        else: self.headlines, source = headlines, {'in_memory_headlines_sha256':fingerprint(headlines)}
        self.assets = {'version':VERSION, 'dictionary_sha256':digest(path),'config_sha256':fingerprint(config),
                       'source':{k:v for k,v in source.items() if k not in ('history_items','available_headlines')},
                       'tokenizers':budget.manifest}
        self.manifest = {**self.assets,'split':split,'history_sha256':fingerprint(histories),'coverage':source}
        self.namespace = fingerprint(self.manifest)
        self.attributes = {}; self.cache = OrderedDict()

    def item_attributes(self, item):
        if item not in self.attributes: self.attributes[item] = self.extractor.extract(self.headlines.get(item,''))
        return self.attributes[item]

    def summarize(self, user, category):
        # This API intentionally has no target item, outfit or context argument.
        counts = self.histories.get(user,{}).get(category,{})
        key = fingerprint([self.namespace,user,category,counts,self.catalog['category_names'][category]])
        if key in self.cache:
            self.cache.move_to_end(key); return self.cache[key]
        support = {g:Counter() for g in GROUPS}
        for item, frequency in counts.items():
            if not isinstance(frequency,int) or frequency<=0: raise ValueError('Positive integer click frequencies required')
            if self.catalog['item_categories'].get(item)!=category: raise ValueError('History category mismatch')
            for group, values in self.item_attributes(item).items():
                for value in values: support[group][value] += frequency
        selected = {g:sorted(support[g],key=lambda a:(-support[g][a],a))[:self.top_k[g]] for g in GROUPS}
        prefix = category_prompt(self.catalog['category_names'][category])
        if not self.budget.fits(prefix): raise ValueError('Category prompt alone exceeds tokenizer budget; increase budget')
        def render():
            fields = [g+': '+', '.join(selected[g]) for g in GROUPS if selected[g]]
            return prefix + (' User preferences: '+'; '.join(fields)+'.' if fields else '')
        removed = []
        while not self.budget.fits(render()):
            candidates = [(support[g][a],-j,a,g) for j,g in enumerate(GROUPS) for a in selected[g]]
            # Lowest support first; ties discard later groups then reverse lexical order.
            minimum = min(x[0] for x in candidates)
            _,_,attribute,group = max((x for x in candidates if x[0]==minimum),key=lambda x:(-x[1],x[2]))
            selected[group].remove(attribute); removed.append({'group':group,'attribute':attribute})
        prompt = render()
        result = {'prompt':prompt,'support':{g:dict(sorted(support[g].items())) for g in GROUPS},
                  'selected':selected,'removed_for_budget':removed,'token_lengths':self.budget.lengths(prompt),
                  'category_only':not any(selected.values()),'cache_key':key}
        self.cache[key]=result
        if len(self.cache)>self.config.get('prompt_cache_size',4096): self.cache.popitem(last=False)
        return result
