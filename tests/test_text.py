import copy
import json
from pathlib import Path
import tempfile
import unittest
from tristyle.text import AttributeExtractor, CategoryOnlyText, HistoryText, GROUPS, load_headlines, TokenBudget

DICTIONARY = Path(__file__).resolve().parents[1]/'configs/text_v1_1_dictionary.json'

class Budget:
    manifest={'test':'character budget, not real tokenizer'}
    def __init__(self,n=1000): self.n=n
    def fits(self,t): return len(t)<=self.n
    def lengths(self,t): return [len(t)]

class TextTests(unittest.TestCase):
    def setUp(self):
        self.extractor=AttributeExtractor(json.loads(DICTIONARY.read_text()))
        self.catalog={'category_names':{'1':'shirt','2':'pants'},'item_categories':{'10':'1','11':'1','12':'2','13':'1'}}
        self.config={'version':'Text-V1.1','dictionary':str(DICTIONARY),'top_k_per_group':2}
        self.history={'7':{'1':{'10':3,'11':1},'2':{'12':20}}}
        self.headlines={'10':'黑色黑色 black 宽松 针织','11':'白色 修身','12':'红色高腰','13':'促销编号 A123'}
    def condition(self,h=None,budget=None,split='train'):
        return HistoryText(self.config,self.history if h is None else h,self.catalog,split,budget or Budget(),self.headlines)
    def test_matching_synonyms_boundaries_and_negation(self):
        r=self.extractor.extract('黑色 black blackened palace bracelet 浅蓝色 非真皮 无口袋 無袖')
        self.assertEqual(r[GROUPS[0]],['black','light blue'])
        self.assertNotIn('lace',r[GROUPS[2]]);self.assertNotIn('leather',r[GROUPS[2]])
        self.assertNotIn('pockets',r[GROUPS[3]])
        self.assertEqual(self.extractor.extract('宽松 寬鬆')[GROUPS[1]],['loose fit'])
    def test_training_headline_coordination_and_specific_color(self):
        attrs=self.extractor.extract('碎花纽扣雪纺衫纯色长袖配阔腿裤的雪纺衫')
        self.assertNotIn('wide leg',attrs[GROUPS[1]])
        self.assertEqual(self.extractor.extract('酒红色单鞋')[GROUPS[0]],['burgundy'])

    def test_frequency_and_category(self):
        r=self.condition().summarize('7','1')
        self.assertEqual(r['support'][GROUPS[0]],{'black':3,'white':1})
        self.assertNotIn('red',r['prompt']); self.assertNotIn('黑色',r['prompt'])
        self.assertTrue(r['prompt'].startswith('A photo of shirt, on white background. User preferences:'))
    def test_item_can_have_more_than_two_attributes_but_history_stays_top_two(self):
        condition=self.condition({'7':{'1':{'10':1}}})
        condition.attributes['10']={g:[] for g in GROUPS}
        condition.attributes['10'][GROUPS[0]]=['blue','black','white']
        result=condition.summarize('7','1')
        self.assertEqual(result['support'][GROUPS[0]],{'black':1,'blue':1,'white':1})
        self.assertEqual(result['selected'][GROUPS[0]],['black','blue'])
    def test_empty_unknown_and_no_attributes(self):
        c=self.condition({'7':{'1':{'13':1}}})
        for user in ('7','missing'):
            self.assertEqual(c.summarize(user,'1')['prompt'],'A photo of shirt, on white background.')
    def test_reproducibility_and_cache_isolation(self):
        h={'7':{'1':{'11':1,'10':1}}}; a=self.condition(h).summarize('7','1')
        b=self.condition({'7':{'1':{'10':1,'11':1}}}).summarize('7','1')
        self.assertEqual(a['prompt'],b['prompt']);self.assertEqual(a['cache_key'],b['cache_key'])
        self.assertNotEqual(a['cache_key'],self.condition(h,split='test').summarize('7','1')['cache_key'])
        self.assertNotEqual(a['cache_key'],self.condition().summarize('7','1')['cache_key'])
        self.assertEqual(a['selected'][GROUPS[0]],['black','white'])
    def test_dictionary_and_configuration_invalidate_cache(self):
        first=self.condition().summarize('7','1')['cache_key']
        with tempfile.TemporaryDirectory() as d:
            dictionary=json.loads(DICTIONARY.read_text())
            dictionary['groups'][GROUPS[0]]['black'].append('墨黑色')
            path=Path(d)/'dictionary.json';path.write_text(json.dumps(dictionary))
            c={**self.config,'dictionary':str(path)}
            changed=HistoryText(c,self.history,self.catalog,'train',Budget(),self.headlines)
            self.assertNotEqual(first,changed.summarize('7','1')['cache_key'])
        changed=HistoryText({**self.config,'top_k_per_group':1},self.history,self.catalog,'train',Budget(),self.headlines)
        self.assertNotEqual(first,changed.summarize('7','1')['cache_key'])

    def test_budget_drops_whole_attributes_preserves_prefix(self):
        r=self.condition(budget=Budget(110)).summarize('7','1')
        self.assertLessEqual(len(r['prompt']),110)
        self.assertTrue(r['removed_for_budget']);self.assertIn('black',r['prompt']);self.assertNotIn('white,',r['prompt'])
        with self.assertRaises(ValueError):self.condition(budget=Budget(3)).summarize('7','1')
    def test_target_overlap_is_not_removed(self):
        # Item 10 can also be a held-out target; only its legitimate history count matters.
        self.assertEqual(self.condition().summarize('7','1')['support'][GROUPS[0]]['black'],3)
        with self.assertRaises(TypeError):self.condition().summarize('7','1',target='10')
    def test_loader_only_retains_history_ids(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'headlines.json';path.write_text(json.dumps({**self.headlines,'999':'秘密目标商品'}))
            found,_=load_headlines({'headline_path':str(path),'headline_format':'internal_json'},self.history,self.catalog)
            self.assertEqual(set(found),{'10','11','12'})
    def test_each_tokenizer_budget(self):
        from types import SimpleNamespace
        class Tokenizer:
            model_max_length=77
            special_tokens_map={}
            init_kwargs={}
            def __init__(self,scale):self.scale=scale
            def get_vocab(self):return {'example':self.scale}
            def __call__(self,text,**kwargs):
                assert kwargs['truncation'] is False
                return {'input_ids':list(range(len(text.split())*self.scale+2))}
        holder=SimpleNamespace(tokenizer=Tokenizer(1),tokenizer_2=Tokenizer(3),tokenizer_3=Tokenizer(2),tokenizer_max_length=77)
        budget=TokenBudget(holder,{'max_tokens':40})
        result=self.condition(budget=budget).summarize('7','1')
        self.assertTrue(all(n<=40 for n in result['token_lengths']))
        self.assertTrue(result['removed_for_budget'])

    def test_split_specific_history(self):
        c=self.condition({'7':{'1':{'11':2}}},split='valid')
        self.assertEqual(c.summarize('7','1')['support'][GROUPS[0]],{'white':2})

    def test_category_only_keeps_canonical_prompt_without_reading_history_assets(self):
        c=CategoryOnlyText(self.config,self.catalog,'train',Budget())
        self.assertEqual(c.summarize('7','1')['prompt'],'A photo of shirt, on white background.')
        self.assertFalse(c.manifest['attribute_cache_read'])
        self.assertFalse(c.manifest['history_read_for_text'])

if __name__=='__main__':unittest.main()
