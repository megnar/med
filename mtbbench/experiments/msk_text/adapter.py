"""PrimeKG text adapter for MTBBench. Exact case-insensitive word matching."""
import csv,re
from pathlib import Path
class PrimeKG:
 def __init__(self, path):
  path=Path(path)
  with (path/'drug_features.tab').open(encoding='utf8',newline='') as f:self.features={r['node_index']:r for r in csv.DictReader(f,delimiter='\t')}
  with (path/'nodes.tab').open(encoding='utf8',newline='') as f:
   self.drugs={r['node_name'].strip().casefold():r for r in csv.DictReader(f,delimiter='\t') if r['node_type']=='drug' and r['node_index'] in self.features and len(r['node_name'].strip())>=4}
  self.pattern=re.compile(r'(?<!\w)(?:'+'|'.join(re.escape(x) for x in sorted(self.drugs,key=len,reverse=True))+r')(?!\w)',re.I)
 def lookup(self,text):
  found=sorted({m.group().casefold() for m in self.pattern.finditer(text)})
  result=[]
  for name in found:
   node=self.drugs[name];attrs=self.features[node['node_index']]
   fields={k:attrs[k].strip() for k in ('description','indication','mechanism_of_action') if attrs.get(k,'').strip()}
   if fields:result.append({'name':node['node_name'],'node_id':node['node_id'],'fields':fields})
  return result
 def format(self,text):
  items=self.lookup(text)
  return '\n'.join(f"{x['name']}: "+' '.join(f'{k}: {v}' for k,v in x['fields'].items()) for x in items)
