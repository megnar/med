"""Wire existing MTBBench tools agent to an already configured PrimeKG and PubMed.
Caller supplies the original agent class and shared, thread-safe PubMed wrapper.
"""
from pathlib import Path

def make_agent(original_agent,primekg,pubmed):
 class PrimeKGAgent(original_agent):
  def __init__(self,main_llm,oracle_llm,model_name,output_dir='./agent_logs'):
   self.main_llm=main_llm;self.oracle_llm=oracle_llm;self.model_name=model_name
   self.output_dir=Path(output_dir);self.output_dir.mkdir(exist_ok=True,parents=True)
   self.chat_history=[];self.strike_count=0;self.pubmed=pubmed
  def _use_drugbank(self,file_contents):
   if not isinstance(file_contents,str):
    import json
    file_contents=json.dumps(file_contents,ensure_ascii=False)
   return primekg.format(file_contents)
 return PrimeKGAgent
