import os, json, time, sys, traceback
from pathlib import Path
from datetime import datetime, timezone
from vllm import LLM, SamplingParams

os.chdir('/content/mtb')
out=Path('/content/mtb_results'); out.mkdir(exist_ok=True)
trace=out/'calls.jsonl'
model='Qwen/Qwen3-32B-AWQ'
settings={'model':model,'max_model_len':16384,'max_tokens':1536,'temperature':0.0,'csv_enabled':True,'agent':'MTBBench DoctorAgent (no external tools)','device':'A100 40GB','pilot_cases':1}
(out/'settings.json').write_text(json.dumps(settings,indent=2))
llm=LLM(model=model,quantization='awq',dtype='half',max_model_len=16384,max_num_seqs=1,gpu_memory_utilization=0.90,enforce_eager=True)
sampling=SamplingParams(temperature=0,max_tokens=1536)
class Eval:
 def evaluate(self,messages):
  started=time.time(); msgs=[{'role':x['role'],'content':x['content']} for x in messages]
  record={'timestamp':datetime.now(timezone.utc).isoformat(),'case':current_case,'messages':msgs}
  try:
   res=llm.chat(messages=msgs,sampling_params=sampling,use_tqdm=False)[0]
   output=res.outputs[0]
   record.update(response=output.text,input_tokens=len(res.prompt_token_ids),output_tokens=len(output.token_ids),finish_reason=output.finish_reason,seconds=time.time()-started)
   return output.text
  except Exception as e:
   record.update(error=str(e),traceback=traceback.format_exc(),seconds=time.time()-started)
   raise
  finally:
   with open(trace,'a') as f: f.write(json.dumps(record,ensure_ascii=False)+'\n'); f.flush(); os.fsync(f.fileno())
source=Path('neurips25/models/agent.py').read_text().replace('from neurips25.eval import *','').replace("['.txt', '.json']", "['.txt', '.json', '.csv']")
namespace={};exec(compile(source,'agent.py','exec'),namespace)
Agent=namespace['DoctorAgent']
cases=json.loads(Path('questions_msk_bench.json').read_text())
for current_case,case in list(cases.items())[:1]:
 print('START',current_case,flush=True)
 agent=Agent(Eval(),None,model,output_dir=str(out))
 agent.run_case(case,case_id=current_case)
 print('COMPLETE',current_case,flush=True)
print('PILOT_COMPLETE',flush=True)
