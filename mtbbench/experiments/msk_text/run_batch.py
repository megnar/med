import os,json,time,traceback,threading,queue,glob
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ThreadPoolExecutor,as_completed
from vllm import LLM,SamplingParams
os.chdir('/content/mtb')
out=Path('/content/mtb_results');out.mkdir(exist_ok=True)
model='Qwen/Qwen3-32B-AWQ'
settings={'model':model,'max_model_len':16384,'max_tokens':1536,'temperature':0.0,'csv_enabled':True,'agent':'MTBBench DoctorAgent (no external tools)','device':'A100 40GB','max_num_seqs':4,'batch_collection_seconds':2}
(out/'settings_batch.json').write_text(json.dumps(settings,indent=2))
llm=LLM(model=model,quantization='awq',dtype='half',max_model_len=16384,max_num_seqs=4,gpu_memory_utilization=0.90,enforce_eager=True)
sampling=SamplingParams(temperature=0,max_tokens=1536)
work=queue.Queue();lock=threading.Lock()
def worker():
 while True:
  first=work.get()
  if first is None:return
  batch=[first];end=time.monotonic()+2
  while len(batch)<4:
   try: batch.append(work.get(timeout=max(0,end-time.monotonic())))
   except queue.Empty: break
  print('BATCH',len(batch),flush=True)
  try:
   results=llm.chat(messages=[x['messages'] for x in batch],sampling_params=sampling,use_tqdm=False)
   if len(results)!=len(batch):raise RuntimeError('Result count does not match batch')
   for item,result in zip(batch,results): item['result']=result
  except Exception as e:
   # A long or invalid prompt must not prevent other cases from progressing.
   print('BATCH_ERROR',str(e),flush=True)
   for item in batch:
    try:item['result']=llm.chat(messages=item['messages'],sampling_params=sampling,use_tqdm=False)[0]
    except Exception as inner:item['error']=inner
  finally:
   for item in batch:item['done'].set()
threading.Thread(target=worker,daemon=True).start()
class Eval:
 def __init__(self,case):self.case=case
 def evaluate(self,messages):
  started=time.time();msgs=[{'role':x['role'],'content':x['content']} for x in messages]
  record={'timestamp':datetime.now(timezone.utc).isoformat(),'case':self.case,'messages':msgs,'mode':'batched'}
  item={'messages':msgs,'done':threading.Event()};work.put(item);item['done'].wait()
  try:
   if 'error' in item:raise item['error']
   res=item['result'];answer=res.outputs[0]
   record.update(response=answer.text,input_tokens=len(res.prompt_token_ids),output_tokens=len(answer.token_ids),finish_reason=answer.finish_reason,seconds=time.time()-started)
   return answer.text
  except Exception as e:
   record.update(error=str(e),traceback=traceback.format_exc(),seconds=time.time()-started);raise
  finally:
   with lock,open(out/'calls.jsonl','a') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
source=Path('neurips25/models/agent.py').read_text().replace('from neurips25.eval import *','').replace("['.txt', '.json']", "['.txt', '.json', '.csv']")
namespace={};exec(compile(source,'agent.py','exec'),namespace);Agent=namespace['DoctorAgent']
cases=json.loads(Path('questions_msk_bench.json').read_text())
def run(case_id,case):
 if list(out.glob(f'{case_id}_chatlog_*.json')): print('SKIP',case_id,flush=True);return
 print('START',case_id,flush=True)
 try:
  Agent(Eval(case_id),None,model,output_dir=str(out)).run_case(case,case_id=case_id)
  print('COMPLETE',case_id,flush=True)
 except Exception as e:
  with lock,open(out/'errors.jsonl','a') as f:f.write(json.dumps({'case':case_id,'error':str(e),'traceback':traceback.format_exc()})+'\n')
  print('ERROR',case_id,str(e),flush=True)
with ThreadPoolExecutor(max_workers=4) as executor:
 futures=[executor.submit(run,k,v) for k,v in cases.items()]
 for future in as_completed(futures):future.result()
work.put(None)
print('BATCH_RUN_COMPLETE',flush=True)
