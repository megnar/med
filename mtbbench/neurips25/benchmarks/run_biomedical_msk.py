"""Run the extended text-only MSK agent with vLLM; completed cases are resumable."""
import argparse
import json
import os
import queue
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data', default='data/questions_msk_bench.json')
    parser.add_argument('--primekg-dir', default='data/primekg')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--model', default='Qwen/Qwen3-32B-AWQ')
    parser.add_argument('--context', type=int, default=32768)
    parser.add_argument('--max-tokens', type=int, default=1536)
    parser.add_argument('--concurrency', type=int, default=2)
    parser.add_argument('--limit-cases', type=int, default=0)
    args=parser.parse_args()
    if not os.environ.get('NCBI_EMAIL'): parser.error('Set NCBI_EMAIL to a contact address for PubMed')
    if args.concurrency < 1: parser.error('--concurrency must be positive')
    from vllm import LLM, SamplingParams
    from neurips25.models.agent_with_biomedical_tools_msk import DoctorAgentWithBiomedicalToolsMsk
    from neurips25.tools.primekg import PrimeKG
    from neurips25.tools.pubmed_ncbi import PubMed

    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    (out/'settings.json').write_text(json.dumps(vars(args),indent=2))
    cases=json.loads(Path(args.data).read_text())
    if args.limit_cases: cases=dict(list(cases.items())[:args.limit_cases])
    primekg=PrimeKG(args.primekg_dir)
    pubmed=PubMed()
    pubmed_lock=threading.Lock(); log_lock=threading.Lock(); local=threading.local()
    class LoggedPubMed:
        def query(self, search_text):
            with pubmed_lock:
                start=time.time(); record={'case_id':getattr(local,'case_id',None),'query':search_text,'timestamp':datetime.now(timezone.utc).isoformat()}
                try:
                    result=pubmed.query(search_text=search_text);record['result']=result;return result
                except Exception as exc:
                    record['error']=str(exc);raise
                finally:
                    record['seconds']=time.time()-start
                    with log_lock,open(out/'pubmed_calls.jsonl','a') as f: f.write(json.dumps(record,ensure_ascii=False)+'\n')
    llm=LLM(model=args.model,quantization='awq' if args.model.lower().endswith('-awq') else None,dtype='half',max_model_len=args.context,max_num_seqs=args.concurrency,gpu_memory_utilization=0.9,enforce_eager=True)
    sampling=SamplingParams(temperature=0,max_tokens=args.max_tokens)
    jobs=queue.Queue()
    def worker():
        while True:
            first=jobs.get()
            if first is None: return
            batch=[first];deadline=time.monotonic()+2
            while len(batch)<args.concurrency:
                try: batch.append(jobs.get(timeout=max(0,deadline-time.monotonic())))
                except queue.Empty: break
            try:
                results=llm.chat(messages=[job['messages'] for job in batch],sampling_params=sampling,use_tqdm=False)
                if len(results)!=len(batch):raise RuntimeError('Batch result count mismatch')
                for job,result in zip(batch,results): job['result']=result
            except Exception:
                for job in batch:
                    try:job['result']=llm.chat(messages=job['messages'],sampling_params=sampling,use_tqdm=False)[0]
                    except Exception as exc:job['error']=exc
            finally:
                for job in batch:job['done'].set()
    threading.Thread(target=worker,daemon=True).start()
    class Evaluator:
        def __init__(self,case):self.case=case
        def evaluate(self,messages):
            start=time.time();msgs=[{'role':m['role'],'content':m['content']} for m in messages]
            record={'case_id':self.case,'timestamp':datetime.now(timezone.utc).isoformat(),'messages':msgs}
            job={'messages':msgs,'done':threading.Event()};jobs.put(job);job['done'].wait()
            try:
                if 'error' in job:raise job['error']
                result=job['result'];answer=result.outputs[0]
                record.update(response=answer.text,input_tokens=len(result.prompt_token_ids),output_tokens=len(answer.token_ids),finish_reason=answer.finish_reason)
                return answer.text
            except Exception as exc:
                record.update(error=str(exc),traceback=traceback.format_exc());raise
            finally:
                record['seconds']=time.time()-start
                with log_lock,open(out/'calls.jsonl','a') as f:
                    f.write(json.dumps(record,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
    def run(case_id,case):
        if list(out.glob(f'{case_id}_chatlog_*.json')):print('SKIP',case_id,flush=True);return
        local.case_id=case_id;print('START',case_id,flush=True)
        try:
            DoctorAgentWithBiomedicalToolsMsk(Evaluator(case_id),None,args.model,output_dir=str(out),primekg=primekg,pubmed=LoggedPubMed()).run_case(case,case_id=case_id)
            print('COMPLETE',case_id,flush=True)
        except Exception as exc:
            with log_lock,open(out/'errors.jsonl','a') as f:f.write(json.dumps({'case_id':case_id,'error':str(exc),'traceback':traceback.format_exc()})+'\n')
            print('ERROR',case_id,str(exc),flush=True)
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures=[pool.submit(run,k,v) for k,v in cases.items()]
        for future in as_completed(futures):future.result()
    jobs.put(None);print('RUN_FINISHED',flush=True)

if __name__=='__main__':main()
