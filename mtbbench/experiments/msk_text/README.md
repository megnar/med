# MSK text experiments

These are the scripts and records from the 26 November 2026 experiments; they retain their original Colab paths as historical artifacts. They are not a turnkey installation. They require the MSK question/case data, a compatible vLLM runtime, Qwen3-32B-AWQ, and (for tools runs) an authorized PrimeKG data directory and NCBI contact email. Do not commit credentials.

- `run.py`, `run_all.py`, `run_batch.py`: no-tools pilot and batched full run, CSV file access enabled. Results: `agent_logs_msk/qwen3-32b-awq-no-tools`, 113/183 correct.
- `run_tools_pilot.py`, `run_tools_all.py`, `run_tools_retry.py`, `agent_factory.py`: PubMed + PrimeKG experiment. Results: `agent_logs_msk/qwen3-32b-awq-pubmed-primekg`, 123/183 correct. Three cases were retried with a 32768-token context after exceeding 16384; the error log and the per-call traces retain failed attempts.
- `audit_summary.json`: exact-name coverage audit of PrimeKG features in case text; this is not recall or a clinical validation.
- `agent_logs_msk/gemini-hydrogen-independent`: independent-question Gemini UI experiment, 131/183; UI reported Hydrogen, exact backend not independently verified. Each question used a fresh conversation with all current-stage files, and the experiment is not equivalent to the sequential agent runs.

The experiments have different prompts, tools, context handling and protocols; percentage comparisons are descriptive only. Some original case files contain fields such as STOP_AGE after a question's prediction date; therefore the results may contain temporal leakage. See the source data documentation and inspect inputs before drawing conclusions.
