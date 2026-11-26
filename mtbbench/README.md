# MTBBench: A Multimodal Sequential Clinical Decision-Making Benchmark in Oncology

**MTBBench** is a benchmark designed to evaluate the reasoning capabilities of multimodal large language models (LLMs) in complex clinical decision-making scenarios. It focuses on two core challenges in oncology: multimodal integration (e.g., pathology, genomics, radiology) and longitudinal reasoning across patient timelines. The benchmark includes agentic tasks requiring interaction with external foundation model-based tools and datasets.

---

## 🛠 Getting Started

To install all required dependencies, simply run:

```bash
bash setup.sh
```

> **Note**: If you want to evaluate the agent on **IHC data**, you will need to additionally clone and install [TRIDENT](https://github.com/mahmoodlab/TRIDENT/tree/main) from source.

---

## ⚙️ Configuration

Before running the benchmark, configure your paths and credentials in:

```
neurips25/configs/base.yaml
```

The config file specifies paths to datasets, tool credentials, and output directories. Below, we provide guidance for acquiring the necessary external datasets.

---

## 📁 Datasets

### HANCOCK (Multimodal Tissue Microarrays)

The **HANCOCK** dataset contains SVS-format tissue micro arrays (TMAs). To prepare it:

1. Follow the original [HANCOCK GitHub repository](https://github.com/ankilab/HANCOCK_MultimodalDataset) to extract tiles and compute cell densities using QuPath.
2. Reproduce ABMIL training by extracting tumor centers and cell density measurements for **Blocks 1 and 2**. The pretrained ABMIL checkpoint is not included in this text-focused copy.
3. Download the dataset files from the [HANCOCK project page](https://hancock.research.fau.eu/download) to replicate the question curation used in the benchmark.

---

### MSK-CHORD (Longitudinal Genomic Profiles)

The **MSK-CHORD** dataset is available on [cBioPortal](https://www.cbioportal.org/study/summary?id=msk_chord_2024). To use it:

* Download the ZIP archive from the cBioPortal page.
* Extract it and update the dataset path in `base.yaml`.

---

### DrugBank API

To enable the **DrugBank tool** for longitudinal drug lookups:

1. [Register for a DrugBank account](https://www.drugbank.com).
2. Apply for a license to access their API.
3. Download and locally host the dataset following their documentation.
4. Update your API path and credentials in the config file.

---

## 📑 Agent Logs

`agent_logs_msk/` contains the five highest-accuracy original MSK runs by scored answers, plus the local Qwen and Gemini experiments. HANCOCK logs are not included in this text-focused copy; the complete original logs are available in the upstream MTBBench repository. Case logs contain agent–model conversations and scored answers; local experiments may use different protocols.

---

## ▶️ Running the Benchmark

Make sure you have:

* Installed dependencies
* Configured Hugging Face access tokens (for model download)
* Set paths in `base.yaml`

To run an evaluation with `Qwen/Qwen2.5-VL-7B-Instruct` on the HANCOCK dataset:

```bash
python -m neurips25.benchmarks.run_agent_benchmark \
  --doctor_model "Qwen/Qwen2.5-VL-7B-Instruct" \
  --output_dir "./agent_logs_hancock/" \
  --dataset "hancock"
```

---

## Extended MSK text agent (experimental)

`neurips25.models.agent_with_biomedical_tools_msk` adds PrimeKG drug descriptions and read-only CIViC, Reactome and SIGNOR lookups to the MSK agent. It is a new protocol and has **not** been benchmarked yet. Tool instructions are in its system prompt; the agent uses `[CIVIC: KRAS G12V]`, `[REACTOME: P04637]`, `[SIGNOR: P04637]`, `[PUBMED: search terms]`, `[REQUEST: filename]` and `[ANSWER: A) answer]` tags. Reactome and SIGNOR need a known UniProt accession; CIViC does a name search and results require variant/disease verification. Sources can contain knowledge newer than the clinical question; no automatic temporal cutoff is applied.

For an A100/vLLM run, provide MSK question data and all referenced files under their expected paths, and PrimeKG `nodes.tab` and `drug_features.tab` in a directory outside version control. PrimeKG data may include third-party material, including DrugBank; verify the applicable terms. Install vLLM, PyTorch, transformers, biopython, pandas, requests, pyyaml, omegaconf and loguru in a suitable environment. Set `NCBI_EMAIL` to a contact address, then run a small pilot first:

```bash
export NCBI_EMAIL='your-contact@example.org'
python -m neurips25.benchmarks.run_biomedical_msk \
  --data data/questions_msk_bench.json --primekg-dir data/primekg \
  --output-dir agent_logs_msk/qwen3-extended-pilot --limit-cases 1 \
  --context 32768 --concurrency 1
```

Omit `--limit-cases` to run all cases after inspecting the pilot. `--concurrency` batches independent cases; the maximum usable context and concurrency depend on GPU memory. Completed per-case files are skipped when resuming, and `errors.jsonl` records failures. The output includes per-case conversations, model call input/output tokens and responses in `calls.jsonl`, PubMed query traces in `pubmed_calls.jsonl`, and CIViC/Reactome/SIGNOR calls in per-case tool logs. Existing case data may contain future information (e.g. `STOP_AGE`), so outputs cannot be treated as leakage-free forecasts without a separate audit.

The earlier Qwen and Gemini experiments, their differing protocols, and their logs are documented in `experiments/msk_text/README.md`.
