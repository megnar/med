import json
import os
import re
import time
import torch
import pandas as pd
from loguru import logger
from omegaconf import OmegaConf
from pathlib import Path
from neurips25.tools.biomedical import BiomedicalTools
from neurips25.tools.primekg import PrimeKG


import yaml

class DoctorAgentWithBiomedicalToolsMsk:
    def __init__(self, main_llm, oracle_llm, model_name, output_dir="./agent_logs", primekg_dir=None, pubmed=None, biomedical=None, primekg=None):
        self.main_llm, self.oracle_llm, self.model_name = main_llm, oracle_llm, model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.chat_history, self.strike_count = [], 0
        if pubmed is None:
            from neurips25.tools.pubmed_ncbi import PubMed
            pubmed = PubMed()
        self.pubmed = pubmed
        self.primekg = primekg or PrimeKG(primekg_dir or os.environ.get("PRIMEKG_DIR", "data/primekg"))
        self.biomedical = biomedical or BiomedicalTools()

    def _use_pubmed(self, search_text):
        """
        Use PubMed to search for articles related to the given search term.
        """
        pubmed_output = self.pubmed.query(search_text=search_text)
        return pubmed_output + "\nNote that you can use the information from PubMed to answer the questions, but it is possible that the information is incorrect or irrelevant. You MUST rely also on your internal knowledge."


    def _use_drugbank(self, file_contents):
        # Kept as a compatibility hook for the upstream file attachment loop.
        if not isinstance(file_contents, str):
            file_contents = json.dumps(file_contents, ensure_ascii=False)
        return self.primekg.format(file_contents)

    def _parse_files(self, file_paths):
        """
        Parse the file paths to separate images and text files.
        """
        images = []
        texts = []

        for path in file_paths:
            ext = Path(path).suffix.lower()
            if ext in ['.jpg', '.jpeg', '.png']:
                images.append(path)
            elif ext in ['.txt', '.json', '.csv']:
                texts.append(path)

        file_overview = []
        for img in images:
            file_overview.append(f"[FILE: {os.path.basename(img)}]")
        for txt_path in texts:
            file_overview.append(f"[FILE: {os.path.basename(txt_path)}]")

        return file_overview


    def _attach_files(self, requested_files, current_file_basenames, current_file_paths):
        """
        Attach files to the conversation based on the requested files.
        """
        file_msg = ""
        attached_files = []
        hallucinated_files = []
        all_given_files = []

        for req_file in set(requested_files):
            req_file = req_file.replace(" ", "") # some models add spaces before extension
            # Check if the requested file is in the current file basenames
            if req_file not in current_file_basenames:
                file_msg += f"[FILE: {req_file}] not found. Only ask for files that were listed to you earlier! Example request format for 2 images: [REQUEST: image1.jpg] [REQUEST: image2.jpg]\n"
                hallucinated_files.append(req_file)
                continue

            # Add to all given files
            all_given_files.append(req_file)

            # Check if the requested file is in the current file paths
            full_path = next((fp for fp in current_file_paths if os.path.basename(fp) == req_file), None)
            if full_path:
                try:
                    with open(full_path, 'r') as f:
                        content = json.load(f) if full_path.endswith('.json') else f.read()
                        file_msg += f"[FILE: {req_file}] included in your context\n{content}\n"

                        # Use DrugBank tool
                        drugbank_info = self._use_drugbank(content)
                        if drugbank_info:
                            file_msg += f"[PrimeKG: {drugbank_info}]\n"
                except Exception as e:
                    file_msg += f"[ERROR: Failed to read {req_file}: {e}]\n"

        return file_msg, attached_files, hallucinated_files, all_given_files


    def _dettach_files(self, conversation):
        """
        Detach files from the conversation.
        """
        for entry in conversation:
            if 'files' in entry:
                del entry['files']
            if entry['role'] == 'user':
                content = entry['content']
                file_names = re.findall(r"\[FILE: (.+?)\] included\n", content)
                content = ""
                for file_name in file_names:
                    content += f"[FILE: {file_name}] was accessed by you\n"
                if file_names:
                    entry['content'] = content + "You can access all these files once again if necessary by asking for them in the format [REQUEST: filename.extension].\n"
                    logger.info(f"Files detached from conversation: {file_names}")
                    file_names = []
        return conversation

    @torch.no_grad()
    def run_case(self, case_data, case_id="patient_case"):
        """
        Run the agent on a single patient case.
        """
        logger.info(f"Running agent on {case_id}")
        self.case_id = case_id

        conversation = []
        self.chat_history = []  # Clear previous runs
        self.strike_count = 0
        # self.question_count = 0

        # System prompt
        system_message = (
            "You are a pathologist AI assistant analyzing patient information and answering multiple-choice questions.\n"
            "Request each available file using [REQUEST: filename.extension] and wait for it before reasoning about its contents.\n"
            "PrimeKG descriptions may be appended automatically when an attached file mentions a drug.\n"
            "To search publications use [PUBMED: search terms]. To search CIViC molecular profiles use "
            "[CIVIC: gene and variant], for example [CIVIC: KRAS G12V]. CIViC name results may be approximate: verify gene, variant and disease.\n"
            "For human pathways use [REACTOME: UniProt accession], for example [REACTOME: P04637]. "
            "For human causal signaling interactions use [SIGNOR: UniProt accession], for example [SIGNOR: P04637].\n"
            "Only use an accession when you know its identity; do not guess. Tool results may be incomplete, "
            "may contain information unavailable at the question date, and do not establish clinical causality.\n"
            "Request one tool at a time and wait for its response. Do not include private patient identifiers in queries.\n"
            "When ready, answer in the format [ANSWER: A) answer] or [ANSWER: B) answer]."
        )
        conversation.append({"role": "system", "content": system_message})

        # Initialize context and file tracking
        context_message = ""
        current_file_paths = []
        current_file_basenames = []

        context_message = ""
        for entry in case_data:
            # Patient context
            if 'context' in entry:
                # Update current context
                context_message = f"You are given the following new patient information: \n{entry['context']}\n"

            # Patient files
            elif 'file_paths' in entry:
                # Update available files
                current_file_paths = entry['file_paths']
                current_file_basenames = [os.path.basename(f) for f in current_file_paths]
                file_overview = self._parse_files(current_file_paths)

                # Announce new available files
                context_message +=( f"New files available:\n {'\n'.join(file_overview)}" +
                                    "Request files using [REQUEST: filename.extension]; use [CIVIC: gene variant], [REACTOME: UniProt accession], [SIGNOR: UniProt accession], or [PUBMED: search terms] when useful. Wait for the result.\n"
                                    "You must provide a separate [REQUEST] tag for each file you need. Example [REQUEST: image1.jpg] [REQUEST: image2.jpg].\n"
                                    f"You also have access to PUBMED that you can use to query articles that may potentially be relevant to the patient history. If you ask for information, you must wait for the tool to provide it before continuing.\n"
                                    f"Example: [PUBMED: search words]\n"
                                )

            # Patient question
            elif 'question' in entry:
                # Now we can ask a question
                question = entry['question']
                expected_answer = entry['answer']
                # self.question_count += 1

                # Detach files from the conversation
                conversation = self._dettach_files(conversation)

                logger.info(f"Processing question: {question.strip()}")
                conversation.append(
                    {
                        "role": "user",
                        "content": (
                                    f"{context_message}\n Question: {question}\n"
                                    "Request files using [REQUEST: filename.extension]; use [CIVIC: gene variant], [REACTOME: UniProt accession], [SIGNOR: UniProt accession], or [PUBMED: search terms] when useful. Wait for the result.\n"
                                    "You must provide a separate [REQUEST] tag for each file you need. Example [REQUEST: image1.jpg] [REQUEST: image2.jpg].\n"
                                    f"You also have access to PUBMED that you can use to query articles that may potentially be relevant to the patient history.\n"
                                    f"Example: [PUBMED: search words]\n"
                                )
                    }
                )
                # Empty the context message since we have included it in the conversation
                context_message = ""

                # Start the conversation loop until a valid answer is provided
                question_start_time = time.time()
                prev_requested_files = []
                all_files_accessed_for_question = []
                all_halucinated_files = []
                tool_turns = 0
                while True:
                    response = self.main_llm.evaluate(messages=conversation)
                    logger.debug(f"Model response: {response}")
                    conversation.append({"role": "assistant", "content": response})

                    # If the model requests files and they are different from the previous request
                    requested_files = re.findall(r"\[REQUEST:\s*([^\]]+)\]", response)
                    if requested_files and requested_files != prev_requested_files:
                        prev_requested_files = requested_files
                        # Check if the files are in list format
                        if len(requested_files) == 1 and requested_files[0].startswith("[REQUEST:") and "," in requested_files[0]:
                            requested_files = requested_files[0].split(",")
                        # Add files to the conversation
                        file_msg, attached_files, hallucinated_files, all_given_files = self._attach_files(requested_files, current_file_basenames, current_file_paths)
                        logger.info(f"Files: {file_msg}")
                        all_files_accessed_for_question.extend(all_given_files)
                        all_halucinated_files.extend(hallucinated_files)

                        conversation.append({"role": "user", "content": file_msg, "files": attached_files})

                    # Biomedical tools return bounded, attributed results or explicit errors.
                    elif re.findall(r"\[(CIVIC|REACTOME|SIGNOR):\s*([^\]]+)\]", response, flags=re.IGNORECASE):
                        tool_turns += 1
                        if tool_turns > 12:
                            conversation.append({"role": "user", "content": "Tool call limit reached. Provide your best [ANSWER: LETTER) answer]."})
                            self.strike_count += 1
                            if self.strike_count > 2:
                                self.chat_history.append({"question": question, "answer": expected_answer, "response": response, "correct": False, "files_accessed": all_files_accessed_for_question, "files_hallucinated": all_halucinated_files, "question_time": time.time() - question_start_time})
                                break
                            continue
                        tool, argument = re.findall(r"\[(CIVIC|REACTOME|SIGNOR):\s*([^\]]+)\]", response, flags=re.IGNORECASE)[-1]
                        result = self.biomedical.query(tool, argument)
                        trace = {"case_id": case_id, "tool": tool.upper(), "argument": argument, "result": result, "timestamp": time.time()}
                        with open(self.output_dir / f"{case_id}_tool_calls.jsonl", "a") as trace_file:
                            trace_file.write(json.dumps(trace, ensure_ascii=False) + "\n")
                        conversation.append({"role": "user", "content": f"[{tool.upper()} RESULT] " + json.dumps(result, ensure_ascii=False)})

                    # If the model uses Pubmed
                    elif re.findall(r"\[PUBMED:\s*([^\]]+)\]", response):

                        pubmed_search = re.findall(r"\[PUBMED:\s*([^\]]+)\]", response)
                        if pubmed_search:
                            pubmed_search = pubmed_search[-1]
                            logger.info(f"Using PubMed for search: {pubmed_search}")
                            pubmed_response = self._use_pubmed(pubmed_search)
                            logger.info(f"PubMed response: {pubmed_response}")
                            conversation.append({"role": "user", "content": pubmed_response})
                        else:
                            logger.info("Model did not provide a valid PUBMED request.")
                            conversation.append(
                                {
                                    "role": "user",
                                    "content": "To use PUBMED, you should provide the search term in the format [PUBMED: search words]. Make sure you have both the opening an closing brackets."
                                }
                            )

                    # If the model provides an answer
                    elif re.findall(r"\[ANSWER:([^\]]+)\]", response):
                        if self.strike_count > 2:
                            logger.info("Model failed to provide a valid answer or request after 3 attempts.")
                            self.strike_count = 0
                            question_end_time = time.time()
                            correct = False
                            self.chat_history.append({
                                "question": question,
                                "answer": expected_answer,
                                "response": response,
                                "correct": correct,
                                "files_accessed": all_files_accessed_for_question,
                                "files_hallucinated": all_halucinated_files,
                                "question_time": question_end_time - question_start_time,
                            })
                            break


                        question_end_time = time.time()
                        # Take first answer only
                        response = re.findall(r"\[ANSWER:\s*([^\]]+)\]", response)[-1]
                        if (len(response.strip()) >=2 and response.strip()[0].lower() in 'abcdef' and response.strip()[1] in [")", "]", " "]) or (len(response.strip()) == 1 and response.strip()[0].lower() in 'abcdef'):
                            self.strike_count = 0
                            correct = expected_answer.strip()[0].lower() == response.strip()[0].lower()
                            self.chat_history.append({
                                "question": question,
                                "answer": expected_answer,
                                "response": response,
                                "correct": correct,
                                "files_accessed": all_files_accessed_for_question,
                                "files_hallucinated": all_halucinated_files,
                                "question_time": question_end_time - question_start_time,
                            })
                            break
                        else:
                            self.strike_count += 1
                            logger.info("Model did not provide a valid answer or request.")
                            conversation.append(
                                {
                                    "role": "user",
                                    "content": "Please provide the final answer in [ANSWER: LETTER) answer] specifying the answer choice letter you picked (A, B, C, D, E, or F) or ask for files with [REQUEST: filename.extension]. Make sure you have both the opening an closing brackets."
                                }
                            )


                    # If the model does neither
                    else:
                        self.strike_count += 1

                        # If the model has failed to follow our template, break the loop
                        if self.strike_count <= 2:
                            logger.info("Model did not provide a valid answer or request.")
                            conversation.append(
                                {
                                    "role": "user",
                                    "content": "Please provide the final answer in [ANSWER: LETTER) answer] specifying the answer choice letter you picked (A, B, C, D, E, or F) or ask for files with [REQUEST: filename.extension]. Make sure you have both the opening an closing brackets."
                                }
                            )
                        else:
                            logger.info("Model failed to provide a valid answer or request after 3 attempts.")
                            self.strike_count = 0
                            question_end_time = time.time()
                            correct = False
                            self.chat_history.append({
                                "question": question,
                                "answer": expected_answer,
                                "response": response,
                                "correct": correct,
                                "files_accessed": all_files_accessed_for_question,
                                "files_hallucinated": all_halucinated_files,
                                "question_time": question_end_time - question_start_time,
                            })
                            break

        # Log the entire conversation
        self.chat_history.append({
            "conversation": conversation
        })
        self._store_log(case_id)
        return self.chat_history


    def _store_log(self, case_id):
        """
        Store the chat history in a JSON file.
        """
        log_path = self.output_dir / f"{case_id}_chatlog_{int(time.time())}.json"
        with open(log_path, 'w') as f:
            json.dump(self.chat_history, f, indent=2)
        logger.info(f"Chat history saved to {log_path}")
