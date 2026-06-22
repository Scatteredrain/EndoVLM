#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LLM-based annotation pipeline for endoscopy reports.

Only two tasks are kept:
  1. Anatomical site structuring from raw endoscopy report
  2. Site-level Binary Abnormality Flags extraction

Usage:
  python extract_sites_and_flags.py --input input.jsonl --output output.jsonl

Input JSONL example:
  {"id": "1", "report_type": "gastroscopy", "raw_report": "..."}
  {"id": "2", "report_type": "colonoscopy", "raw_report": "..."}

Required fields:
  - id
  - report_type: "gastroscopy" or "colonoscopy"
  - raw_report: original English endoscopy report text

Output fields added:
  - structured_sites
  - site_abnormality_flags

Requires:
  pip install openai tqdm
"""

import os
import re
import json
import time
import random
import threading
import argparse
import signal
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
from openai import OpenAI


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
MODEL_NAME = "xxx"
MAX_WORKERS = 20
QPS_LIMIT = 4.0
MIN_DELAY = 1.0 / QPS_LIMIT
JITTER = 0.2
RETRY_MAX_ATTEMPTS = 5
RETRY_BASE_DELAY = 2

GASTRIC_SITES = [
    "Esophagus", "Cardia", "Fundus", "Body",
    "Angulus", "Antrum", "Pylorus", "Duodenum",
]

COLON_SITES = [
    "Ileum", "Ileocecal region", "Ascending colon", "Hepatic flexure",
    "Transverse colon", "Splenic flexure", "Descending colon",
    "Sigmoid colon", "Rectum",
]

_lock = threading.Lock()
_last_request_time = 0
_interrupted = False


def _signal_handler(signum, frame):
    global _interrupted
    _interrupted = True


signal.signal(signal.SIGINT, _signal_handler)


# ─────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────
def throttle():
    global _last_request_time
    with _lock:
        elapsed = time.time() - _last_request_time
        sleep_time = max(0, MIN_DELAY + random.uniform(0, JITTER) - elapsed)
        if sleep_time > 0:
            time.sleep(sleep_time)
        _last_request_time = time.time()


def _clean_json_text(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start:end + 1]
    return text.strip()


def call_llm(client, messages, max_tokens=2000, temperature=0.01):
    for attempt in range(RETRY_MAX_ATTEMPTS):
        try:
            throttle()
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=temperature,
                top_p=0.1,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
            raw = completion.choices[0].message.content.strip()
            return json.loads(_clean_json_text(raw))
        except Exception as e:
            if "429" in str(e) or "limit" in str(e):
                backoff = (2 ** attempt) * RETRY_BASE_DELAY + random.uniform(0, 1)
                time.sleep(backoff)
                continue
            if attempt == RETRY_MAX_ATTEMPTS - 1:
                raise
            time.sleep(1)


# ─────────────────────────────────────────────
# Prompt builders
# ─────────────────────────────────────────────
def build_structure_and_flags_prompt(raw_report, report_type):
    if report_type == "gastroscopy":
        site_list = GASTRIC_SITES
        site_desc = (
            "Gastroscopy standard sites (8): "
            "Esophagus, Cardia, Fundus, Body, Angulus, Antrum, Pylorus, Duodenum"
        )
        output_template = {
            "structured_sites": {site: "" for site in GASTRIC_SITES},
            "site_abnormality_flags": [0] * len(GASTRIC_SITES)
        }
    elif report_type == "colonoscopy":
        site_list = COLON_SITES
        site_desc = (
            "Colonoscopy standard sites (9): "
            "Insertion, Ileocecal region, Ascending colon, Hepatic flexure, "
            "Transverse colon, Splenic flexure, Descending colon, Sigmoid colon, Rectum"
        )
        output_template = {
            "structured_sites": {site: "" for site in COLON_SITES},
            "site_abnormality_flags": [0] * len(COLON_SITES)
        }
    else:
        raise ValueError(f"Unsupported report_type: {report_type}")

    system_message = (
        "You are a senior gastroenterologist and medical information extraction expert. "
        "Your task is to process an ORIGINAL endoscopy report and complete TWO tasks at the same time:\n\n"

        "Task 1: Anatomical site structuring\n"
        "Task 2: Site-level binary abnormality flag extraction\n\n"

        "### 1. Site schema\n"
        f"{site_desc}\n\n"

        "### 2. Task 1: Anatomical site structuring rules\n"
        "- Extract the description for each standard anatomical site from the raw report.\n"
        "- Output one field `structured_sites`, which is a dictionary whose keys must exactly match the standard site names.\n"
        "- The value for each site should be the original or near-original report text corresponding to that site.\n"
        "- If a site is not mentioned, use an empty string \"\".\n"
        "- If one sentence describes multiple adjacent sites together, assign the relevant text to each corresponding site when appropriate.\n"
        "- Do not invent content not supported by the report.\n\n"

        "### 3. Task 2: Site-level abnormality flags rules\n"
        "- Output one field `site_abnormality_flags`, which must be a list of binary values in the SAME ORDER as the standard site schema.\n"
        "- 0 = no significant abnormality.\n"
        "- 1 = abnormality present.\n\n"

        "### 4. Definition of normal vs abnormal\n"
        "- Normal / no significant abnormality (flag 0): descriptions mainly containing words like "
        "'smooth', 'soft', 'clear', 'distinct', 'normal', 'regular', 'patent', 'no abnormality', etc.\n"
        "- Color descriptions such as 'pale red', 'reddish', 'mixed red and white appearance', "
        "'predominantly red' are considered normal mucosal appearance and should NOT by themselves be marked abnormal.\n"
        "- For colonoscopy, 'bluish' or 'bluish-gray' at hepatic flexure / splenic flexure can be normal and should NOT by itself be marked abnormal.\n"
        "- Abnormal (flag 1): any congestion, edema, erythema, erosion, ulcer, polyp, nodule, mass, stenosis, diverticulum, exudate, bleeding, inflammation, biopsy taken, lesion, scar, mucosal injury, or any other clearly non-normal finding.\n\n"

        "### 5. Important instructions\n"
        "- Base your judgment only on the raw report text.\n"
        "- Return only pure JSON.\n"
        "- Do not output markdown.\n"
        "- Do not output explanations.\n"
        "- Keys in `structured_sites` must be complete and exactly match the predefined site list.\n"
        "- Length of `site_abnormality_flags` must exactly match the number of standard sites.\n"
    )

    user_message = (
        f"Report type: {report_type}\n\n"
        f"Raw report:\n{raw_report}\n\n"
        f"Output template:\n{json.dumps(output_template, ensure_ascii=False, indent=2)}"
    )

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]


# ─────────────────────────────────────────────
# Validation / normalization
# ─────────────────────────────────────────────
def normalize_result(result, report_type):
    if report_type == "gastroscopy":
        sites = GASTRIC_SITES
    else:
        sites = COLON_SITES

    structured_sites = result.get("structured_sites", {})
    if not isinstance(structured_sites, dict):
        structured_sites = {}

    normalized_structured = {}
    for site in sites:
        value = structured_sites.get(site, "")
        if value is None:
            value = ""
        normalized_structured[site] = str(value).strip()

    flags = result.get("site_abnormality_flags", [])
    if not isinstance(flags, list):
        flags = []

    normalized_flags = []
    for i in range(len(sites)):
        if i < len(flags):
            v = flags[i]
            normalized_flags.append(1 if str(v) == "1" or v is True else 0)
        else:
            normalized_flags.append(0)

    return {
        "structured_sites": normalized_structured,
        "site_abnormality_flags": normalized_flags,
    }


# ─────────────────────────────────────────────
# Processing
# ─────────────────────────────────────────────
def process_item(client, item):
    try:
        report_type = str(item.get("report_type", "")).strip().lower()
        raw_report = item.get("raw_report", "")

        if report_type not in ("gastroscopy", "colonoscopy"):
            raise ValueError("report_type must be 'gastroscopy' or 'colonoscopy'")

        messages = build_structure_and_flags_prompt(raw_report, report_type)
        result = call_llm(client, messages, max_tokens=2000)
        normalized = normalize_result(result, report_type)

        item["structured_sites"] = normalized["structured_sites"]
        item["site_abnormality_flags"] = normalized["site_abnormality_flags"]
        return item

    except Exception as e:
        item["error"] = str(e)
        return item


def get_processed_ids(output_file):
    ids = set()
    if not os.path.exists(output_file):
        return ids

    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                ids.add(str(obj.get("id", "")))
            except json.JSONDecodeError:
                continue
    return ids


def run(input_file, output_file, api_key, base_url):
    client = OpenAI(api_key=api_key, base_url=base_url)

    with open(input_file, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]

    processed = get_processed_ids(output_file)
    to_process = [d for d in data if str(d.get("id", "")) not in processed]

    print(f"Total: {len(data)} | Skipped: {len(processed)} | To process: {len(to_process)}")

    if not to_process:
        return

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_item, client, item): item for item in to_process}

        with open(output_file, "a", encoding="utf-8") as f_out:
            pbar = tqdm(as_completed(futures), total=len(to_process), desc="Processing")
            for future in pbar:
                if _interrupted:
                    print("\nInterrupted by user. Stopping early...")
                    break

                result = future.result()
                if result and "error" not in result:
                    f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                    f_out.flush()
                else:
                    # If you want to also save errors, uncomment below:
                    # f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                    # f_out.flush()
                    pass


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract structured anatomical sites and site-level abnormality flags from raw endoscopy reports"
    )
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--output", required=True, help="Output JSONL file")
    parser.add_argument(
        "--api_key",
        default=os.environ.get("DASHSCOPE_API_KEY", ""),
        help="DashScope API key (or set DASHSCOPE_API_KEY env var)"
    )
    parser.add_argument(
        "--base_url",
        default="...",
        help="DashScope API base URL"
    )
    args = parser.parse_args()

    if not args.api_key:
        raise ValueError("Please provide --api_key or set DASHSCOPE_API_KEY environment variable.")

    run(args.input, args.output, args.api_key, args.base_url)
