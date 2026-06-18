"""
LLM-based annotation pipeline for endoscopy reports.

Assumes input reports are already in English (structured format).

Four tasks:
  1. QC (report type classification: gastroscopy/colonoscopy, standard/non-standard)
  2. Site-level Binary Abnormality Flags
  3. Abnormal Findings Extraction
  4. Disease Entity Extraction (endoscopy + pathology)

Usage:
  python extract_anatomy_pathology.py --task all --input input.jsonl --output output.jsonl

Requires: openai, tqdm
  pip install openai tqdm
"""

import os
import re
import json
import time
import random
import threading
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
from openai import OpenAI

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
MODEL_NAME = "qwen3-max-2025-09-23"   # or "qwen-max"
MAX_WORKERS = 30
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
    "Insertion", "Ileocecal region", "Ascending colon", "Hepatic flexure",
    "Transverse colon", "Splenic flexure", "Descending colon",
    "Sigmoid colon", "Rectum",
]

_lock = threading.Lock()
_last_request_time = 0
_interrupted = False

import signal
def _signal_handler(signum, frame):
    global _interrupted
    _interrupted = True
signal.signal(signal.SIGINT, _signal_handler)


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


# ═══════════════════════════════════════════════════════════════
# Task 1: QC — Report Type Classification
# ═══════════════════════════════════════════════════════════════
def build_qc_prompt(exam_item, endo_report):
    system_message = (
        "You are a senior gastroenterologist and medical data scientist. "
        "Your task is to classify the input endoscopy report based on examination type and whether it is a standard diagnostic procedure. "
        "You must use the built-in anatomical knowledge base for logical reasoning and output results in strict JSON format.\n\n"

        "### 1. Knowledge Base\n"
        "**A) Gastroscopy standard sites (8):**\n"
        "Esophagus, Cardia, Fundus, Body, Angulus, Antrum, Pylorus, Duodenum\n\n"
        "**B) Colonoscopy standard sites (9):**\n"
        "Insertion, Ileocecal region, Ascending colon, Hepatic flexure, Transverse colon, "
        "Splenic flexure, Descending colon, Sigmoid colon, Rectum\n\n"

        "### 2. Classification Tasks\n"
        "**Flag 1: is_colonoscopy (examination type)**\n"
        "Note: The examination item may contain both gastroscopy and colonoscopy keywords; in that case, check the endoscopy findings.\n"
        "- Set to 1: The endoscopy findings contain colonoscopy keywords and are entirely lower GI sites.\n"
        "- Set to 0: The endoscopy findings contain gastroscopy keywords and are entirely upper GI sites.\n\n"

        "**Flag 2: is_non_standard (non-standard examination)**\n"
        "Consider both the examination item and endoscopy findings.\n"
        "- Set to 1 (True): If the report includes therapeutic or special procedures such as EUS, ESD, EMR, POEM, STER, stent placement, tube drainage, foreign body removal, emergency hemostasis, etc.\n"
        "- Set to 0 (False): If it is a standard diagnostic gastroscopy/colonoscopy (including simple biopsy, polypectomy with cold snare/hot snare, band ligation, clip closure, etc.).\n\n"
        "Note: Only classify as non-standard when the procedure carries high risk, requires special preparation, or is not routinely performed in outpatient settings. "
        "For example: a 6mm polyp found and removed with snare + clip closure is still a standard colonoscopy.\n\n"

        "### 3. Output Requirements\n"
        "- Return only a pure JSON object\n"
        "- No markdown formatting (e.g., ```json)\n"
        "- No explanatory text\n"
        "- Must be directly parseable by json.loads()"
    )

    user_message = (
        f"【Input Data】\n"
        f"Examination Item: {exam_item}\n"
        f"Endoscopy Findings: {endo_report}\n\n"

        "【Output Template】\n"
        "{\n"
        '  "qc_flags": {\n'
        '    "is_colonoscopy": 0,\n'
        '    "is_non_standard": 0\n'
        '  }\n'
        "}"
    )

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]


# ═══════════════════════════════════════════════════════════════
# Task 2: Site-level Binary Abnormality Flags
# ═══════════════════════════════════════════════════════════════
def build_abnormality_flag_prompt(endo_report, is_gastric):
    if is_gastric:
        system_message = (
            "You are a senior gastroenterologist. Your task is to analyze the endoscopy report "
            "and determine whether each standard anatomical site has any abnormality.\n\n"

            "### 1. Criteria\n"
            "- **0 (No significant abnormality)**: The description mainly contains: 'smooth', 'soft', "
            "'clear/distinct', 'normal', 'no abnormalities', 'patent', 'regular', etc.\n"
            "  - Descriptions such as 'pale red', 'reddish in color', 'mixed red and white areas', "
            "'mixed red and white appearance', 'predominantly red' indicate normal mucosal color and should be flagged as 0.\n"
            "- **1 (Abnormality present)**: Includes any congestion, edema, inflammation, injury, lesion, "
            "polyp, ulcer, biopsy taken, or other non-normal findings.\n\n"

            "### 2. Anatomical Site Order\n"
            "**Gastroscopy (length: 8)**: [Esophagus, Cardia, Fundus, Body, Angulus, Antrum, Pylorus, Duodenum]\n\n"

            "### 3. Output Requirements\n"
            "- Return only pure JSON containing `site_abnormality_flags`."
        )
        template = "[0, 0, 0, 0, 0, 0, 0, 0]"
    else:
        system_message = (
            "You are a senior gastroenterologist. Your task is to analyze the endoscopy report "
            "and determine whether each standard anatomical site has any abnormality.\n\n"

            "### 1. Criteria\n"
            "- **0 (No significant abnormality)**: The description mainly contains: 'smooth', 'soft', "
            "'clear/distinct', 'normal', 'no abnormalities', 'patent', 'regular', etc.\n"
            "  - Descriptions such as 'pale red', 'reddish in color', 'mixed red and white areas', "
            "'mixed red and white appearance', 'predominantly red' indicate normal mucosal color and should be flagged as 0.\n"
            "  - Descriptions such as 'bluish', 'bluish-gray' at the Hepatic/Splenic flexure indicate normal mucosal appearance and should be flagged as 0.\n"
            "- **1 (Abnormality present)**: Includes any congestion, edema, inflammation, injury, lesion, "
            "polyp, ulcer, diverticulum, biopsy taken, or other non-normal findings.\n\n"

            "### 2. Anatomical Site Order\n"
            "**Colonoscopy (length: 9)**: [Insertion, Ileocecal region, Ascending colon, Hepatic flexure, "
            "Transverse colon, Splenic flexure, Descending colon, Sigmoid colon, Rectum]\n\n"

            "### 3. Output Requirements\n"
            "- Return only pure JSON containing `site_abnormality_flags`."
        )
        template = "[0, 0, 0, 0, 0, 0, 0, 0, 0]"

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": f"【Endoscopy Report】\n{endo_report}\n\n【Output Template】\n{{\"site_abnormality_flags\": {template}}}"},
    ]


# ═══════════════════════════════════════════════════════════════
# Task 3: Abnormal Findings Extraction
# ═══════════════════════════════════════════════════════════════
def build_findings_prompt(endo_report):
    system_message = (
        "You are a senior endoscopy data analyst. Your task is to analyze the endoscopy report, "
        "extract abnormal findings, and standardize them into concise pathology terms.\n\n"

        "### 1. Core Extraction Principles\n"
        "- **No location or modifier words**: Output only the pathology name (e.g., 'polyp', 'erosion'). "
        "Do NOT include location words (e.g., 'antral', 'gastric body') or descriptive modifiers (e.g., 'multiple', 'scattered', 'rice-grain sized').\n"
        "- **Normal state (do NOT extract)**:\n"
        "  - Descriptions containing 'smooth', 'soft', 'clear', 'normal', 'patent', 'regular', etc.\n"
        "  - Mucosal color descriptions (e.g., 'reddish in color', 'pale red', 'mixed red and white areas', "
        "'predominantly red') represent normal states and must NOT be extracted as 'congestion'.\n"
        "- **Must extract (abnormal types)**:\n"
        "  - Mucosal lesions: erosion, ulcer, polyp, nodule, atrophy, stenosis, mass, exudate.\n"
        "  - Structural abnormalities: hiatal hernia, cardia laxity, diverticulum.\n"
        "  - Metabolic/reflux signs: bile reflux, bile staining.\n"
        "  - Inflammatory signs: congestion, edema, erythema, bleeding.\n\n"

        "### 2. Standardization Rules\n"
        "- Map original descriptions to standard terms. For example:\n"
        "  - 'punctate erosion in the antrum' -> 'erosion'\n"
        "  - '0.3cm polyp on the posterior wall of the body' -> 'polyp'\n"
        "  - 'cardia displaced upward, hernia sac seen on retroflexion' -> 'hiatal hernia'\n"
        "  - 'bile staining on antral mucosa' -> 'bile staining'\n"
        "  - 'antral mucosal congestion and edema' -> 'congestion', 'edema'\n\n"

        "### 3. Output Requirements\n"
        "- Return only pure JSON in the format: `{\"abnormal_findings\": [\"term1\", \"term2\"]}`.\n"
        "- Deduplicate similar findings; no need to split by anatomical site.\n"
        "- If no clear abnormality, return an empty list [].\n"
        "- Do not output any explanation or extra fields."
    )

    user_message = (
        f"【Endoscopy Report】\n{endo_report}\n\n"
        "【Output Template】\n"
        "{\n"
        '  "abnormal_findings": []\n'
        "}"
    )

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]


# ═══════════════════════════════════════════════════════════════
# Task 4: Disease Entity Extraction
# ═══════════════════════════════════════════════════════════════
def build_endoscopy_diagnosis_prompt(endo_diagnosis):
    system_message = (
        "You are a senior endoscopy data analyst. "
        "Your task is to extract all disease entities with clear diagnostic significance from the endoscopy diagnosis text "
        "and structure them systematically.\n\n"

        "### 1. Extraction Targets\n"
        "Extract endoscopy diagnosis entities with clear diagnostic significance, including but not limited to: "
        "inflammatory diseases, ulcers, polyps, tumors, reflux esophagitis, atrophic gastritis, non-atrophic gastritis, "
        "submucosal elevation, anastomotic inflammation, remnant gastritis, bile reflux, etc.\n"
        "If the text contains multiple diagnosis entities, extract each separately.\n\n"

        "### 2. Do NOT Extract\n"
        "- Pure procedural actions: e.g., forceps removal, resection, biopsy, tissue sampling, submission for pathology.\n"
        "- Pure recommendations: e.g., recommend follow-up, recommend surgery, recommend EUS.\n"
        "- Pure findings that do not constitute a clear diagnosis: e.g., congestion, edema, redness, roughness; "
        "if they do not form a clear disease diagnosis, do not extract them separately.\n"
        "- Pure anatomical site names cannot serve as diagnosis entities alone.\n\n"

        "### 3. Output Field Definitions\n"
        "- source: always 'endoscopy'\n"
        "- full_text: original complete diagnosis phrase\n"
        "- normalized_text: simplified core disease name\n"
        "- location_modifier: specific site, e.g., angulus, antrum, fundus, below cardia, duodenal bulb; empty string if none\n"
        "- severity_modifier: degree modifier, e.g., mild, moderate, severe; empty string if none\n"
        "- grade_modifier: grading info, e.g., Grade A, Grade B, LA-A, low-grade, high-grade; empty string if none\n"
        "- status_modifier: staging/type/status info, e.g., A1 stage, H1 stage, S1 stage, C-1 type, C-2 type, post-surgery, recurrence; empty string if none\n"
        "- uncertain: 1 if the text contains uncertain expressions (e.g., 'consider', 'possible', 'suspected', 'cannot exclude'), 0 otherwise\n"
        "- clinical_action_related: 1 if the entity phrase contains management-related info (e.g., post-surgery, already treated, recommend treatment), 0 otherwise\n\n"

        "### 4. Important Rules\n"
        "- For endoscopy text, only extract based on the endoscopy original text; do not upgrade diagnoses using pathology results.\n"
        "- If the endoscopy only mentions 'polyp', keep normalized_text as 'polyp' or 'gastric polyp'/'colonic polyp'; "
        "do NOT write 'hyperplastic polyp' or 'adenomatous polyp' (those are pathology types).\n"
        "- Mild, moderate, severe belong to severity_modifier.\n"
        "- Grade A, Grade B, LA-A belong to grade_modifier.\n"
        "- A1 stage, H1 stage, C-1 type, C-2 type, post-surgery, recurrence belong to status_modifier.\n"
        "- 'Reflux esophagitis Grade A' -> normalized_text='reflux esophagitis', grade_modifier='Grade A'\n"
        "- 'Angular ulcer H1 stage' -> normalized_text='gastric ulcer', location_modifier='angulus', status_modifier='H1 stage'\n"
        "- 'Chronic atrophic gastritis (C2)' -> normalized_text='chronic atrophic gastritis', status_modifier='C-2 type'\n"
        "- 'Fundic elevated lesion, recommend EUS' -> extract normalized_text='submucosal elevation', "
        "location_modifier='fundus', clinical_action_related=1; but 'recommend EUS' itself should not be an entity.\n"
        "- For 'with bile reflux', if it has diagnostic significance, extract it separately as 'bile reflux'.\n"
        "- For 'with erosion', if it does not form a clear disease name (e.g., erosive gastritis), do not extract 'erosion' as a disease entity.\n\n"

        "### 5. Output Requirements\n"
        "- Return only pure JSON.\n"
        "- Top-level contains only one field: diagnosis_entities\n"
        "- diagnosis_entities is a list; if no clear diagnosis entity, return empty list []\n"
        "- Do not output explanation or extra fields."
    )

    user_message = (
        f"【Endoscopy Diagnosis Text】\n{endo_diagnosis if endo_diagnosis and endo_diagnosis.lower() != 'nan' else ''}\n\n"
        "【Output Template】\n"
        "{\n"
        '  "diagnosis_entities": [\n'
        "    {\n"
        '      "source": "endoscopy",\n'
        '      "full_text": "",\n'
        '      "normalized_text": "",\n'
        '      "location_modifier": "",\n'
        '      "severity_modifier": "",\n'
        '      "grade_modifier": "",\n'
        '      "status_modifier": "",\n'
        '      "uncertain": 0,\n'
        '      "clinical_action_related": 0\n'
        "    }\n"
        "  ]\n"
        "}"
    )

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]


def build_pathology_diagnosis_prompt(path_diagnosis):
    system_message = (
        "You are a senior pathology data analyst. "
        "Your task is to extract all entities with clear pathological diagnostic significance from the pathology diagnosis text "
        "and structure them systematically.\n\n"

        "### 1. Extraction Targets\n"
        "Extract entities with clear pathological diagnostic significance, including but not limited to: "
        "chronic gastritis, chronic active gastritis, chronic superficial gastritis, chronic atrophic gastritis, "
        "intestinal metaplasia, intraepithelial neoplasia, adenoma, polyp, hyperplastic polyp, adenocarcinoma, "
        "lymphoid hyperplasia, Hp infection, etc.\n"
        "If the text contains multiple diagnosis entities, extract each separately.\n\n"

        "### 2. Do NOT Extract\n"
        "- Pure procedural actions: e.g., biopsy, submission, tissue sampling, staining.\n"
        "- Pure negative results should not be extracted as positive disease entities.\n"
        "- HP(-), Helicobacter pylori negative, Hp not detected should not be extracted as disease entities.\n"
        "- Pure anatomical site names cannot serve as diagnosis entities alone.\n\n"

        "### 3. Output Field Definitions\n"
        "- source: always 'pathology'\n"
        "- full_text: original complete pathology diagnosis phrase\n"
        "- normalized_text: simplified core pathology diagnosis name\n"
        "- location_modifier: specific site, e.g., antrum, angulus, greater curvature of body, anastomosis; empty string if none\n"
        "- severity_modifier: degree modifier, e.g., mild, moderate, severe, mild-to-moderate, moderate-to-severe; empty string if none\n"
        "- grade_modifier: grade/differentiation degree, e.g., low-grade, high-grade, moderately differentiated, poorly differentiated; empty string if none\n"
        "- status_modifier: status word, e.g., active, post-surgery, recurrence; empty string if none\n"
        "- uncertain: 1 if the text contains uncertain expressions (e.g., 'consider', 'possible', 'suspected', 'cannot exclude'), 0 otherwise\n"
        "- clinical_action_related: generally 0 for pathology; set to 1 only if the entity phrase itself contains clear management-related info\n\n"

        "### 4. Important Rules\n"
        "- For 'chronic active inflammation', 'chronic inflammation', 'chronic superficial gastritis', 'chronic atrophic gastritis' in pathology, "
        "if the context clearly indicates gastric pathology, standardize to 'chronic active gastritis', 'chronic gastritis', "
        "'chronic superficial gastritis', 'chronic atrophic gastritis' respectively.\n"
        "- 'HP(+)' -> normalized_text='Hp infection'\n"
        "- 'HP(-)' -> do not extract\n"
        "- 'focal intestinal metaplasia', 'mild intestinal metaplasia', 'moderate intestinal metaplasia' -> normalized_text='intestinal metaplasia'\n"
        "- 'hyperplastic polyp formation' -> normalized_text='hyperplastic polyp'\n"
        "- 'polypoid hyperplasia' -> keep as 'polypoid hyperplasia'; do not simplify to 'polyp' unless the original text clearly diagnoses it as polyp.\n"
        "- For 'with intestinal metaplasia', 'with Hp infection', 'with lymphoid hyperplasia', 'with hyperplastic polyp formation' "
        "and other content with independent pathological diagnostic significance, split into separate entities.\n"
        "- For 'with erosion', if it is only an accompanying histological change, do not extract it as a disease entity.\n"
        "- If multiple entities in the same sentence share the same site, the split entities should inherit that site.\n\n"

        "### 5. Output Requirements\n"
        "- Return only pure JSON.\n"
        "- Top-level contains only one field: diagnosis_entities\n"
        "- diagnosis_entities is a list; if no clear diagnosis entity, return empty list []\n"
        "- Do not output explanation or extra fields."
    )

    user_message = (
        f"【Pathology Diagnosis Text】\n{path_diagnosis if path_diagnosis and path_diagnosis.lower() != 'nan' else ''}\n\n"
        "【Output Template】\n"
        "{\n"
        '  "diagnosis_entities": [\n'
        "    {\n"
        '      "source": "pathology",\n'
        '      "full_text": "",\n'
        '      "normalized_text": "",\n'
        '      "location_modifier": "",\n'
        '      "severity_modifier": "",\n'
        '      "grade_modifier": "",\n'
        '      "status_modifier": "",\n'
        '      "uncertain": 0,\n'
        '      "clinical_action_related": 0\n'
        "    }\n"
        "  ]\n"
        "}"
    )

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]


# ═══════════════════════════════════════════════════════════════
# Processing logic
# ═══════════════════════════════════════════════════════════════
def process_item(client, item, task):
    try:
        if task in ("qc", "all"):
            messages = build_qc_prompt(
                item.get("exam_item", ""),
                item.get("endo_report", ""),
            )
            result = call_llm(client, messages, max_tokens=300)
            item["qc_flags"] = result.get("qc_flags", {})

        if task in ("abnormality_flags", "all"):
            endo_report = item.get("endo_report", "")
            is_gastric = item.get("qc_flags", {}).get("is_colonoscopy", 0) == 0
            messages = build_abnormality_flag_prompt(endo_report, is_gastric)
            result = call_llm(client, messages, max_tokens=300)
            item["site_abnormality_flags"] = result.get("site_abnormality_flags", [])

        if task in ("findings", "all"):
            endo_report = item.get("endo_report", "")
            messages = build_findings_prompt(endo_report)
            result = call_llm(client, messages, max_tokens=300)
            item["abnormal_findings"] = result.get("abnormal_findings", [])

        if task in ("disease_entities", "all"):
            endo_diag = item.get("endo_diagnosis", "")
            path_diag = item.get("path_diagnosis", "")
            entities = []
            if endo_diag and endo_diag.lower() != "nan":
                messages = build_endoscopy_diagnosis_prompt(endo_diag)
                result = call_llm(client, messages, max_tokens=2000)
                entities.extend(result.get("diagnosis_entities", []))
            if path_diag and path_diag.lower() != "nan":
                messages = build_pathology_diagnosis_prompt(path_diag)
                result = call_llm(client, messages, max_tokens=2000)
                entities.extend(result.get("diagnosis_entities", []))
            item["diagnosis_entities"] = entities

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
            if line.strip():
                try:
                    ids.add(str(json.loads(line).get("id", "")))
                except json.JSONDecodeError:
                    continue
    return ids


def run(input_file, output_file, task, api_key, base_url):
    client = OpenAI(api_key=api_key, base_url=base_url)

    with open(input_file, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]

    processed = get_processed_ids(output_file)
    to_process = [d for d in data if str(d.get("id", "")) not in processed]

    print(f"Total: {len(data)} | Skipped: {len(processed)} | To process: {len(to_process)}")

    if not to_process:
        return

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_item, client, item, task): item for item in to_process}
        with open(output_file, "a", encoding="utf-8") as f_out:
            pbar = tqdm(as_completed(futures), total=len(to_process), desc="Processing")
            for future in pbar:
                if _interrupted:
                    break
                result = future.result()
                if result and "error" not in result:
                    f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                    f_out.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM-based endoscopy report annotation")
    parser.add_argument("--task", default="all",
                        choices=["all", "qc", "abnormality_flags", "findings", "disease_entities"],
                        help="Which annotation task to run")
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--output", required=True, help="Output JSONL file")
    parser.add_argument("--api_key", default=os.environ.get("DASHSCOPE_API_KEY", ""),
                        help="DashScope API key (or set DASHSCOPE_API_KEY env var)")
    parser.add_argument("--base_url", default="https://dashscope.aliyuncs.com/compatible-mode/v1",
                        help="DashScope API base URL")
    args = parser.parse_args()

    if not args.api_key:
        raise ValueError("Please provide --api_key or set DASHSCOPE_API_KEY environment variable.")

    run(args.input, args.output, args.task, args.api_key, args.base_url)
