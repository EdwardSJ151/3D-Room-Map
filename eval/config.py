from __future__ import annotations

import os
from pathlib import Path

# --- Service endpoints ---
CUTR_BASE = os.environ.get("CUTR_BASE", "http://localhost:8090").rstrip("/")
QWEN_BASE  = os.environ.get("QWEN_BASE",  "http://localhost:8091").rstrip("/")
ADK_BASE   = os.environ.get("ADK_BASE",   "http://localhost:8002").rstrip("/")
ADK_APP    = os.environ.get("ADK_APP",    "env_agent")

# --- Ground-truth / question generation ---
GENERATION_BACKEND       = os.environ.get("GENERATION_BACKEND",       "vllm")   # "gemini" or "vllm"
GEMINI_GT_MODEL          = os.environ.get("GEMINI_GT_MODEL",          "gemini-3-pro")
GEMINI_QUESTION_MODEL    = os.environ.get("GEMINI_QUESTION_MODEL",    "gemini-3-pro")
VLLM_GENERATION_BASE_URL = os.environ.get("VLLM_GENERATION_BASE_URL", "http://localhost:8030/v1")
VLLM_GT_MODEL            = os.environ.get("VLLM_GT_MODEL",            "QuantTrio/Qwen3.5-397B-A17B-AWQ")
VLLM_QUESTION_MODEL      = os.environ.get("VLLM_QUESTION_MODEL",      "QuantTrio/Qwen3.5-397B-A17B-AWQ")
GOOGLE_API_KEY           = os.environ.get("GOOGLE_API_KEY", "")

# --- Judge (vLLM: QuantTrio/Qwen3.5-397B-A17B-AWQ on localhost:8030) ---
JUDGE_BACKEND       = os.environ.get("JUDGE_BACKEND",       "vllm")   # "gemini" or "vllm"
JUDGE_MODEL         = os.environ.get("JUDGE_MODEL",         "gemini-3-pro")
VLLM_JUDGE_BASE_URL = os.environ.get("VLLM_JUDGE_BASE_URL", "http://localhost:8030/v1")
VLLM_JUDGE_MODEL    = os.environ.get("VLLM_JUDGE_MODEL",    "QuantTrio/Qwen3.5-397B-A17B-AWQ")

# --- Paths ---
REPO_ROOT   = Path(__file__).resolve().parent.parent
EVAL_DIR    = REPO_ROOT / "eval"
SCENES_DIR  = EVAL_DIR / "scenes"
RESULTS_DIR = EVAL_DIR / "results"
SCENES_JSON = SCENES_DIR / "scenes.json"

CUTR_JOBS_DIR = REPO_ROOT / "cutr_jobs"
QWEN_JOBS_DIR = REPO_ROOT / "qwen_jobs"

# --- Polling ---
POLL_INTERVAL_S = float(os.environ.get("EVAL_POLL_S", "2.0"))
POLL_TIMEOUT_S  = float(os.environ.get("EVAL_POLL_TIMEOUT_S", "600.0"))

# --- Prompt versioning ---
PROMPT_VERSION = "v1"

# --- Ground-truth prompts ---
GT_ROOM_PROMPT = """\
You are a ground-truth annotator for an indoor scene evaluation dataset.

Analyze this room image and return a JSON object with EXACTLY these keys:
- "room_type": short string describing the type of room (e.g. "bedroom", "office", "kitchen")
- "room_description": 3-5 sentence natural-language description of the room
- "objects_present": list of object names (strings) that are clearly visible in the image
- "objects_absent_examples": list of exactly 15 object names that someone might reasonably \
expect to find in this type of room, but that are NOT visibly present in this image. \
Do not list objects that are present. Avoid objects that would never belong in this room type \
(e.g. do not list a forklift in a bedroom). Be realistic and varied.

Return ONLY valid JSON. No markdown, no explanation."""

GT_BBOX_PROMPT = """\
You are a ground-truth annotator. This image is a crop of a single detected object from a room scene.

Analyze the crop and return a JSON object with EXACTLY these keys:
- "object_name": the primary object shown (short noun phrase)
- "rich_description": 1-2 sentences describing the object's appearance, color, material, and any notable features
- "attributes": an object with keys such as "color", "material", "size", "condition", "brand" (only include what is visible)
- "location_hint": a short phrase describing where the object appears to be located in the room (e.g. "on the desk", "near the window")

Return ONLY valid JSON. No markdown, no explanation."""

# --- Question generation prompt ---
QUESTION_GENERATION_PROMPT = """\
You are an evaluation dataset generator for an indoor scene question-answering system.

You will receive:
- A room description and list of present/absent objects
- Descriptions of all detected bounding-box objects (with their idx numbers)

Generate EXACTLY 60 evaluation questions: 10 per category.

Categories:
1. positive_object_existence — ask if a specific PRESENT object is in the room (requires tool call)
2. negative_object_existence — ask if a specific ABSENT object is in the room (requires tool call)
3. attribute_grounding — ask about a specific visual attribute (color, material, size) of a PRESENT object (requires tool call)
4. category_retrieval — ask to list all objects of a type/category (requires tool call)
5. affordance_retrieval — ask what objects can be used for a specific function (requires tool call)
6. spatial_or_local_relation — ask about spatial relationships between objects (requires tool call)

IMPORTANT: Every question MUST require a tool call to the object memory to answer correctly. \
Do NOT generate questions answerable from general knowledge alone.

Return a JSON object with key "questions" containing a list of 60 objects, each with:
- "question_id": string like "q001" through "q060"
- "category": one of the 6 category names above
- "question": the question string
- "expected_answer": what a correct answer should convey
- "expected_visible_evidence": object name(s) or descriptions that support the answer
- "acceptable_alternatives": list of alternative phrasings that would also be correct (can be empty)
- "requires_absent_object": true if the question targets an absent object, false otherwise
- "target_idx": integer bbox idx this question targets (null if category-level or multi-object)

Rules:
- Exactly 10 questions per category. No more, no less.
- No duplicate question text.
- Repeated target_idx is allowed if the question type differs.
- For negative_object_existence, use objects from objects_absent_examples.
- For spatial/attribute/affordance questions, use objects from bbox_descriptions.

Return ONLY valid JSON. No markdown, no explanation."""

# --- Judge prompts ---
EVIDENCE_RETRIEVAL_PROMPT_TEMPLATE = """\
You are an evidence retrieval judge for an indoor scene QA evaluation.

Your task: determine whether the top-5 retrieved records contain sufficient mapped evidence \
to answer the question.

QUESTION:
Category: {category}
Question: {question}
Expected answer: {expected_answer}
Expected visible evidence: {expected_visible_evidence}

TOP-5 RETRIEVED RECORDS:
{retrieved_records_text}
Retrieved idx list: {retrieved_idx_list}

RULES:
1. Parse expected_visible_evidence to identify required evidence objects. Each \
comma-separated item is one required object. If an item lists idx alternatives \
(e.g. "laptop (idx=2 or idx=9)"), treat all alternatives as a single object — \
it is found if ANY of its idx alternatives appears in the retrieved idx list.
2. Ignore evidence items that have no idx reference (e.g. "keyboard (mentioned in context)") \
— these are unmapped and cannot be evaluated.
3. Count R = number of mapped required objects (those with at least one idx reference).
4. If R = 0 after ignoring unmapped items, return not_applicable.
5. Count M = number of those mapped objects found in the retrieved idx list.
6. required_hits = min(R, 3). Result is true if M >= required_hits.

Return a JSON object with EXACTLY these keys:
- "evidence_retrieval_at_5": "true", "false", or "not_applicable"
- "evidence_retrieval_explanation": 1-2 sentences explaining the judgment

Return ONLY valid JSON."""

BEST_IDX_PROMPT_TEMPLATE = """\
You are a best-idx accuracy judge for an indoor scene QA evaluation.

Your task: determine whether best_idx is an appropriate object anchor for the question.

QUESTION:
Category: {category}
Question: {question}
Expected answer: {expected_answer}
Expected visible evidence: {expected_visible_evidence}
Target idx: {target_idx}

RETRIEVED RECORDS:
{retrieved_records_text}

ASSISTANT ANSWER:
{assistant_answer}

BEST IDX SELECTED: {best_idx}

RULES:
1. best_idx is correct if it points to one of the valid evidence objects for the question \
as indicated by expected_visible_evidence.
2. For multi-object questions, best_idx only needs to point to ONE valid object.
3. If best_idx is null for an applicable question, result is "false" unless no object \
anchor is meaningful for this question.
4. Return "not_applicable" only if no object anchor is meaningful for this question type.
5. This judgment is independent of Evidence Retrieval@5.

Return a JSON object with EXACTLY these keys:
- "best_idx_accuracy": "true", "false", or "not_applicable"
- "best_idx_explanation": 1-2 sentences explaining the judgment

Return ONLY valid JSON."""

JUDGE_PROMPT_TEMPLATE = """\
You are a grounded-QA judge for an indoor scene assistant evaluation.

CONTEXT:
- Room description: {room_description}
- Ground-truth bbox descriptions for retrieved records: {bbox_gt_descriptions}

QUESTION:
Category: {category}
Question: {question}
Expected answer: {expected_answer}

ASSISTANT ANSWER:
{assistant_answer}

TOOL WAS CALLED: {tool_was_called}
RETRIEVED IDX LIST: {retrieved_idx_list}
TARGET IDX: {target_idx}

JUDGMENT RULES:
1. If tool_was_called is false: judgment = "failure", failure_reason = "retrieval_error"
2. The answer must be correct for the actual room (as described in room_description and bbox gt)
3. The answer must be grounded in what is visually plausible for the scene — retrieved records \
   and room description are reference annotations, not exhaustive truth. Accept visually \
   reasonable claims that go beyond the annotation if they are consistent with the scene.
4. For negative existence questions: the answer must confirm the object was NOT observed; \
   the judge still evaluates this — do NOT auto-pass negative questions
5. Fail the answer only if it mentions objects that are clearly absent from the room entirely \
   — do not fail for mentioning correct visual details that happen to be missing from the \
   ground-truth annotation. Do not fail for imprecise or approximate spatial descriptions \
   when the object location is visually consistent with the scene.
6. Accept reasonable paraphrases, synonyms, and functional descriptions of objects — the \
   assistant describes objects in natural language, not by ground-truth labels. If a \
   description plausibly refers to a visible object, treat it as a match. Only fail if the \
   described object is different from anything present in the room. Do not fail when the \
   assistant confuses visually similar or functionally overlapping objects that could \
   reasonably be mistaken for one another in the scene.
7. For category_retrieval questions, an answer is a success if it correctly identifies the \
   majority of relevant objects — do not fail for missing a small number of items unless the \
   omission makes the answer misleading.

Return a JSON object with EXACTLY these keys:
- "judgment": "success" or "failure"
- "failure_reason": null if success, else one of: \
  "missing_memory_record", "semantic_record_error", "retrieval_error", \
  "unsupported_generation", "failed_abstention", "incomplete_answer", "ambiguous_question"
- "judge_explanation": 1-2 sentences explaining the judgment
- "retrieved_idx_correct": true/false if target_idx is not null, else null

Return ONLY valid JSON."""
