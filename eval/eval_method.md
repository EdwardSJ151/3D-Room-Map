# Final Evaluation Method

## Evaluation dataset

Use 15 room images.

For each image, run the full Paper AR Space mapping pipeline once. Store the generated object records, the stage timings, and the final scene-memory index.

For each image, generate 60 evaluation questions, with 10 questions from each category:

1. positive object existence
2. negative object existence
3. attribute grounding
4. category retrieval
5. affordance retrieval
6. spatial or local relation

Total QA set:

```text
15 images × 60 questions = 900 questions
```

Each question should include:

```text
scene_id
question_id
category
question
expected_answer
expected_visible_evidence
acceptable_alternatives
```

The expected evidence can be object names or short descriptions. It does not need full bounding boxes or dense annotation.

---

# Metric 1: Mapping Performance

## Purpose

Measure the one-time cost of converting a single image into a usable object memory.

This metric should show both absolute mapping cost and cost normalized by the number of generated object records.

## Required logged values per image

For each image, log:

```text
num_generated_records
cutr_inference_time
crop_generation_time
qwen_labeling_time
embedding_indexing_time
total_capture_to_memory_time
```

## Final reported numbers

### 1.1 CuTR inference time

```text
mean(cutr_inference_time)
std(cutr_inference_time)
```

This is reported per image because CuTR runs once on the full capture.

### 1.2 Crop generation time per object record

```text
crop_generation_time / num_generated_records
```

Then report:

```text
mean(crop_generation_time_per_record)
std(crop_generation_time_per_record)
```

This normalizes crop generation by scene complexity.

### 1.3 Qwen labeling time per object record

```text
qwen_labeling_time / num_generated_records
```

Then report:

```text
mean(qwen_labeling_time_per_record)
std(qwen_labeling_time_per_record)
```

This is important because labeling scales with the number of detected objects.

### 1.4 Embedding/indexing time per object record

```text
embedding_indexing_time / num_generated_records
```

Then report:

```text
mean(embedding_indexing_time_per_record)
std(embedding_indexing_time_per_record)
```

This measures the cost of adding one object to the memory.

### 1.5 Total capture-to-memory time

```text
mean(total_capture_to_memory_time)
std(total_capture_to_memory_time)
```

This is the full one-time mapping cost.

### 1.6 Total capture-to-memory time per object record

```text
total_capture_to_memory_time / num_generated_records
```

Then report:

```text
mean(total_capture_to_memory_time_per_record)
std(total_capture_to_memory_time_per_record)
```

This is the main normalized mapping number.

---

# Metric 2: Query Performance

## Purpose

Measure how expensive it is to answer a question after the scene memory already exists.

This should be reported per query and also interpreted relative to memory size.

## Required logged values per question

For each question, log:

```text
scene_id
question_id
num_records_in_scene
retrieval_time
response_generation_time
total_query_to_answer_time
```

## Final reported numbers

### 2.1 Retrieval time per query

```text
mean(retrieval_time)
std(retrieval_time)
```

This measures the cost of searching the object memory.

### 2.2 Retrieval time per 100 object records

```text
(retrieval_time / num_records_in_scene) × 100
```

Then report:

```text
mean(retrieval_time_per_100_records)
std(retrieval_time_per_100_records)
```

This normalizes retrieval cost by scene-memory size.

### 2.3 Response generation time per query

```text
mean(response_generation_time)
std(response_generation_time)
```

This measures the cost of generating the answer after retrieval.

### 2.4 Total query-to-answer time per query

```text
mean(total_query_to_answer_time)
std(total_query_to_answer_time)
```

This is the user-facing dialogue latency.

---

# Metric 3: Object-Memory Usability

## Purpose

Measure whether the pipeline produces object records that are usable for spatial-semantic dialogue.

This metric evaluates the complete output record, not the detector or language model separately.

## Required inputs per generated object record

For each object record, the evaluator should receive:

```text
scene image
object crop
object box or box visualization
object name
object description
object tags
spatial anchor visualization if available
```

## Per-record judgment

Each generated object record should be judged as one of:

```text
usable
partial
unusable
```

Use the following rules.

A record is **usable** if:

```text
it corresponds to a visible object
the object name is reasonable
the description is visually supported
the tags are visually supported
it is not a clear duplicate
the spatial anchor is good enough for embodied reference
```

A record is **partial** if:

```text
it corresponds to a real visible object
but the name, description, tags, duplicate status, or anchor is imperfect
```

A record is **unusable** if:

```text
it does not correspond to a clear object
or the semantic label is wrong enough to mislead retrieval/dialogue
or it is a clear duplicate with no added value
or the anchor is unusable for reference
```

## Final reported numbers

### 3.1 Average generated records per scene

```text
total_generated_records / 15
```

### 3.2 Usable object-record rate

```text
usable_records / generated_records
```

### 3.3 Partial object-record rate

```text
partial_records / generated_records
```

### 3.4 Unusable object-record rate

```text
unusable_records / generated_records
```

### 3.5 Average usable records per scene

```text
usable_records / 15
```

This is the strongest memory-quality number because it estimates how many usable spatial-semantic objects the system produces per room capture.

---

# Metric 4: Strict Grounded QA Success

## Purpose

Measure whether the assistant can answer room questions correctly, using the mapped object memory, without inventing unsupported objects.

This is the main end-to-end quality metric.

## Question categories

Generate 10 questions per category for each image.

### Category 1: Positive object existence

Tests whether the system can confirm visible objects.

Example:

```text
Did you observe a chair in the room?
```

Expected behavior:

```text
Answer yes if the object is visible and mapped.
```

### Category 2: Negative object existence

Tests whether the system avoids hallucinating absent objects.

Example:

```text
Did you observe a backpack in the room?
```

Expected behavior:

```text
Answer that the object was not observed if it is absent or unsupported by memory.
```

### Category 3: Attribute grounding

Tests whether object attributes are preserved.

Example:

```text
What color is the chair?
```

Expected behavior:

```text
Answer using visible attributes supported by the mapped record.
```

### Category 4: Category retrieval

Tests whether the memory supports object groups.

Example:

```text
What electronics are visible in the room?
```

Expected behavior:

```text
List only mapped objects that fit the category.
```

### Category 5: Affordance retrieval

Tests whether the memory supports functional queries.

Example:

```text
What could I use to sit down?
```

Expected behavior:

```text
Answer with mapped objects that support the requested use.
```

### Category 6: Spatial or local relation

Tests whether the memory supports approximate local relations.

Example:

```text
What objects are on the desk?
```

Expected behavior:

```text
Answer using mapped objects and avoid unsupported spatial claims.
```

## Required logged values per question

For each question, log:

```text
scene_id
question_id
category
question
expected_answer
retrieved_records
assistant_answer
```

## Per-question judgment

Each answer is judged as:

```text
success
failure
```

An answer is a **success** only if all conditions are true:

```text
the answer is correct for the original image
the answer is supported by retrieved object records
the answer does not mention unsupported objects
for negative questions, the answer correctly says the object was not observed
```

An answer is a **failure** if any condition is false.

## Failure reason

For every failed answer, assign one primary failure reason:

```text
missing_memory_record
semantic_record_error
retrieval_error
unsupported_generation
failed_abstention
incomplete_answer
ambiguous_question
```

## Final reported numbers

### 4.1 Strict Grounded QA Success

```text
successful_answers / 900
```

This is the main QA result.

### 4.2 Category-level success rates

Report the same success calculation for each category:

```text
positive_object_existence_success
negative_object_existence_success
attribute_grounding_success
category_retrieval_success
affordance_retrieval_success
spatial_or_local_relation_success
```

Each category has:

```text
15 images × 10 questions = 150 questions
```

### 4.3 Failure distribution

For failed questions only:

```text
count(failure_reason) / total_failed_questions
```

Report the distribution across:

```text
missing_memory_record
semantic_record_error
retrieval_error
unsupported_generation
failed_abstention
incomplete_answer
ambiguous_question
```

This shows where the pipeline fails without turning the paper into a standalone CuTR or Qwen benchmark.

---

# Confidence intervals

For all main rates, compute confidence intervals using scene-level resampling.

Do not treat all 900 questions as fully independent, because questions from the same image share the same scene memory.

Use the 15 scenes as the resampling units.

Report confidence intervals for:

```text
usable object-record rate
strict grounded QA success
category-level QA success rates
```

---

# Final output tables

## Table 1: System performance

Rows:

```text
CuTR inference time
crop generation time per object record
Qwen labeling time per object record
embedding/indexing time per object record
total capture-to-memory time
total capture-to-memory time per object record
retrieval time per query
retrieval time per 100 object records
response generation time per query
total query-to-answer time per query
```

Columns:

```text
mean
std
```

## Table 2: Memory and QA quality

Rows:

```text
generated records per scene
usable records per scene
usable object-record rate
partial object-record rate
unusable object-record rate
strict grounded QA success
positive object existence success
negative object existence success
attribute grounding success
category retrieval success
affordance retrieval success
spatial/local relation success
```

Columns:

```text
score
95% confidence interval
```

## Table 3: Failure analysis

Rows:

```text
missing memory record
semantic record error
retrieval error
unsupported generation
failed abstention
incomplete answer
ambiguous question
```

Columns:

```text
count
percentage of failed questions
```
