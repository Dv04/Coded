# Zero-shot ShapesQA with Qwen2.5-VL (no finetuning).

import os
import re
import json
import copy
import time
from collections import defaultdict

import torch
import transformers
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

print("transformers version:", transformers.__version__)
print(f"[{time.strftime('%H:%M:%S')}] Loading model...")

MODEL_ID = os.environ.get("SHAPESQA_VLM_ID", "Qwen/Qwen2.5-VL-3B-Instruct")
MAX_EVAL_SAMPLES = None  # set e.g. 200 for quick tests

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=model_dtype,
    low_cpu_mem_usage=True,
    attn_implementation="eager",
)
model.to(device)
model.eval()

processor = AutoProcessor.from_pretrained(MODEL_ID, use_fast=False)

print(f"[{time.strftime('%H:%M:%S')}] Model loaded: {MODEL_ID}")

# Clean generation config
gen_cfg = copy.deepcopy(model.generation_config)
gen_cfg.do_sample = False
gen_cfg.temperature = None
gen_cfg.top_p = None
gen_cfg.top_k = None
gen_cfg.typical_p = None
gen_cfg.min_p = None
gen_cfg.epsilon_cutoff = None
gen_cfg.eta_cutoff = None

VALID_ANSWERS = [
    "red", "blue", "green", "yellow", "purple", "orange",
    "circle", "square", "triangle",
    "0", "1", "2", "3", "4", "5",
]
VALID_SET = set(VALID_ANSWERS)
WORD_TO_DIGIT = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "no": "0", "none": "0",
}
# Color name variants
COLOR_ALIASES = {
    "violet": "purple",
    "magenta": "purple",
    "cyan": "blue",
    "lime": "green",
    "crimson": "red",
    "scarlet": "red",
    "amber": "orange",
    "golden": "yellow",
    "gold": "yellow",
}


def normalize_answer(text):
    t = text.lower().strip()
    t = re.sub(r"[^a-z0-9 ]", " ", t)
    t = " ".join(t.split())

    tokens = t.split()
    # Apply word-to-digit and color alias mappings
    tokens = [WORD_TO_DIGIT.get(tok, COLOR_ALIASES.get(tok, tok)) for tok in tokens]

    # First pass: exact token match
    for tok in tokens:
        if tok in VALID_SET:
            return tok

    # Second pass: substring match
    joined = " ".join(tokens)
    for ans in VALID_ANSWERS:
        if ans in joined:
            return ans

    return "<unk>"


# ── System and user prompts ───────────────────────────────────────────────────

system_prompt = (
    "You are a precise visual question answering assistant. "
    "You will see an image containing simple geometric shapes (circles, squares, triangles) "
    "in different colors (red, blue, green, yellow, purple, orange) on a black background. "
    "Answer each question with exactly ONE word or number. "
    "Valid answers are ONLY: red, blue, green, yellow, purple, orange, circle, square, triangle, 0, 1, 2, 3, 4, 5. "
    "Never explain. Never output anything except the single answer."
)

instruction = (
    "Look at the image carefully. "
    "Answer the question with EXACTLY ONE word or number from this list: "
    "red, blue, green, yellow, purple, orange, circle, square, triangle, 0, 1, 2, 3, 4, 5.\n"
    "Rules:\n"
    "- For counting questions: answer with a number (0, 1, 2, 3, 4, or 5)\n"
    "- For color questions: answer with a color name (red, blue, green, yellow, purple, or orange)\n"
    "- For shape questions: answer with a shape name (circle, square, or triangle)\n"
    "- Output ONLY the answer, nothing else."
)


# ── Load dataset ──────────────────────────────────────────────────────────────
val_annotations = json.load(open("ShapesQA_valset/annotations.json"))
if MAX_EVAL_SAMPLES is not None:
    val_annotations = val_annotations[:MAX_EVAL_SAMPLES]

print(f"Evaluating {len(val_annotations)} validation samples")

correct = 0
unknown = 0
start_time = time.time()

# Detailed Tracking
type_correct = defaultdict(int)
type_total = defaultdict(int)
type_unknown = defaultdict(int)

per_answer_correct = defaultdict(int)
per_answer_total = defaultdict(int)
confusion = defaultdict(lambda: defaultdict(int))
error_examples = defaultdict(list)

for i, ann in enumerate(val_annotations):
    image_path = os.path.join("ShapesQA_valset", ann["image"])
    question = ann["question"]
    q_type = ann.get("question_type", "unknown")

    user_prompt = f"{instruction}\nQuestion: {question}\nAnswer:"

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": user_prompt},
            ],
        },
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            generation_config=gen_cfg,
            max_new_tokens=4,
            use_cache=True,
        )

    new_tokens = output_ids[:, inputs["input_ids"].shape[-1]:]
    decoded = processor.batch_decode(
        new_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    pred = normalize_answer(decoded)
    gt = str(ann["answer"]).strip().lower()

    # Track overall & type
    type_total[q_type] += 1
    per_answer_total[gt] += 1
    
    if pred == "<unk>":
        unknown += 1
        type_unknown[q_type] += 1
        
    if pred == gt:
        correct += 1
        type_correct[q_type] += 1
        per_answer_correct[gt] += 1
    else:
        confusion[gt][pred] += 1
        if len(error_examples[q_type]) < 5:
            error_examples[q_type].append((question, decoded, pred, gt))

    if (i + 1) % 100 == 0:
        elapsed = time.time() - start_time
        running_acc = correct / (i + 1)
        print(f"  [{time.strftime('%H:%M:%S')}] {i+1}/{len(val_annotations)} "
              f"| acc={running_acc:.4f} | {elapsed:.0f}s elapsed")

total_elapsed = time.time() - start_time
final_acc = correct / max(1, len(val_annotations))
print(f"\nFinal ShapesQA zero-shot accuracy: {final_acc:.4f} ({100*final_acc:.2f}%)")
print(f"Unknown / unparsable outputs: {unknown}")
print(f"Total time: {total_elapsed:.1f}s")

# Per-question-type accuracy
print("\n--- Accuracy by question type ---")
for qt in sorted(type_total.keys()):
    c = type_correct[qt]
    t = type_total[qt]
    u = type_unknown[qt]
    print(f"  {qt:12s}: {c}/{t} = {100*c/t:.1f}%  (unknown: {u})")

# Per-answer accuracy
print("\n--- Per-Answer Accuracy ---")
for ans in sorted(per_answer_total.keys(), key=lambda a: per_answer_correct.get(a,0)/max(1,per_answer_total[a])):
    c = per_answer_correct.get(ans, 0)
    t = per_answer_total[ans]
    print(f"  {ans:10s}: {c:3d}/{t:3d} = {100*c/t:.1f}%")

# Top confusion pairs
print("\n--- Top Confusion Pairs (true -> predicted: count) ---")
all_confusions = []
for gt, preds_dict in confusion.items():
    for p, cnt in preds_dict.items():
        all_confusions.append((gt, p, cnt))

for gt, p, cnt in sorted(all_confusions, key=lambda x: -x[2])[:20]:
    print(f"  {gt:10s} -> {p:10s}: {cnt}")

# Sample errors per type
print("\n--- Sample errors per type ---")
for qt in sorted(error_examples.keys()):
    print(f"\n  [{qt}]")
    for q, raw, p_ans, gt in error_examples[qt]:
        print(f"    Q: {q}")
        print(f"    Raw: '{raw}' -> Pred: '{p_ans}' | GT: '{gt}'")

