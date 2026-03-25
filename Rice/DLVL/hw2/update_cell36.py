import json

with open('COMP646_Vision_and_Language_Assignment_02_Spring_2026_upgraded.ipynb', 'r') as f:
    nb = json.load(f)

# Re-write the validation loop in cell 36 to include comprehensive tracking
src = ''.join(nb['cells'][36]['source'])

target_script = r'''# ── Load dataset ──────────────────────────────────────────────────────────────
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
'''

import re
# We'll use regex to split the source safely
pattern = r'# ── Load dataset ──────────────────────────────────────────────────────────────.*?$'
new_src = re.sub(pattern, target_script, src, flags=re.DOTALL)

lines = new_src.split('\n')
nb['cells'][36]['source'] = [line + '\n' for line in lines[:-1]] + [lines[-1]]

with open('COMP646_Vision_and_Language_Assignment_02_Spring_2026_upgraded.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print("Updated Cell 36 successfully with comprehensive analytics.")
