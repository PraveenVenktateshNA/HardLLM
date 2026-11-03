"""Write multiple-choice questions as the task file llama-perplexity --multiple-choice reads.

Each question becomes "Question: <question>\\nAnswer:" (the format every
fine-tuned model was trained on, and plain text for the base model) and each
option is scored as the answer's full text: the model's log-likelihood of the
option, so no model has to write a letter. Plain Python.

    python3 HardLLM/Step_5/eval/make_mc_tasks.py IN.jsonl OUT.bin [--limit N]

IN.jsonl rows: {"question": str, "choices": [str, ...], "answer": index}.
File layout (llama.cpp tools/perplexity): uint32 n_tasks, uint32 offset of
each task, then per task: string question, then two answer sets (mc1, mc2),
each uint32 n, n strings, n int32 labels (1 = correct). Strings are uint32
length + UTF-8 bytes. mc2 is left empty.
"""
import argparse
import json
import struct


def string(s):
    b = s.encode("utf-8")
    return struct.pack("<I", len(b)) + b


def answers(choices, correct):
    out = struct.pack("<I", len(choices)) + b"".join(string(c) for c in choices)
    return out + struct.pack(f"<{len(choices)}i", *[int(i == correct) for i in range(len(choices))])


def task(row):
    question = f"Question: {' '.join(row['question'].split())}\nAnswer:"
    choices = [" " + " ".join(str(c).split()) for c in row["choices"]]
    return string(question) + answers(choices, row["answer"]) + answers([], -1)


def write(rows, path):
    blobs = [task(r) for r in rows]
    head = 4 + 4 * len(blobs)
    offsets, pos = [], head
    for b in blobs:
        offsets.append(pos)
        pos += len(b)
    with open(path, "wb") as f:
        f.write(struct.pack("<I", len(blobs)) + struct.pack(f"<{len(blobs)}I", *offsets) + b"".join(blobs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    with open(args.inp, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    if args.limit:
        rows = rows[:args.limit]
    write(rows, args.out)
    print(f"{len(rows)} tasks -> {args.out}")


if __name__ == "__main__":
    main()
