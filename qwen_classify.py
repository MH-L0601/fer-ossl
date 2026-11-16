"""Ask Qwen2-VL to directly classify each RAF-DB face into one of 7 emotions.

This measures the *upper bound* of the text/VLM route: if the VLM cannot even
classify the 7 expressions directly, then downstream text clustering + alignment
with image clustering is unlikely to help.

Reads the same image list used by caption_rafdb.py (from the captions jsonl) so
results are directly comparable to text_cluster_probe.py.

Output: jsonl with {"img", "label", "pred", "pred_idx", "correct"} + prints a
confusion matrix and per-class / overall / balanced accuracy.
"""
import argparse
import json
import os
import re
import time

import numpy as np
import torch

ID2EMO = {
    0: 'surprise', 1: 'fear', 2: 'disgust', 3: 'happiness',
    4: 'sadness', 5: 'anger', 6: 'neutral',
}
EMO2ID = {v: k for k, v in ID2EMO.items()}

# category name -> canonical idx, with common morphological variants
PATTERNS = [
    ('surprise', 0), ('surprised', 0),
    ('fear', 1), ('fearful', 1), ('scared', 1),
    ('disgust', 2), ('disgusted', 2),
    ('happiness', 3), ('happy', 3),
    ('sadness', 4), ('sad', 4),
    ('anger', 5), ('angry', 5),
    ('neutral', 6),
]

PROMPT = (
    "Classify the facial expression in this image into exactly one of these 7 "
    "categories: surprise, fear, disgust, happiness, sadness, anger, neutral. "
    "Answer with only the category name, one word, no explanation."
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='/root/autodl-tmp/DebGCD-expclip/RAF-DB/RAF-DB_train')
    p.add_argument('--captions', default='dev_outputs/rafdb_captions.jsonl',
                   help='read the same img+label list from this jsonl')
    p.add_argument('--out', default='dev_outputs/rafdb_vlm_preds.jsonl')
    p.add_argument('--model', default='Qwen/Qwen2-VL-2B-Instruct')
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--max-new-tokens', type=int, default=16)
    return p.parse_args()


def parse_pred(text):
    """Return canonical category idx from the VLM's free-text answer."""
    t = text.strip().lower()
    # exact-ish word match first
    for word, idx in PATTERNS:
        if re.search(r'\b' + re.escape(word) + r'\b', t):
            return idx
    # fallback: substring match (longer first to avoid sad/sadness ambiguity)
    for word, idx in sorted(PATTERNS, key=lambda x: -len(x[0])):
        if word in t:
            return idx
    return -1


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    img_dir = os.path.join(args.root, 'Image', 'aligned')

    items = []
    with open(args.captions, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            items.append((d['img'], int(d['label'])))
    print(f'[vlm-cls] {len(items)} images to classify')

    # resume
    done = set()
    if os.path.exists(args.out):
        with open(args.out, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    done.add(json.loads(line)['img'])
                except Exception:
                    pass
    todo = [(i, l) for i, l in items if i not in done]
    print(f'[vlm-cls] {len(done)} done, {len(todo)} to process')

    if not todo:
        print('[vlm-cls] nothing to do.')
        return

    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation='eager',
    ).to(device).eval()
    processor = AutoProcessor.from_pretrained(args.model)

    out_f = open(args.out, 'a')
    start = time.time()
    n = 0
    i = 0
    while i < len(todo):
        batch = todo[i:i + args.batch_size]
        msgs, paths = [], []
        for img, lab in batch:
            p = os.path.join(img_dir, img)
            msgs.append([{'role': 'user', 'content': [
                {'type': 'image', 'image': p},
                {'type': 'text', 'text': PROMPT},
            ]}])
            paths.append((img, lab))

        texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                 for m in msgs]
        image_inputs, video_inputs = process_vision_info(msgs)
        inputs = processor(text=texts, images=image_inputs, videos=video_inputs,
                           padding=True, return_tensors='pt').to(device)
        with torch.no_grad():
            gids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
        gids = gids[:, inputs.input_ids.shape[1]:]
        answers = processor.batch_decode(gids, skip_special_tokens=True)

        for (img, lab), ans in zip(paths, answers):
            pred = parse_pred(ans)
            correct = int(pred == lab)
            out_f.write(json.dumps({'img': img, 'label': lab, 'pred': pred,
                                    'answer': ans.strip(), 'correct': correct}) + '\n')
            n += 1
        out_f.flush()
        i += len(batch)

        if n % 20 == 0 or i >= len(todo):
            rate = n / max(time.time() - start, 1e-6)
            eta = (len(todo) - i) / max(rate, 1e-6)
            print(f'[vlm-cls] {n}/{len(todo)} done, {rate:.2f} img/s, ETA {eta/60:.1f} min')

    out_f.close()
    print(f'[vlm-cls] finished. wrote {n} predictions to {args.out}')

    # ---- evaluate ----
    labels, preds = [], []
    with open(args.out, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            labels.append(d['label'])
            preds.append(d['pred'])
    labels = np.array(labels)
    preds = np.array(preds)
    valid = preds >= 0
    print(f'\n[eval] valid predictions (parseable): {valid.sum()}/{len(preds)}')

    from sklearn.metrics import accuracy_score, confusion_matrix, recall_score
    # treat unparseable as wrong
    preds_filled = preds.copy()
    preds_filled[~valid] = -1
    acc = accuracy_score(labels, preds_filled)
    bal = recall_score(labels, preds_filled, average='macro')
    print(f'[eval] overall accuracy (unparseable=wrong): {acc:.4f}')
    print(f'[eval] balanced accuracy (macro recall): {bal:.4f}')

    print('\nper-class accuracy:')
    for c in range(7):
        m = labels == c
        a = accuracy_score(labels[m], preds_filled[m])
        print(f'  {ID2EMO[c]:10s} ({m.sum():4d} samples): {a:.4f}')

    cm = confusion_matrix(labels, preds_filled, labels=list(range(7)) + [-1])
    print('\nconfusion matrix (rows=true, cols=pred; last col = unparseable):')
    header = '           ' + ''.join(f'{ID2EMO[c][:5]:>6s}' for c in range(7)) + '  unpar'
    print(header)
    for c in range(7):
        row = '  '.join(f'{cm[c, j]:6d}' for j in range(7))
        print(f'{ID2EMO[c]:10s} {row} {cm[c, -1]:6d}')


if __name__ == '__main__':
    main()
