"""Generate label-aware facial-expression captions for RAF-DB with a VLM.

The default model is Qwen/Qwen2-VL-2B-Instruct (bf16, eager attention). Each
aligned face image is captioned into a unified sentence of the form::

    The person <facial-action description> This is <emotion>.

where <emotion> is the ground-truth adjective for LABELED samples (e.g.
"This is happy") and a blank placeholder for UNLABELED samples (e.g. "This is  ."),
so the label slot is reserved for later filling / prediction. Captions can then
be embedded (bge-m3) and clustered as a text-side view.

Output: jsonl, one dict per line:
    {"img": <filename>, "label": <0..6>, "is_labeled": bool, "caption": str}
Supports resume (skips already-captioned filenames) and balanced subsampling.
"""
import argparse
import json
import os
import re
import time

import torch

# RAF-DB label id -> emotion noun (used for the VLM label hint).
ID2EMO = {
    0: 'surprise', 1: 'fear', 2: 'disgust', 3: 'happiness',
    4: 'sadness', 5: 'anger', 6: 'neutral',
}

# RAF-DB label id -> emotion adjective (used in the "This is <emotion>." slot).
ID2ADJ = {
    0: 'surprised', 1: 'afraid', 2: 'disgusted', 3: 'happy',
    4: 'sad', 5: 'angry', 6: 'neutral',
}

# Rich captioning prompt: forces a uniform "The person" subject and enumerates
# the facial regions we care about, which keeps descriptions specific and
# reduces the vague "neutral" captions seen with a weaker prompt.
BASE_PROMPT = (
    "Describe the facial expression of the person in this image. "
    "Start with the exact phrase 'The person'. "
    "In one or two specific sentences, describe: "
    "(1) the eyebrows (raised / furrowed / relaxed), "
    "(2) the eyes (wide / narrowed / open / closed), "
    "(3) the mouth (smiling / frowning / open / pressed together), and "
    "(4) the overall emotion. "
    "Be precise and vivid; avoid vague wording such as 'neutral to slightly' "
    "unless the face truly shows no strong expression."
)


def build_prompt(emotion, is_labeled):
    """Return the VLM prompt, injecting the ground-truth emotion for labeled
    samples so the description is consistent with the label (weakly supervised)."""
    if is_labeled:
        return (BASE_PROMPT + f" The ground-truth emotion is '{emotion}'; "
                "describe the facial signs that match this emotion.")
    return BASE_PROMPT


def finalize_caption(cap, emotion, is_labeled, blank=''):
    """Normalize the VLM output to: 'The person <desc>. This is <slot>.'"""
    cap = cap.strip().rstrip('.')
    if not cap.lower().startswith('the person'):
        cap = re.sub(
            r'^the\s+(?:facial\s+expression|expression|image|child|baby|man|woman)\b\s*',
            '', cap, flags=re.IGNORECASE)
        cap = 'The person ' + cap.lstrip()
    slot = emotion if is_labeled else blank
    return f'{cap}. This is {slot}.'


def finalize_desc(cap):
    """Normalize the VLM output to a bare 'The person <desc>.' (no label slot)."""
    cap = cap.strip().rstrip('.')
    if not cap.lower().startswith('the person'):
        cap = re.sub(
            r'^the\s+(?:facial\s+expression|expression|image|child|baby|man|woman)\b\s*',
            '', cap, flags=re.IGNORECASE)
        cap = 'The person ' + cap.lstrip()
    return cap


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='/root/autodl-tmp/DebGCD-expclip/RAF-DB/RAF-DB_train')
    p.add_argument('--label_file', default=None,
                   help='default: <root>/EmoLabel/train_label.txt')
    p.add_argument('--out', default='dev_outputs/rafdb_captions.jsonl')
    p.add_argument('--model', default='Qwen/Qwen2-VL-2B-Instruct')
    p.add_argument('--unknown_classes', type=str, default='',
                   help='comma-separated RAF-DB ids treated as UNLABELED '
                        '(e.g. "3"); these get a blank "This is  ." slot. '
                        'All other classes are labeled and get "This is <emotion>."')
    p.add_argument('--blank', default='',
                   help='placeholder text for the unlabeled emotion slot '
                        '(default empty -> "This is  .")')
    p.add_argument('--max_per_class', type=int, default=200,
                   help='balanced subsample per class; 0 = all')
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--max-new-tokens', type=int, default=96)
    p.add_argument('--desc_only', action='store_true', default=False,
                   help='generate label-agnostic descriptions only: output '
                        '{"img", "desc"} with no "This is <emotion>." slot and no '
                        'ground-truth emotion hint (reusable across splits).')
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


def load_samples(root, label_file, max_per_class, seed):
    label_file = label_file or os.path.join(root, 'EmoLabel', 'train_label.txt')
    img_dir = os.path.join(root, 'Image', 'aligned')
    samples = []
    with open(label_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            img, lab = line.split()
            samples.append((img, int(lab)))

    if max_per_class and max_per_class > 0:
        import random
        rng = random.Random(seed)
        by_class = {}
        for img, lab in samples:
            by_class.setdefault(lab, []).append((img, lab))
        kept = []
        for lab in sorted(by_class):
            pool = by_class[lab]
            rng.shuffle(pool)
            kept.extend(pool[:max_per_class])
        # keep original order roughly by filename
        kept.sort(key=lambda x: x[0])
        samples = kept

    return samples, img_dir


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)

    unknown_set = {int(x) for x in args.unknown_classes.split(',') if x.strip() != ''}
    print(f'[caption] unknown/unlabeled classes: '
          f'{sorted(unknown_set) if unknown_set else "none (all labeled)"}')

    samples, img_dir = load_samples(args.root, args.label_file,
                                    args.max_per_class, args.seed)
    print(f'[caption] {len(samples)} images to caption (max_per_class={args.max_per_class})')

    # resume: skip already-done filenames
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
    todo = [(i, l) for i, l in samples if i not in done]
    print(f'[caption] {len(done)} already done, {len(todo)} to process')

    if not todo:
        print('[caption] nothing to do.')
        return

    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation='eager',
    ).to(device).eval()
    processor = AutoProcessor.from_pretrained(args.model)

    out_f = open(args.out, 'a')
    bs = args.batch_size
    start = time.time()
    n = 0
    i = 0
    while i < len(todo):
        batch = todo[i:i + bs]
        msgs = []
        paths = []
        for img, lab in batch:
            p = os.path.join(img_dir, img)
            is_labeled = lab not in unknown_set
            prompt = BASE_PROMPT if args.desc_only else build_prompt(ID2EMO[lab], is_labeled)
            msgs.append([{'role': 'user', 'content': [
                {'type': 'image', 'image': p},
                {'type': 'text', 'text': prompt},
            ]}])
            paths.append((img, lab, is_labeled))

        texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                 for m in msgs]
        image_inputs, video_inputs = process_vision_info(msgs)
        inputs = processor(text=texts, images=image_inputs, videos=video_inputs,
                           padding=True, return_tensors='pt').to(device)
        with torch.no_grad():
            gids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
        gids = gids[:, inputs.input_ids.shape[1]:]
        captions = processor.batch_decode(gids, skip_special_tokens=True)

        for (img, lab, is_labeled), cap in zip(paths, captions):
            if args.desc_only:
                out_f.write(json.dumps({'img': img, 'desc': finalize_desc(cap)}) + '\n')
            else:
                cap = finalize_caption(cap, ID2ADJ[lab], is_labeled, args.blank)
                out_f.write(json.dumps({'img': img, 'label': lab,
                                        'is_labeled': is_labeled, 'caption': cap}) + '\n')
            n += 1
        out_f.flush()
        i += bs

        if n % 20 == 0 or i >= len(todo):
            rate = n / max(time.time() - start, 1e-6)
            eta = (len(todo) - i) / max(rate, 1e-6)
            print(f'[caption] {n}/{len(todo)} done, {rate:.2f} img/s, ETA {eta/60:.1f} min')

    out_f.close()
    print(f'[caption] finished. wrote {n} captions to {args.out}')


if __name__ == '__main__':
    main()
