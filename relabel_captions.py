"""Post-process existing captions into the unified label-aware format.

Rewrites each caption produced by caption_rafdb.py into::

    The person <facial-action description> This is <emotion>.

where LABELED samples get the ground-truth adjective (e.g. "This is happy") and
UNLABELED samples get a blank slot ("This is  ."). Uses the same finalize logic
as caption_rafdb.py, so you can convert an existing caption file without
re-running the VLM.

Usage:
    python relabel_captions.py \
        --captions dev_outputs/rafdb_captions.jsonl \
        --out dev_outputs/rafdb_captions.labelaware.jsonl \
        --unknown_classes 3
"""
import argparse
import json
import re

# RAF-DB label id -> emotion adjective (mirrors caption_rafdb.ID2ADJ).
ID2ADJ = {
    0: 'surprised', 1: 'afraid', 2: 'disgusted', 3: 'happy',
    4: 'sad', 5: 'angry', 6: 'neutral',
}


def finalize_caption(cap, emotion, is_labeled, blank=''):
    """Normalize to: 'The person <desc>. This is <slot>.'"""
    cap = cap.strip().rstrip('.')
    if not cap.lower().startswith('the person'):
        cap = re.sub(
            r'^the\s+(?:facial\s+expression|expression|image|child|baby|man|woman)\b\s*',
            '', cap, flags=re.IGNORECASE)
        cap = 'The person ' + cap.lstrip()
    slot = emotion if is_labeled else blank
    return f'{cap}. This is {slot}.'


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--captions', default='dev_outputs/rafdb_captions.jsonl')
    p.add_argument('--out', default='dev_outputs/rafdb_captions.labelaware.jsonl')
    p.add_argument('--unknown_classes', type=str, default='',
                   help='comma-separated RAF-DB ids treated as unlabeled (e.g. "3")')
    p.add_argument('--blank', default='',
                   help='placeholder for the unlabeled emotion slot '
                        '(default empty -> "This is  .")')
    return p.parse_args()


def main():
    args = parse_args()
    unknown = {int(x) for x in args.unknown_classes.split(',') if x.strip()}
    with open(args.captions, 'r') as f_in, open(args.out, 'w') as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            lab = int(d['label'])
            is_labeled = lab not in unknown
            d['is_labeled'] = is_labeled
            d['caption'] = finalize_caption(d['caption'], ID2ADJ[lab],
                                            is_labeled, args.blank)
            f_out.write(json.dumps(d) + '\n')
    print(f'[relabel] wrote label-aware captions to {args.out} '
          f'(unknown classes = {sorted(unknown) or "none"})')


if __name__ == '__main__':
    main()
