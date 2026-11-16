"""Dynamic text prototypes from image descriptions + prototype anchor loss.

Captions follow the unified format::

    <image description>. This is <class name>.

where ``<slot>`` is:
  * the ground-truth emotion name for LABELED samples (e.g. "This is sadness."),
  * the GCD pseudo-label name for UNLABELED samples, filled in
    progressively as pseudo-labels are assigned during training.

The description body ("The person ...") is kept fixed; only the slot changes as
pseudo-labels evolve. Each caption is encoded with the frozen CLIP text encoder
(``ExpCLIPBackbone.encode_text``), optionally mapped into the backbone feature
space via a frozen mapper, and mean-pooled per predicted class into an
L2-normalized text prototype.

These prototypes act as *anchors*: the cluster centers (``weight_v`` rows) are
softly pulled toward them through a prototype anchor loss (cosine similarity),
rather than being hard-overwritten. Both known and unknown classes are anchored.
"""
import json

import torch
import torch.nn.functional as F

from expclip.text_prototypes import RAFDB_ID_TO_EMOTION, DEFAULT_TEMPLATE

# The "This is <slot>." slot uses the emotion NAME (noun form, e.g. "happiness"),
# consistent with RAFDB_ID_TO_EMOTION imported above.


def split_caption(caption):
    """Split 'The person <desc>. This is <slot>.' into (desc, slot).

    Returns the description body (trailing period removed) and the slot text
    ('' if blank). Falls back to the whole caption as desc when no 'This is'
    marker is present.
    """
    if 'This is' in caption:
        desc, _, slot = caption.rpartition('This is')
        desc = desc.strip().rstrip('.').strip()
        slot = slot.strip().rstrip('.').strip()
    else:
        desc = caption.strip().rstrip('.').strip()
        slot = ''
    return desc, slot


def load_captions(caption_path):
    """Load label-aware captions into ``{img_filename: (desc, label, is_labeled)}``."""
    captions = {}
    with open(caption_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            desc, _ = split_caption(d['caption'])
            captions[d['img']] = (desc, int(d['label']), bool(d['is_labeled']))
    return captions


def load_desc(desc_path):
    """Load label-agnostic image descriptions into ``{img_filename: desc}``.

    These descriptions are generated with a single image-only prompt (no
    ground-truth emotion hint) and are therefore independent of the
    labeled/unlabeled split and of ``prop_train_labels``. They can be cached and
    reused across every experiment; only the "This is <slot>." slot is filled
    dynamically at training time.
    """
    descs = {}
    with open(desc_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            desc = d.get('desc', d.get('caption', ''))
            if not desc:
                continue
            desc, _ = split_caption(desc)
            descs[d['img']] = desc
    return descs


def build_uqidx_to_caption(train_label_file, captions):
    """Map RAF-DB global ``uq_idx`` -> ``(desc, label, is_labeled)``.

    RAF-DB assigns ``uq_idxs = range(len(samples))`` in the order ``find_classes``
    reads the label file (skipping blank lines), so ``uq_idx`` equals the index of
    the image in the label file. Returns ``(uq_to_caption, num_samples)`` where the
    dict only contains samples that have a caption.
    """
    image_ids = []
    with open(train_label_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            image_ids.append(line.split()[0])

    uq_to_caption = {}
    for uq_idx, img in enumerate(image_ids):
        if img in captions:
            uq_to_caption[uq_idx] = captions[img]
    return uq_to_caption, len(image_ids)


def build_uqidx_to_desc_label(train_label_file, desc_map, labelled_uq_idxs):
    """Map ``uq_idx`` to descriptions and labels for labeled samples only.

    Descriptions are label-agnostic. Ground-truth labels are retained only for
    samples explicitly included in ``labelled_uq_idxs``; unlabeled targets are
    deliberately not exposed to text-prototype construction.
    """
    uq_to_desc = {}
    uq_to_label = {}
    num_samples = 0
    with open(train_label_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            uq_idx = num_samples
            img = parts[0]
            if img in desc_map:
                uq_to_desc[uq_idx] = desc_map[img]
            if uq_idx in labelled_uq_idxs:
                uq_to_label[uq_idx] = int(parts[1])
            num_samples += 1
    return uq_to_desc, uq_to_label, num_samples


class CaptionTextPrototypeBuilder:
    """Build per-known-class text prototypes from captions + pseudo-labels."""

    def __init__(self, text_backbone, mapper, known_classes, unlabeled_classes,
                 labelled_uq_idxs, uq_to_desc, uq_to_label, feat_dim, device):
        """
        Args:
            text_backbone: object exposing ``encode_text(list[str]) -> [N, 4096]``
                (an ``ExpCLIPBackbone``).
            mapper: optional frozen mapper ``4096 -> feat_dim`` (``None`` if the
                text encoder already outputs ``feat_dim``).
            known_classes: original RAF-DB ids of known classes (ordered, remapped
                to ``0..num_known-1``).
            unlabeled_classes: original RAF-DB ids of unknown classes.
            labelled_uq_idxs: set of global ``uq_idx`` that have a real label.
            uq_to_desc: dict ``uq_idx -> desc`` (label-agnostic image description).
            uq_to_label: ground-truth labels for labeled samples only.
            feat_dim: backbone feature dimension the prototypes must live in.
            device: torch device.
        """
        self.text_backbone = text_backbone
        self.mapper = mapper
        self.known_classes = list(known_classes)
        self.labelled_uq_idxs = set(int(u) for u in labelled_uq_idxs)
        self.uq_to_desc = uq_to_desc
        self.uq_to_label = uq_to_label
        self.feat_dim = feat_dim
        self.device = device

        self.num_known = len(self.known_classes)
        self.unknown_classes = list(unlabeled_classes)
        self.num_unknown = len(self.unknown_classes)
        self.num_total = self.num_known + self.num_unknown
        # original RAF-DB id -> remapped known index (0..num_known-1)
        self.remap = {orig: i for i, orig in enumerate(self.known_classes)}
    def _encode(self, texts, batch_size=128):
        """Encode a list of caption texts into L2-normalized ``feat_dim`` features.

        Text is encoded in mini-batches to bound GPU memory (per-class caption
        pooling can accumulate thousands of captions).
        """
        if not texts:
            return torch.empty(0, self.feat_dim, device=self.device)
        feats = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                chunk = texts[i:i + batch_size]
                f = self.text_backbone.encode_text(chunk)  # [n, 4096] L2-normalized
                if self.mapper is not None:
                    f = self.mapper(f)
                feats.append(f.float())
        return torch.cat(feats, dim=0)

    def _fallback_proto(self, c):
        """Single-template fallback prototype for a class with no captioned samples."""
        if c < self.num_known:
            orig_id = self.known_classes[c]
        else:
            orig_id = self.unknown_classes[c - self.num_known]
        text = DEFAULT_TEMPLATE.format(RAFDB_ID_TO_EMOTION[orig_id])
        return self._encode([text])[0]

    def build_prototypes(self, pseudo_labels=None):
        """Build L2-normalized text prototypes for every class (known + unknown).

        Labeled samples use their ground-truth emotion. All unlabeled samples,
        including samples from unknown classes, are included only after they
        receive a valid GCD pseudo-label.

        Args:
            pseudo_labels: dict mapping global ``uq_idx`` -> GCD pseudo-label
                (``0..num_known-1`` = known, ``num_known`` = unknown), or ``None``.

        Returns:
            tensor of shape ``[num_known + num_unknown, feat_dim]``, L2-normalized.
            Rows ``0..num_known-1`` are known classes; rows ``num_known..`` are
            unknown classes.
        """
        num_total = self.num_total
        per_class_texts = {c: [] for c in range(num_total)}

        for uq, desc in self.uq_to_desc.items():
            if uq in self.labelled_uq_idxs:
                # Only labeled samples may contribute their ground-truth class.
                orig_label = self.uq_to_label.get(uq)
                if orig_label is None:
                    continue
                if orig_label not in self.remap:
                    continue
                c = self.remap[orig_label]
                per_class_texts[c].append(f'{desc}. This is {RAFDB_ID_TO_EMOTION[orig_label]}.')
                continue

            # Every unlabeled sample, known or unknown, must have a valid
            # confidence-filtered GCD pseudo-label before it can contribute.
            p = -1 if pseudo_labels is None else int(pseudo_labels.get(uq, -1))
            if p < 0 or p >= self.num_total:
                continue
            if p < self.num_known:
                orig_class = self.known_classes[p]
            else:
                orig_class = self.unknown_classes[p - self.num_known]
            per_class_texts[p].append(
                f'{desc}. This is {RAFDB_ID_TO_EMOTION[orig_class]}.')

        prototypes = []
        for c in range(num_total):
            texts = per_class_texts[c]
            if texts:
                feats = self._encode(texts)                 # [n, feat_dim]
                proto = feats.mean(dim=0)
                proto = proto / (proto.norm(dim=-1, keepdim=True) + 1e-8)
            else:
                proto = self._fallback_proto(c)
            prototypes.append(proto)

        return torch.stack(prototypes, dim=0).to(self.device)


def prototype_anchor_loss(weight, prototypes):
    """Cosine-similarity anchor loss between cluster centers and text prototypes.

    All rows (known + unknown) are anchored to their text prototype.

    Args:
        weight: clustering head weight rows ``[num_classes, feat_dim]``
            (e.g. ``projector.last_layer.weight_v``).
        prototypes: L2-normalized text prototypes ``[num_classes, feat_dim]``.

    Returns:
        scalar ``1 - mean(cos_sim(weight_i, prototype_i))`` in ``[0, 2]``.
    """
    w = F.normalize(weight.float(), dim=-1)
    p = F.normalize(prototypes.float(), dim=-1)
    cos = (w * p).sum(dim=-1)          # [num_known]
    return 1.0 - cos.mean()
