"""Text prototype generation + automatic class-ID alignment for RAF-DB.

RAF-DB original labels (7 emotions):
    0=Surprise, 1=Fear, 2=Disgust, 3=Happiness, 4=Sadness, 5=Anger, 6=Neutral

Exp-CLIP stores prompts in its own order (Happiness, Sadness, Neutral, Anger,
Surprise, Disgust, Fear). We deliberately do **not** use that order. Instead we
map a RAF-DB id -> emotion name -> prompt string -> CLIP text feature, which
keeps the user's original class numbering intact. Setting ``--unknown_classes 3``
still means "Happiness is unknown", no matter how Exp-CLIP internally orders its
prompts.

Only the KNOWN classes get a text prototype; the unknown class (if any) keeps its
randomly-initialized cluster center, so no unknown-class text prior leaks in.
"""
import torch

# RAF-DB label id -> lowercase emotion name (used to build the prompt string).
RAFDB_ID_TO_EMOTION = {
    0: 'surprise',
    1: 'fear',
    2: 'disgust',
    3: 'happiness',
    4: 'sadness',
    5: 'anger',
    6: 'neutral',
}

# Rich, LLM-style facial-action descriptions per RAF-DB emotion id.
# These mimic what a "facial action describer" LLM (e.g. BLIP2-T5 with the
# Exp-CLIP instruction) would output, giving CLIP a much richer text prior than
# the single "an expression of {}." template. Each id maps to several
# paraphrases; the text prototype for a class is the mean of their L2-normalized
# CLIP text features (see build_llm_text_prototypes).
RAFDB_ID_TO_LLM_DESCRIPTIONS = {
    0: [
        'eyebrows are raised high and curved, eyes are wide open showing the whites, and the mouth is dropped open',
        'the forehead shows horizontal wrinkles, the upper eyelids are lifted, and the jaw is lowered with the lips parted',
        'a startled face with wide eyes, raised brows, and an open mouth forming an o-shape',
        'the person looks astonished, with both eyebrows lifted, eyelids retracted, and lips slightly parted',
        'eyes opened wider than normal, brows pulled up toward the hairline, and mouth gaping open',
        'a gasp-like expression with elevated eyebrows, visible upper eyelids, and a slack open jaw',
    ],
    1: [
        'eyebrows are raised and drawn together, the upper eyelids are lifted, and the lips are stretched horizontally back toward the ears',
        'eyes are tense and widened, brows pulled up in the center, and the mouth is open with the corners pulled straight back',
        'a frightened face with tensed eyelids, raised inner brows, and lips stretched thin across the face',
        'the person looks afraid, with wide staring eyes, forehead pulled upward, and mouth corners retracted',
        'upper eyelids are raised exposing the whites above the iris, and the lips are pulled back tightly',
        'an alarmed expression with raised and drawn-together brows, taut lower eyelids, and a horizontally stretched mouth',
    ],
    2: [
        'the nose is wrinkled, the upper lip is raised, and the cheeks are lifted as if smelling something unpleasant',
        'eyebrows are lowered, the skin between the brows is furrowed, and the upper lip is curled upward',
        'a grimacing face with a wrinkled nose, narrowed eyes, and the corners of the mouth pulled down',
        'the person looks repulsed, with the upper lip lifted and the nose scrunched',
        'nostrils flare slightly, the upper lip rises, and the lower lip pushes up',
        'an expression of aversion with a wrinkled nose, a raised upper lip, and slightly narrowed eyes',
    ],
    3: [
        'the corners of the lips are pulled up and back, the cheeks are raised, and crows feet wrinkles appear around the eyes',
        'a bright smile with lifted lip corners, raised cheeks, and eyes that crinkle at the corners',
        'the person is smiling with teeth showing, cheeks pushed up, and eyes slightly narrowed',
        'a cheerful face with upward-turned mouth corners, prominent cheeks, and wrinkles at the outer corners of the eyes',
        'the mouth is curved upward into a smile, the cheek muscles are engaged, and the lower eyelids tighten',
        'a joyful expression with lips stretched wide, cheeks bunched up, and eyes crinkling',
    ],
    4: [
        'the inner corners of the eyebrows are raised, the corners of the mouth are pulled down, and the eyes look downward',
        'the upper eyelids are drooping, the lips are pouting, and the chin is pulled up',
        'a sorrowful face with downturned mouth corners, raised inner brows, and a lowered gaze',
        'the person looks sad, with drooping eyelids, a frown, and the inner eyebrows tilted upward',
        'mouth corners turn downward, the lower lip pouts, and the eyebrows slant upward in the middle',
        'an expression of grief with downcast eyes, a downward-curved mouth, and raised inner brow tips',
    ],
    5: [
        'the eyebrows are lowered and drawn together, the eyes are narrowed, and the lips are pressed tightly together',
        'the jaw is clenched, the nostrils are flared, and vertical lines appear between the brows',
        'a furious face with furrowed brows, glaring narrowed eyes, and a tightened mouth',
        'the person looks angry, with the brow ridge pulled down, eyes glaring, and lips thinned',
        'eyebrows slant sharply downward toward the nose, the eyelids are tensed, and the mouth is tight',
        'an enraged expression with a scowl, glaring eyes, and lips pressed into a thin line',
    ],
    6: [
        'the face is relaxed with no strong muscular activity, the mouth is closed and the lips are gently together',
        'the facial muscles are at rest, the eyes are open normally, and the mouth is neutral',
        'a calm expressionless face with relaxed brows, a straight mouth, and steady eyes',
        'the person shows no emotion, with the face muscles relaxed and the lips lightly closed',
        'the brows are level, the eyelids are relaxed, and the mouth is neither smiling nor frowning',
        'a neutral face with a relaxed forehead, calm open eyes, and a closed mouth',
    ],
}

# Matches Exp-CLIP's default zero-shot prompt ("an_expression_of_name").
DEFAULT_TEMPLATE = 'an expression of {}.'


def known_class_prompts(known_classes, template=DEFAULT_TEMPLATE):
    """Build prompts for known classes, aligned with ``known_classes`` order.

    Args:
        known_classes: original RAF-DB ids (e.g. [0, 1, 2, 4, 5, 6]).
        template: prompt template with a single ``{}`` slot for the emotion name.

    Returns:
        list[str] of prompts, one per known class, in the same order.
    """
    return [template.format(RAFDB_ID_TO_EMOTION[c]) for c in known_classes]


def build_llm_text_prototypes(backbone, known_classes):
    """Build per-class text prototypes from rich LLM-style descriptions.

    Every description in ``RAFDB_ID_TO_LLM_DESCRIPTIONS`` is encoded with the
    frozen CLIP text encoder; the L2-normalized features of each class are then
    mean-pooled and re-normalized into a single prototype per known class.

    Args:
        backbone: an ``ExpCLIPBackbone`` (provides ``encode_text``).
        known_classes: original RAF-DB ids.

    Returns:
        tensor of shape [num_known, feat_dim], L2-normalized, ordered to match
        the remapped known-class indices 0..num_known-1.
    """
    prototypes = []
    for c in known_classes:
        feats = backbone.encode_text(RAFDB_ID_TO_LLM_DESCRIPTIONS[c])  # [n_desc, dim]
        proto = feats.mean(dim=0)
        proto = proto / proto.norm(dim=-1, keepdim=True)
        prototypes.append(proto)
    return torch.stack(prototypes, dim=0)


def build_text_prototypes(backbone, known_classes, template=DEFAULT_TEMPLATE, style='single'):
    """Encode known-class prompts/descriptions into L2-normalized text prototypes.

    Args:
        backbone: an ``ExpCLIPBackbone`` (provides ``encode_text``).
        known_classes: original RAF-DB ids.
        template: prompt template (used when ``style='single'``).
        style: 'single' (one template prompt per class) or 'llm' (rich
            LLM-style descriptions, mean-pooled per class).

    Returns:
        tensor of shape [num_known, feat_dim], L2-normalized, ordered to match
        the remapped known-class indices 0..num_known-1.
    """
    if style == 'llm':
        return build_llm_text_prototypes(backbone, known_classes)
    prompts = known_class_prompts(known_classes, template)
    return backbone.encode_text(prompts)


def inject_text_prototypes(projector, text_prototypes):
    """Initialize the known-class rows of the clustering head with text prototypes.

    DebGCD's clustering head is ``last_layer = weight_norm(Linear(feat_dim, C))``
    with ``weight_g`` frozen to 1, so the effective weight is ``weight_v / ||weight_v||``.
    Because the prototypes are already unit-norm, copying them into ``weight_v``
    makes ``weight`` equal to the prototypes directly.

    Only rows [0, num_known) are overwritten; the unknown-class row (index >=
    num_known) keeps its random init. Training is allowed to keep updating the
    centers afterwards (the prototypes act as initialization).
    """
    num_known = text_prototypes.shape[0]
    w_v = projector.last_layer.weight_v
    with torch.no_grad():
        w_v.data[:num_known] = text_prototypes.to(w_v.device, w_v.dtype)
