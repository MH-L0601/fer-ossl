"""Embed captions with bge-m3, then visualize/quantify text-side clustering.

Loads the caption jsonl produced by caption_rafdb.py, encodes with BAAI/bge-m3,
reduces to 2D with t-SNE, and produces:
  - t-SNE colored by TRUE emotion label
  - t-SNE colored by KMeans(k=7) cluster assignment
Also prints a nearest-centroid 7-way accuracy to quantify separability.
"""
import argparse
import json
import os

import numpy as np

ID2EMO = {
    0: 'surprise', 1: 'fear', 2: 'disgust', 3: 'happiness',
    4: 'sadness', 5: 'anger', 6: 'neutral',
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--captions', default='dev_outputs/rafdb_captions.labelaware.jsonl')
    p.add_argument('--out_dir', default='dev_outputs')
    p.add_argument('--encoder', default='BAAI/bge-m3')
    p.add_argument('--k', type=int, default=7)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--max_points', type=int, default=0,
                   help='cap points for plotting only; 0 = all')
    return p.parse_args()


def main():
    args = parse_args()
    imgs, labels, captions = [], [], []
    with open(args.captions, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            imgs.append(d['img'])
            labels.append(int(d['label']))
            captions.append(d['caption'])

    labels = np.array(labels)
    print(f'[probe] {len(captions)} captions loaded, '
          f'class dist: {dict(zip(*np.unique(labels, return_counts=True)))}')

    # ---- encode ----
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer(args.encoder)
    feats = enc.encode(captions, normalize_embeddings=True, show_progress_bar=True,
                       batch_size=64, convert_to_numpy=True)
    print(f'[probe] text features shape: {feats.shape}')

    # ---- nearest-centroid 7-way accuracy (quantify separability) ----
    from sklearn.metrics import accuracy_score
    uniq = np.unique(labels)
    centers = np.stack([feats[labels == c].mean(axis=0) for c in uniq])
    centers = centers / np.linalg.norm(centers, axis=1, keepdims=True)
    sim = feats @ centers.T
    pred = uniq[np.argmax(sim, axis=1)]
    acc = accuracy_score(labels, pred)
    print(f'[probe] nearest-centroid {len(uniq)}-way accuracy: {acc:.4f}')

    # ---- KMeans ----
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=args.k, random_state=args.seed, n_init=10).fit(feats)
    print(f'[probe] KMeans(k={args.k}) inertia={km.inertia_:.1f}')

    # ---- t-SNE ----
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, random_state=args.seed, perplexity=min(30, len(feats) - 1))
    emb = tsne.fit_transform(feats)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # colormap for 7 emotions
    cmap = plt.cm.get_cmap('tab10', 10)

    def plot(colors, title, fname):
        plt.figure(figsize=(9, 7))
        order = sorted(uniq)
        for c in order:
            m = labels == c
            plt.scatter(emb[m, 0], emb[m, 1], s=8, alpha=0.7,
                        color=cmap(c % 10), label=ID2EMO[c])
        plt.legend(markerscale=3, fontsize=9)
        plt.title(title)
        plt.xticks([]); plt.yticks([])
        plt.tight_layout()
        p = os.path.join(args.out_dir, fname)
        plt.savefig(p, dpi=150)
        print(f'[probe] saved {p}')
        plt.close()

    plot(None, f'Text features (bge-m3) colored by TRUE emotion (acc={acc:.3f})',
         'text_cluster_true_label.png')

    # KMeans coloring
    plt.figure(figsize=(9, 7))
    for c in range(args.k):
        m = km.labels_ == c
        plt.scatter(emb[m, 0], emb[m, 1], s=8, alpha=0.7, label=f'cluster {c}')
    plt.legend(markerscale=3, fontsize=9)
    plt.title(f'Text features (bge-m3) colored by KMeans(k={args.k})')
    plt.xticks([]); plt.yticks([])
    plt.tight_layout()
    p2 = os.path.join(args.out_dir, 'text_cluster_kmeans.png')
    plt.savefig(p2, dpi=150)
    print(f'[probe] saved {p2}')
    plt.close()


if __name__ == '__main__':
    main()
