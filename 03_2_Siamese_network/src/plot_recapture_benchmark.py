#!/usr/bin/env python3


import os
import json
import unicodedata
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent

pred_dir = project_root / 'results' / 'predictions'
manifest_path = project_root / '03_2_Siamese_network' / 'data' / 'test' / 'test_manifest.json'

with open(manifest_path, 'r', encoding='utf-8') as f:
    manifest = json.load(f)

toad_clusters = manifest.get('clusters', {})
img_to_toad = {}
for t_id, imgs in toad_clusters.items():
    for img in imgs:
        img_to_toad[unicodedata.normalize('NFC', img)] = t_id


def evaluate_predictions(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    preds = data.get('predictions', data)
    test_files = list(preds.keys())

    seen = set()
    ranks = []
    for q in test_files:
        norm_q = unicodedata.normalize('NFC', q)
        t_id = img_to_toad.get(norm_q)
        if t_id in seen:
            cands = preds.get(q, [])
            m_rank = 999
            for c in cands:
                c_name = unicodedata.normalize('NFC', c.get('candidate') or c.get('candidate_image'))
                if img_to_toad.get(c_name) == t_id:
                    m_rank = c['rank']
                    break
            ranks.append(m_rank)
        seen.add(t_id)

    n_recaps = len(ranks)
    cmc = [sum(1 for r in ranks if r <= k) / n_recaps * 100 for k in range(1, 21)]
    valid_ranks = [r for r in ranks if r <= 20]
    mean_rank = float(np.mean(valid_ranks)) if valid_ranks else 20.0
    return {
        'cmc': cmc,
        'mean_rank': mean_rank,
        'ranks': ranks,
        'top1': cmc[0],
        'top5': cmc[4],
        'top20': cmc[19]
    }


def generate_recapture_benchmark_plot():
    # Load 4 models comparing Throat vs Throat+Belly on Bi-Model and Siamese
    models = {
        'BiModel_ThroatBelly': {
            'label': 'Bi-Model Consensus [Throat+Belly] (Top-1: 96.7%)',
            'display': 'Bi-Model\n(Throat+Belly)',
            'file': pred_dir / 'consensus' / 'consensus_predictions.json',
            'color': '#059669',  # Emerald Green
            'marker': 's',
            'linestyle': '-'
        },
        'Siamese_Throat': {
            'label': 'Siamese ConvNeXt RGB [Throat Only] (Top-1: 71.7%)',
            'display': 'Siamese RGB\n(Throat Only)',
            'file': pred_dir / 'siamese' / 'siamese_predictions.json',
            'color': '#2563eb',  # Royal Blue
            'marker': 'o',
            'linestyle': '-'
        },
        'Siamese_ThroatBelly': {
            'label': 'Siamese ConvNeXt RGB [Throat+Belly] (Top-1: 61.7%)',
            'display': 'Siamese RGB\n(Throat+Belly)',
            'file': pred_dir / 'siamese' / 'siamese_rgb_throat_belly_predictions.json',
            'color': '#8b5cf6',  # Violet / Purple
            'marker': 'v',
            'linestyle': '-'
        },
        'BiModel_Throat': {
            'label': 'Bi-Model Consensus [Throat Only] (Top-1: 41.7%)',
            'display': 'Bi-Model\n(Throat Only)',
            'file': pred_dir / 'consensus' / 'consensus_throat_predictions.json',
            'color': '#dc2626',  # Crimson Red
            'marker': 'x',
            'linestyle': '-'
        }
    }

    results = {}
    for key, cfg in models.items():
        if cfg['file'].exists():
            results[key] = evaluate_predictions(cfg['file'])
        else:
            raise FileNotFoundError(f"Missing predictions file: {cfg['file']}")

    # -------------------------------------------------------------
    # Plotting: 2-Panel Figure
    # -------------------------------------------------------------
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5), dpi=300)
    fig.suptitle("Anatomical Recapture Identification Benchmark: Throat vs. Throat+Belly", fontsize=15, fontweight='bold', y=0.98)

    ranks_x = np.arange(1, 21)

    # -------------------------------------------------------------
    # Panel A: Cumulative Match Characteristic (CMC) Curves
    # -------------------------------------------------------------
    for key, cfg in models.items():
        res = results[key]
        ax1.plot(
            ranks_x, res['cmc'],
            marker=cfg['marker'],
            linewidth=2.5,
            markersize=6,
            color=cfg['color'],
            linestyle=cfg['linestyle'],
            label=cfg['label']
        )

    ax1.set_title("A: Cumulative Match Characteristic (CMC) Curves (Rank 1 to 20)", fontsize=12.5, fontweight='bold', pad=10)
    ax1.set_xlabel("Rank k", fontsize=11, fontweight='semibold')
    ax1.set_ylabel("Identification Accuracy (%)", fontsize=11, fontweight='semibold')
    ax1.set_xlim(0.5, 20.5)
    ax1.set_ylim(35, 103)
    ax1.set_xticks([1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 20])
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc="lower right", frameon=True, facecolor='white', framealpha=0.95, fontsize=9.5)

    # -------------------------------------------------------------
    # Panel B (formerly Panel C): Mean Recapture Rank
    # -------------------------------------------------------------
    model_keys = list(models.keys())
    model_labels = [models[k]['display'] for k in model_keys]
    mean_ranks = [results[k]['mean_rank'] for k in model_keys]
    colors = [models[k]['color'] for k in model_keys]

    y_pos = np.arange(len(model_keys))
    bars = ax2.barh(y_pos, mean_ranks, color=colors, alpha=0.88, edgecolor='black', linewidth=0.8, height=0.55)

    for bar, val in zip(bars, mean_ranks):
        ax2.text(
            val + 0.1, bar.get_y() + bar.get_height() / 2,
            f"Rank: {val:.2f}",
            va='center', ha='left',
            fontsize=10, fontweight='bold', color='#1e293b'
        )

    ax2.set_title("B: Mean Recapture Rank (Lower is Better)", fontsize=12.5, fontweight='bold', pad=10)
    ax2.set_xlabel("Mean Rank within Top-20", fontsize=11, fontweight='semibold')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(model_labels, fontsize=10.5, fontweight='semibold')
    ax2.set_xlim(0, 5.8)
    ax2.invert_yaxis()  # Best on top
    ax2.grid(True, axis='x', linestyle='--', alpha=0.5)

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])

    out_p = project_root / 'results' / 'plots' / 'recapture_benchmark_comparison_all_models.png'
    out_p.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(out_p, dpi=300)
    plt.close()

    print(f"✅ Saved updated 2-panel plot to:\n   - {out_p}")


if __name__ == '__main__':
    generate_recapture_benchmark_plot()
