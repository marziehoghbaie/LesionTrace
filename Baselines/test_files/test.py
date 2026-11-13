import inspect
import os
import sys
import re

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    balanced_accuracy_score
)
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import TSNE
import umap
from utils import utils
from utils.rps import *

# ------------------------------
# Path setup for local imports
# ------------------------------
utils.add_project_paths(3)


# ------------------------------
# Evaluation & Scoring
# ------------------------------
def specificity(y_true, y_pred, eps=1e-6):
    """Calculate average specificity for multi-class classification."""
    cm = confusion_matrix(y_true, y_pred)
    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)
    return (TN / (TN + FP + eps)).mean()


def scoring(y_true, y_pred):
    """Compute F1, Matthews Correlation, and Specificity."""
    return {
        "F1_score": f1_score(y_true, y_pred, average="macro"),
        "Rk-correlation": matthews_corrcoef(y_true, y_pred),
        "Specificity": specificity(y_true, y_pred),
    }


# ------------------------------
# Save Results
# ------------------------------
# ------------------------------
# Save Results
# ------------------------------
def save_results(
        y_probs, y_pred, y_true, case_ids, save_path, embeds, classes,
        postfix="test_complete", logger=None
):
    """Save predictions, confusion matrices, classification report, and embeddings."""
    os.makedirs(save_path, exist_ok=True)

    # --- Predictions CSV ---
    results_df = pd.DataFrame({
        "case_id": case_ids,
        "y_probs": list(y_probs),
        "y_pred": y_pred,
        "y_true": y_true
    })
    results_df.to_csv(os.path.join(save_path, f"results.csv"), index=False)

    # --- Confusion Matrices ---
    cm = confusion_matrix(y_true, y_pred)
    cm_percent = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    # 1️⃣ Confusion Matrix (Counts)
    fig1, ax1 = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=ax1)
    ax1.set_title("Confusion Matrix (Counts)")
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"confusion_matrix_counts_{postfix}.png"), dpi=300)
    plt.close(fig1)

    # 2️⃣ Confusion Matrix (Percentages)
    fig2, ax2 = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm_percent, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=ax2)
    ax2.set_title("Confusion Matrix (%)")
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"confusion_matrix_percent_{postfix}.png"), dpi=300)
    plt.close(fig2)

    # # --- Classification Report ---
    # cls_report = metrics.classification_report(
    #     y_true, y_pred, target_names=classes, output_dict=True
    # )
    # if logger:
    #     logger.info("[INFO] Classification report:")
    #     for k, v in cls_report.items():
    #         logger.info(f"{k}: {v}")

    # --- Additional metrics ---
    scores = scoring(y_true, y_pred)
    if logger:
        logger.info("[INFO] Scoring metrics:")
        logger.info(scores)

    # --- Save embeddings ---
    np.savez(
        os.path.join(save_path, "Embeddings.npz"),
        embeds=np.array(embeds),
        labels=np.array(y_true),
        case_ids=np.array(case_ids)
    )


# ------------------------------
# Test / Inference
# ------------------------------
def test(loader, model, loss_fn, logger, late_fusion=False, complete=False, save_path=None, train_config=None, model_stats=True):
    """Run inference and optionally save results."""
    model.eval()
    device = next(model.parameters()).device

    y_pred, y_true, y_probs, embeds, case_ids = [], [], [], [], []
    total_loss, total_samples = 0.0, 0

    with torch.no_grad():
        for idx_test, batch_test in enumerate(loader):
            if idx_test % 1000 == 0:
                logger.info(f"[INFO] ✅ Test {idx_test}/{len(loader)}")
            labels = batch_test["label"].to(device, non_blocking=True)
            if len(labels) == 1 and labels[0] == -1:
                continue
            A = batch_test["A"].to(device, non_blocking=True)
            B = batch_test["B"].to(device, non_blocking=True)

            if model_stats:
                stats_dict = utils.get_model_stats(
                    model,
                    A,
                    B
                )

                logger.info('Model type: {}, # Parameters: {} M(illion), FLOPS: {} G'.format(train_config['model_type'], stats_dict['params_m'], stats_dict['flops_g']))
                return

            outputs = model(A, B)

            logits = outputs["logits"]
            probs = F.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)

            embeds.extend(outputs["embed"].cpu().numpy())
            y_probs.extend(probs.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_true.extend(labels.cpu().numpy())
            case_ids.extend(batch_test["name"])

            if not complete:
                total_loss += loss_fn(logits, labels).item() * labels.size(0)
            total_samples += labels.size(0)

    avg_loss = total_loss / max(total_samples, 1)
    bacc = balanced_accuracy_score(y_true, y_pred)
    scores = scoring(y_true, y_pred)

    logger.info(f"[INFO] Test Results — BACC: {bacc:.4f}, Loss: {avg_loss:.4f}")
    logger.info("[INFO] Detailed metrics:")
    logger.info(scores)

    if complete and save_path:
        save_results(y_probs, y_pred, y_true, case_ids, save_path, embeds, classes=list(loader.dataset.classes.keys()),
                     logger=logger)
    rps_metric = sa_rps_metric(torch.tensor(y_probs), torch.tensor(y_true), len(loader.dataset.classes))
    logger.info(f'[INFO] RPS for ordanility measurement: {rps_metric}')

    return {"F1_score": scores["F1_score"], "bacc": bacc, "loss": avg_loss, 'rps_metric': rps_metric}


# ------------------------------
# t-SNE & UMAP Visualizations
# ------------------------------
def plot_tsne(features, labels, perplexity=30, n_iter=1000, save_path=None, n_components=2):
    """Compute and save t-SNE embeddings."""
    if isinstance(features, torch.Tensor):
        features = features.detach().cpu().numpy()

    tsne = TSNE(n_components=n_components, perplexity=perplexity, n_iter=n_iter, init='random', random_state=42)
    tsne_result = tsne.fit_transform(features)

    df = pd.DataFrame(tsne_result, columns=[f'tsne{i + 1}' for i in range(n_components)])
    df['label'] = labels
    if save_path:
        df.to_csv(os.path.join(save_path, 'tsne_embeddings.csv'), index=False)


def tsne_vis(path, read_name, label4legend=None):
    df = pd.read_csv(os.path.join(path, f'{read_name}.csv'))
    plt.figure(figsize=(8, 6))
    str_labels = df['label'].astype(str)

    label2idx = {label: idx for idx, label in enumerate(sorted(set(str_labels)))}
    idx2label = {v: k for k, v in label2idx.items()}
    y = np.array([label2idx[l] for l in str_labels])

    scatter = plt.scatter(df['tsne1'], df['tsne2'], c=y, cmap='tab10', alpha=0.7)

    handles, legend_labels = scatter.legend_elements()
    mapped_labels = [idx2label[int(re.search(r'\d+', lbl).group())] for lbl in legend_labels if re.search(r'\d+', lbl)]
    plt.legend(handles, mapped_labels, title="Classes")
    plt.title('t-SNE Visualization')
    plt.xlabel('t-SNE Dim 1')
    plt.ylabel('t-SNE Dim 2')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(path, 'tsne2D.png'), dpi=300)


def tsne_plot(t_npy_path):
    """Generate t-SNE plots from saved embeddings."""
    embedding_file = np.load(os.path.join(t_npy_path, 'Embeddings.npz'), allow_pickle=True)
    embeddings, labels = embedding_file['embeds'], embedding_file['labels']
    assert len(labels) == len(embeddings)

    plot_tsne(embeddings, labels, save_path=t_npy_path)
    tsne_vis(t_npy_path, 'tsne_embeddings')


def plot_umap(t_npy_path, n_neighbors=15, min_dist=0.1, n_components=2, metric='cosine', title='UMAP Projection'):
    """Generate UMAP projection plot."""
    embedding_file = np.load(os.path.join(t_npy_path, 'Embeddings.npz'), allow_pickle=True)
    embeddings, labels = embedding_file['embeds'], embedding_file['labels']

    if not np.issubdtype(np.array(labels).dtype, np.number):
        label_encoder = LabelEncoder()
        labels_encoded = label_encoder.fit_transform(labels)
        label_names = label_encoder.classes_
    else:
        labels_encoded = labels
        label_names = np.unique(labels)

    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, metric=metric,
                        random_state=42)
    embedding_umap = reducer.fit_transform(embeddings)

    if n_components == 2:
        plt.figure(figsize=(8, 6))
        palette = sns.color_palette("hls", len(np.unique(labels_encoded)))
        sns.scatterplot(x=embedding_umap[:, 0], y=embedding_umap[:, 1], hue=labels_encoded,
                        palette=palette, legend='full', alpha=0.8, s=40)
        plt.title(title)
        plt.xlabel('UMAP-1')
        plt.ylabel('UMAP-2')
        plt.legend(title='Classes', labels=label_names, bbox_to_anchor=(1.05, 1), loc='upper left')
    elif n_components == 3:
        # from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(embedding_umap[:, 0], embedding_umap[:, 1], embedding_umap[:, 2], c=labels_encoded,
                   cmap='Spectral', s=30, alpha=0.8)
        ax.set_title(title)
        ax.set_xlabel('UMAP-1')
        ax.set_ylabel('UMAP-2')
        ax.set_zlabel('UMAP-3')
    else:
        raise ValueError("n_components must be 2 or 3")

    plt.savefig(os.path.join(t_npy_path, 'umap_projection.png'), dpi=150, bbox_inches='tight')
    return embedding_umap
