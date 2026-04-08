"""
match_faces.py
--------------
Compares selfie embeddings against ID embeddings using cosine similarity.
Evaluates accuracy with ROC-AUC, EER, optimal threshold, and classification metrics.
"""

import sys
import os
import numpy as np
from itertools import product
from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score, roc_curve, classification_report,
    accuracy_score, precision_recall_fscore_support, confusion_matrix
)

# Force UTF-8 output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from setup_database import get_connection


def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def load_embeddings():
    """Load all embeddings from MySQL grouped by person and image type."""
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)

    cursor.execute("""
        SELECT fe.id AS emb_id, fe.person_id, fe.embedding,
               i.image_type, i.file_name, p.name AS person_name
        FROM face_embeddings fe
        JOIN images i ON fe.image_id = i.id
        JOIN persons p ON fe.person_id = p.id
        ORDER BY fe.person_id, i.image_type
    """)
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    # Parse embeddings
    embeddings = []
    for row in rows:
        emb_array = np.frombuffer(row["embedding"], dtype=np.float32)
        embeddings.append({
            "emb_id": row["emb_id"],
            "person_id": row["person_id"],
            "person_name": row["person_name"],
            "image_type": row["image_type"],
            "file_name": row["file_name"],
            "embedding": emb_array
        })

    return embeddings


def compute_all_matches(embeddings):
    """
    Compute cosine similarity for:
      1. Genuine pairs: selfie <-> ID of SAME person
      2. Impostor pairs: selfie <-> ID of DIFFERENT persons
    """
    # Separate selfies and IDs
    selfies = [e for e in embeddings if e["image_type"] == "Selfie"]
    ids = [e for e in embeddings if e["image_type"] == "ID"]

    print(f"[i] Selfie embeddings: {len(selfies)}")
    print(f"[i] ID embeddings:     {len(ids)}")

    genuine_scores = []
    impostor_scores = []
    all_results = []

    print("[i] Computing genuine matches (same person)...")
    for s in tqdm(selfies, desc="Genuine matching"):
        for i in ids:
            if s["person_id"] == i["person_id"]:
                sim = cosine_similarity(s["embedding"], i["embedding"])
                genuine_scores.append(sim)
                all_results.append({
                    "selfie_emb_id": s["emb_id"],
                    "id_emb_id": i["emb_id"],
                    "selfie_person_id": s["person_id"],
                    "id_person_id": i["person_id"],
                    "similarity": sim,
                    "is_same": True
                })

    print("[i] Computing impostor matches (different persons)...")
    for s in tqdm(selfies, desc="Impostor matching"):
        for i in ids:
            if s["person_id"] != i["person_id"]:
                sim = cosine_similarity(s["embedding"], i["embedding"])
                impostor_scores.append(sim)
                all_results.append({
                    "selfie_emb_id": s["emb_id"],
                    "id_emb_id": i["emb_id"],
                    "selfie_person_id": s["person_id"],
                    "id_person_id": i["person_id"],
                    "similarity": sim,
                    "is_same": False
                })

    print(f"\n[i] Genuine pairs:  {len(genuine_scores)}")
    print(f"[i] Impostor pairs: {len(impostor_scores)}")

    return genuine_scores, impostor_scores, all_results


def find_optimal_threshold(genuine_scores, impostor_scores):
    """Find optimal threshold using EER (Equal Error Rate)."""
    labels = [1] * len(genuine_scores) + [0] * len(impostor_scores)
    scores = genuine_scores + impostor_scores

    labels = np.array(labels)
    scores = np.array(scores)

    # ROC curve
    fpr, tpr, thresholds = roc_curve(labels, scores)

    # EER: where FPR = 1 - TPR (i.e., FPR = FNR)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = float(fpr[eer_idx])
    eer_threshold = float(thresholds[eer_idx])

    # AUC
    auc = roc_auc_score(labels, scores)

    # Also find threshold that maximizes accuracy
    best_acc = 0
    best_thresh = 0.5
    for t in thresholds:
        preds = (scores >= t).astype(int)
        acc = accuracy_score(labels, preds)
        if acc > best_acc:
            best_acc = acc
            best_thresh = float(t)

    return {
        "eer": eer,
        "eer_threshold": eer_threshold,
        "auc": auc,
        "best_accuracy": best_acc,
        "best_threshold": best_thresh,
        "labels": labels,
        "scores": scores
    }


def store_results(all_results, threshold):
    """Store match results in MySQL."""
    conn = get_connection()
    cursor = conn.cursor()

    # Clear previous results
    cursor.execute("DELETE FROM match_results")

    batch = []
    for r in all_results:
        verdict = r["similarity"] >= threshold
        batch.append((
            r["selfie_emb_id"], r["id_emb_id"],
            r["selfie_person_id"], r["id_person_id"],
            r["similarity"], r["is_same"],
            verdict, threshold
        ))

    batch_size = 1000
    for i in range(0, len(batch), batch_size):
        cursor.executemany(
            """INSERT INTO match_results
               (selfie_embedding_id, id_embedding_id, selfie_person_id, id_person_id,
                cosine_similarity, is_same_person, match_verdict, threshold_used)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
            batch[i:i+batch_size]
        )

    conn.commit()
    cursor.close()
    conn.close()
    print(f"[OK] Stored {len(batch)} match results in database.")


def print_report(genuine_scores, impostor_scores, metrics):
    """Print comprehensive accuracy report."""
    print("\n" + "=" * 60)
    print("  FACE MATCHING RESULTS")
    print("=" * 60)

    print(f"\n  {'Metric':<30} {'Value':>15}")
    print(f"  {'-'*45}")
    print(f"  {'Genuine pairs':<30} {len(genuine_scores):>15}")
    print(f"  {'Impostor pairs':<30} {len(impostor_scores):>15}")
    print(f"  {'Genuine mean similarity':<30} {np.mean(genuine_scores):>15.4f}")
    print(f"  {'Genuine min similarity':<30} {np.min(genuine_scores):>15.4f}")
    print(f"  {'Impostor mean similarity':<30} {np.mean(impostor_scores):>15.4f}")
    print(f"  {'Impostor max similarity':<30} {np.max(impostor_scores):>15.4f}")
    print()
    print(f"  {'ROC-AUC':<30} {metrics['auc']:>15.4f}")
    print(f"  {'EER (Equal Error Rate)':<30} {metrics['eer']:>15.4f}")
    print(f"  {'EER Threshold':<30} {metrics['eer_threshold']:>15.4f}")
    print(f"  {'Best Accuracy':<30} {metrics['best_accuracy']:>15.4f}")
    print(f"  {'Best Threshold':<30} {metrics['best_threshold']:>15.4f}")

    # Classification report at best threshold
    preds = (metrics["scores"] >= metrics["best_threshold"]).astype(int)
    labels = metrics["labels"]

    print(f"\n  Classification Report (threshold = {metrics['best_threshold']:.4f}):")
    print("  " + "-" * 55)
    report = classification_report(labels, preds,
                                   target_names=["Impostor", "Genuine"],
                                   digits=4)
    for line in report.split("\n"):
        print(f"  {line}")

    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    print(f"\n  Confusion Matrix:")
    print(f"  {'':>20} {'Pred Impostor':>15} {'Pred Genuine':>15}")
    print(f"  {'True Impostor':<20} {cm[0][0]:>15} {cm[0][1]:>15}")
    print(f"  {'True Genuine':<20} {cm[1][0]:>15} {cm[1][1]:>15}")

    # TAR @ FAR
    from sklearn.metrics import roc_curve as roc_curve_fn
    fpr, tpr, _ = roc_curve_fn(labels, metrics["scores"])
    for target_far in [0.01, 0.001, 0.0001]:
        idx = np.searchsorted(fpr, target_far)
        if idx < len(tpr):
            print(f"\n  TAR @ FAR={target_far:.4f}: {tpr[idx]:.4f}")

    print("\n" + "=" * 60)


def run():
    """Run matching pipeline."""
    print("=" * 60)
    print("  FACE MATCHING (Cosine Similarity)")
    print("=" * 60)

    # Load embeddings
    embeddings = load_embeddings()
    if len(embeddings) == 0:
        print("[!] No embeddings found. Run extract_embeddings.py first.")
        return

    # Compute matches
    genuine, impostor, all_results = compute_all_matches(embeddings)

    if len(genuine) == 0 or len(impostor) == 0:
        print("[!] Insufficient data for evaluation.")
        return

    # Find optimal threshold
    metrics = find_optimal_threshold(genuine, impostor)

    # Store results in DB
    store_results(all_results, metrics["best_threshold"])

    # Print report
    print_report(genuine, impostor, metrics)

    return metrics


if __name__ == "__main__":
    run()
