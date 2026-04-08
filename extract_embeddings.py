"""
extract_embeddings.py
---------------------
Uses ArcFace (via insightface) to extract 512-dimensional face embeddings
from detected faces. Stores embeddings as BLOBs in MySQL.
"""

import sys
import os
import numpy as np
import cv2
from tqdm import tqdm

# Force UTF-8 output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from setup_database import get_connection


def extract_and_store(app):
    """Extract ArcFace embeddings for all detected faces."""
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)

    # Get all successful detections joined with image info
    cursor.execute("""
        SELECT fd.id AS detection_id, fd.image_id, i.file_path, i.person_id,
               fd.detection_status
        FROM face_detections fd
        JOIN images i ON fd.image_id = i.id
        WHERE fd.detection_status IN ('SUCCESS', 'MULTI_FACE')
        ORDER BY fd.id
    """)
    detections = cursor.fetchall()
    print(f"[i] Extracting ArcFace embeddings for {len(detections)} faces...\n")

    count = 0
    errors = 0

    for det in tqdm(detections, desc="Extracting embeddings"):
        try:
            img = cv2.imread(det["file_path"])
            if img is None:
                errors += 1
                continue

            faces = app.get(img)
            if len(faces) == 0:
                errors += 1
                continue

            # Take the largest face (same logic as detection)
            if len(faces) > 1:
                faces = sorted(
                    faces,
                    key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
                    reverse=True
                )

            face = faces[0]
            embedding = face.normed_embedding  # 512-d, L2-normalized

            # Validate
            assert embedding.shape == (512,), f"Unexpected shape: {embedding.shape}"

            # Store as bytes
            emb_bytes = embedding.astype(np.float32).tobytes()

            cursor.execute(
                """INSERT INTO face_embeddings
                   (detection_id, image_id, person_id, embedding, embedding_dim, model_name)
                   VALUES (%s, %s, %s, %s, %s, %s)""",
                (det["detection_id"], det["image_id"], det["person_id"],
                 emb_bytes, 512, "buffalo_l")
            )
            count += 1

        except Exception as e:
            print(f"  [!] Error: {e}")
            errors += 1

    conn.commit()
    cursor.close()
    conn.close()

    print(f"\n[OK] Embedding Extraction Complete!")
    print(f"    Embeddings stored: {count}")
    print(f"    Errors: {errors}")


def run(app=None):
    """Run embedding extraction."""
    print("=" * 60)
    print("  EMBEDDING EXTRACTION (ArcFace)")
    print("=" * 60)

    if app is None:
        from detect_faces import build_face_app
        app = build_face_app()

    extract_and_store(app)
    print()


if __name__ == "__main__":
    run()
