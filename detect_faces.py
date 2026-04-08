"""
detect_faces.py
---------------
Uses RetinaFace (via insightface) to detect faces in all images.
Stores bounding boxes, confidence scores, and 5-point landmarks in MySQL.
"""

import sys
import os
import numpy as np
import cv2
import insightface
from insightface.app import FaceAnalysis
import mysql.connector
from tqdm import tqdm

# Force UTF-8 output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from setup_database import get_connection


def build_face_app():
    """Initialize insightface FaceAnalysis with buffalo_l model."""
    app = FaceAnalysis(
        name="buffalo_l",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    # det_size: larger = better detection for small faces in ID cards
    app.prepare(ctx_id=0, det_size=(640, 640))
    print("[OK] InsightFace model loaded (RetinaFace + ArcFace).")
    return app


def detect_and_store(app):
    """Detect faces in all images and store results in MySQL."""
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)

    # Get all images
    cursor.execute("SELECT id, file_path, image_type, person_id FROM images ORDER BY id")
    images = cursor.fetchall()
    print(f"[i] Processing {len(images)} images for face detection...\n")

    stats = {"success": 0, "no_face": 0, "multi_face": 0, "error": 0}

    for img_row in tqdm(images, desc="Detecting faces"):
        img_id = img_row["id"]
        img_path = img_row["file_path"]

        try:
            # Read image
            img = cv2.imread(img_path)
            if img is None:
                print(f"  [!] Cannot read: {img_path}")
                stats["error"] += 1
                continue

            # Detect faces
            faces = app.get(img)

            if len(faces) == 0:
                # No face found
                cursor.execute(
                    "INSERT INTO face_detections (image_id, detection_status) "
                    "VALUES (%s, 'NO_FACE')",
                    (img_id,)
                )
                stats["no_face"] += 1
                continue

            # If multiple faces, pick the one with largest bounding box area
            if len(faces) > 1:
                faces = sorted(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
                status = "MULTI_FACE"
                stats["multi_face"] += 1
            else:
                status = "SUCCESS"
                stats["success"] += 1

            face = faces[0]
            bbox = face.bbox.astype(float)
            kps = face.kps.astype(float)  # 5-point landmarks
            det_score = float(face.det_score)

            cursor.execute(
                """INSERT INTO face_detections
                   (image_id, bbox_x1, bbox_y1, bbox_x2, bbox_y2, confidence,
                    landmark_left_eye_x, landmark_left_eye_y,
                    landmark_right_eye_x, landmark_right_eye_y,
                    landmark_nose_x, landmark_nose_y,
                    landmark_left_mouth_x, landmark_left_mouth_y,
                    landmark_right_mouth_x, landmark_right_mouth_y,
                    detection_status)
                   VALUES (%s, %s, %s, %s, %s, %s,
                           %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (img_id,
                 bbox[0], bbox[1], bbox[2], bbox[3], det_score,
                 kps[0][0], kps[0][1],
                 kps[1][0], kps[1][1],
                 kps[2][0], kps[2][1],
                 kps[3][0], kps[3][1],
                 kps[4][0], kps[4][1],
                 status)
            )

        except Exception as e:
            print(f"  [!] Error processing {img_path}: {e}")
            stats["error"] += 1

    conn.commit()
    cursor.close()
    conn.close()

    print(f"\n[OK] Face Detection Complete!")
    print(f"    Success:    {stats['success']}")
    print(f"    Multi-face: {stats['multi_face']} (largest face kept)")
    print(f"    No face:    {stats['no_face']}")
    print(f"    Errors:     {stats['error']}")
    return stats


def run():
    """Run face detection pipeline."""
    print("=" * 60)
    print("  FACE DETECTION (RetinaFace)")
    print("=" * 60)
    app = build_face_app()
    stats = detect_and_store(app)
    print()
    return app  # Return app so embeddings can reuse it


if __name__ == "__main__":
    run()
