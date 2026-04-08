"""
run_pipeline.py
---------------
Main entry point that orchestrates the complete face matching pipeline:
  1. Setup MySQL database & populate from CSV
  2. Detect faces using RetinaFace
  3. Extract 512-d ArcFace embeddings
  4. Match selfies vs IDs and evaluate accuracy
"""

import time
import sys
import os

# Force UTF-8 output on Windows
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def main():
    total_start = time.time()

    print("+" + "=" * 58 + "+")
    print("|  FACE MATCHING PIPELINE: RetinaFace + ArcFace + MySQL   |")
    print("+" + "=" * 58 + "+")
    print()

    # -- Step 1: Database Setup ----------------------------------------
    print(">> STEP 1/4: Database Setup")
    t = time.time()
    import setup_database
    setup_database.run()
    print(f"  Time: {time.time() - t:.1f}s\n")

    # -- Step 2: Face Detection ----------------------------------------
    print(">> STEP 2/4: Face Detection (RetinaFace)")
    t = time.time()
    import detect_faces
    app = detect_faces.run()
    print(f"  Time: {time.time() - t:.1f}s\n")

    # -- Step 3: Embedding Extraction ----------------------------------
    print(">> STEP 3/4: Embedding Extraction (ArcFace)")
    t = time.time()
    import extract_embeddings
    extract_embeddings.run(app=app)
    print(f"  Time: {time.time() - t:.1f}s\n")

    # -- Step 4: Face Matching & Evaluation ----------------------------
    print(">> STEP 4/4: Face Matching & Evaluation")
    t = time.time()
    import match_faces
    metrics = match_faces.run()
    print(f"  Time: {time.time() - t:.1f}s\n")

    # -- Summary -------------------------------------------------------
    total_time = time.time() - total_start
    print("+" + "=" * 58 + "+")
    print("|  PIPELINE COMPLETE                                      |")
    print("+" + "=" * 58 + "+")
    if metrics:
        print(f"  ROC-AUC:       {metrics['auc']:.4f}")
        print(f"  Best Accuracy: {metrics['best_accuracy']:.4f}")
        print(f"  EER:           {metrics['eer']:.4f}")
    print(f"  Total Time:    {total_time:.1f}s")
    print("+" + "=" * 58 + "+")


if __name__ == "__main__":
    main()
