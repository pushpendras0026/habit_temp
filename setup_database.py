"""
setup_database.py
-----------------
Creates the MySQL database and tables for the face matching pipeline.
Tables:
  - persons: metadata about each individual
  - images: file paths and types (ID / Selfie)
  - face_detections: RetinaFace bounding boxes, confidence, landmarks
  - face_embeddings: ArcFace 512-d vectors stored as BLOBs
  - match_results: selfie vs ID cosine similarity scores
"""

import mysql.connector
import pandas as pd
import os
import sys

# Force UTF-8 output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "",  # XAMPP default
    "port": 3306,
}
DB_NAME = "face_matching_db"

DATASET_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "Selfies ID Images dataset")
CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "selfie_id.csv")


def get_connection(use_db=True):
    """Return a MySQL connection. If use_db, connect to the project database."""
    cfg = dict(DB_CONFIG)
    if use_db:
        cfg["database"] = DB_NAME
    return mysql.connector.connect(**cfg)


def create_database():
    """Create database if it doesn't exist."""
    conn = get_connection(use_db=False)
    cursor = conn.cursor()
    cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{DB_NAME}` "
                   f"CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
    conn.commit()
    cursor.close()
    conn.close()
    print(f"[OK] Database '{DB_NAME}' ready.")


def create_tables():
    """Create all required tables."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS persons (
            id INT AUTO_INCREMENT PRIMARY KEY,
            set_id VARCHAR(255) NOT NULL UNIQUE,
            user_race VARCHAR(50),
            age INT,
            name VARCHAR(255),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        ) ENGINE=InnoDB
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS images (
            id INT AUTO_INCREMENT PRIMARY KEY,
            person_id INT NOT NULL,
            file_name VARCHAR(255) NOT NULL,
            file_path VARCHAR(1024) NOT NULL,
            image_type ENUM('ID', 'Selfie') NOT NULL,
            url VARCHAR(1024),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (person_id) REFERENCES persons(id) ON DELETE CASCADE,
            INDEX idx_person_type (person_id, image_type)
        ) ENGINE=InnoDB
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS face_detections (
            id INT AUTO_INCREMENT PRIMARY KEY,
            image_id INT NOT NULL,
            bbox_x1 FLOAT, bbox_y1 FLOAT,
            bbox_x2 FLOAT, bbox_y2 FLOAT,
            confidence FLOAT,
            landmark_left_eye_x FLOAT, landmark_left_eye_y FLOAT,
            landmark_right_eye_x FLOAT, landmark_right_eye_y FLOAT,
            landmark_nose_x FLOAT, landmark_nose_y FLOAT,
            landmark_left_mouth_x FLOAT, landmark_left_mouth_y FLOAT,
            landmark_right_mouth_x FLOAT, landmark_right_mouth_y FLOAT,
            detection_status ENUM('SUCCESS', 'NO_FACE', 'MULTI_FACE') DEFAULT 'SUCCESS',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (image_id) REFERENCES images(id) ON DELETE CASCADE,
            INDEX idx_image (image_id)
        ) ENGINE=InnoDB
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS face_embeddings (
            id INT AUTO_INCREMENT PRIMARY KEY,
            detection_id INT NOT NULL,
            image_id INT NOT NULL,
            person_id INT NOT NULL,
            embedding BLOB NOT NULL,
            embedding_dim INT DEFAULT 512,
            model_name VARCHAR(100) DEFAULT 'buffalo_l',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (detection_id) REFERENCES face_detections(id) ON DELETE CASCADE,
            FOREIGN KEY (image_id) REFERENCES images(id) ON DELETE CASCADE,
            FOREIGN KEY (person_id) REFERENCES persons(id) ON DELETE CASCADE,
            INDEX idx_person (person_id)
        ) ENGINE=InnoDB
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS match_results (
            id INT AUTO_INCREMENT PRIMARY KEY,
            selfie_embedding_id INT NOT NULL,
            id_embedding_id INT NOT NULL,
            selfie_person_id INT NOT NULL,
            id_person_id INT NOT NULL,
            cosine_similarity FLOAT NOT NULL,
            is_same_person BOOLEAN NOT NULL,
            match_verdict BOOLEAN,
            threshold_used FLOAT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (selfie_embedding_id) REFERENCES face_embeddings(id) ON DELETE CASCADE,
            FOREIGN KEY (id_embedding_id) REFERENCES face_embeddings(id) ON DELETE CASCADE,
            INDEX idx_same (is_same_person),
            INDEX idx_verdict (match_verdict)
        ) ENGINE=InnoDB
    """)

    conn.commit()
    cursor.close()
    conn.close()
    print("[OK] All tables created successfully.")


def populate_persons_and_images():
    """Read CSV and populate persons + images tables."""
    df = pd.read_csv(CSV_PATH)
    conn = get_connection()
    cursor = conn.cursor()

    # Determine the correct subfolder for each SetId
    hispanic_dir = os.path.join(DATASET_ROOT, "11_sets_Hispanics")
    caucasian_dir = os.path.join(DATASET_ROOT, "18_sets_ Caucasians")

    persons_inserted = 0
    images_inserted = 0

    for set_id, group in df.groupby("SetId"):
        set_id = set_id.strip()
        row = group.iloc[0]
        race = str(row["UserRace"]).strip()
        age = int(row["Age"])
        name = str(row["Name"]).strip()

        # Insert person
        cursor.execute(
            "INSERT IGNORE INTO persons (set_id, user_race, age, name) "
            "VALUES (%s, %s, %s, %s)",
            (set_id, race, age, name)
        )
        if cursor.rowcount > 0:
            persons_inserted += 1

        cursor.execute("SELECT id FROM persons WHERE set_id = %s", (set_id,))
        person_id = cursor.fetchone()[0]

        # Determine race folder
        race_folder = hispanic_dir if "Hispanic" in race else caucasian_dir

        for _, img_row in group.iterrows():
            fname = str(img_row["FName"]).strip()
            url = str(img_row["URL"]).strip()

            # Image type
            img_type = "ID" if fname.startswith("ID") else "Selfie"

            # Build full path
            full_path = os.path.join(race_folder, url)

            cursor.execute(
                "INSERT INTO images (person_id, file_name, file_path, image_type, url) "
                "VALUES (%s, %s, %s, %s, %s)",
                (person_id, fname, full_path, img_type, url)
            )
            images_inserted += 1

    conn.commit()
    cursor.close()
    conn.close()
    print(f"[OK] Populated {persons_inserted} persons and {images_inserted} images.")


def run():
    """Full database setup."""
    print("=" * 60)
    print("  DATABASE SETUP")
    print("=" * 60)
    create_database()
    create_tables()
    populate_persons_and_images()
    print("[OK] Database setup complete!\n")


if __name__ == "__main__":
    run()
