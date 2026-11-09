"""SQLite backend: stores sessions, detection events and snapshot records."""
import sqlite3
from datetime import datetime

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT, ended_at TEXT,
    camera_index INTEGER, width INTEGER, height INTEGER,
    total_detections INTEGER DEFAULT 0, avg_fps REAL
);
CREATE TABLE IF NOT EXISTS detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER REFERENCES sessions(id),
    timestamp TEXT, color TEXT, num_regions INTEGER, largest_area INTEGER,
    bbox_x INTEGER, bbox_y INTEGER, bbox_w INTEGER, bbox_h INTEGER, fps REAL
);
CREATE TABLE IF NOT EXISTS snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER REFERENCES sessions(id),
    timestamp TEXT, path TEXT
);
"""


def _now():
    return datetime.now().isoformat(timespec="seconds")


class DetectionStore:
    def __init__(self, path="detections.db"):
        self.conn = sqlite3.connect(path)
        self.conn.executescript(SCHEMA)

    def start_session(self, camera_index, width, height):
        cur = self.conn.execute(
            "INSERT INTO sessions (started_at, camera_index, width, height) VALUES (?,?,?,?)",
            (_now(), camera_index, width, height))
        self.conn.commit()
        return cur.lastrowid

    def log_detection(self, session_id, color, num_regions, largest_area, bbox, fps):
        self.conn.execute(
            "INSERT INTO detections (session_id, timestamp, color, num_regions, largest_area,"
            " bbox_x, bbox_y, bbox_w, bbox_h, fps) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (session_id, _now(), color, num_regions, largest_area, *bbox, round(fps, 1)))
        self.conn.commit()

    def log_snapshot(self, session_id, path):
        self.conn.execute("INSERT INTO snapshots (session_id, timestamp, path) VALUES (?,?,?)",
                          (session_id, _now(), path))
        self.conn.commit()

    def end_session(self, session_id, total, avg_fps):
        self.conn.execute("UPDATE sessions SET ended_at=?, total_detections=?, avg_fps=? WHERE id=?",
                          (_now(), total, round(avg_fps, 1), session_id))
        self.conn.commit()

    def close(self):
        self.conn.close()
