"""Print stored sessions and the most recent detection events (proof the backend stores data)."""
import sqlite3
import sys

db = sys.argv[1] if len(sys.argv) > 1 else "detections.db"
conn = sqlite3.connect(db)

print("SESSIONS")
print(f"{'id':>3} {'started':<20} {'res':<9} {'detections':>10} {'avg_fps':>8}")
for r in conn.execute("SELECT id, started_at, width, height, total_detections, avg_fps FROM sessions"):
    print(f"{r[0]:>3} {r[1]:<20} {f'{r[2]}x{r[3]}':<9} {r[4]:>10} {r[5] or 0:>8}")

print("\nLAST 10 DETECTIONS")
print(f"{'id':>3} {'session':>7} {'time':<20} {'color':<7} {'regions':>7} {'area':>7} {'fps':>5}")
for r in conn.execute("SELECT id, session_id, timestamp, color, num_regions, largest_area, fps "
                      "FROM detections ORDER BY id DESC LIMIT 10"):
    print(f"{r[0]:>3} {r[1]:>7} {r[2]:<20} {r[3]:<7} {r[4]:>7} {r[5]:>7} {r[6]:>5}")
