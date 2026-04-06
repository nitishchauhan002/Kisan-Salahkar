#!/usr/bin/env python3
"""
ESP32 Live Data Receiver — watches for incoming sensor data in real-time.
Keeps running until Ctrl+C.
"""
import sqlite3, time, os

DB_PATH = os.path.join(os.path.dirname(__file__), "users.db")

print("\n" + "="*60)
print("  📡 ESP32 → FLASK LIVE DATA RECEIVER")
print("="*60)
print(f"\n  Flask:  http://10.186.88.219:5000  (listening)")
print(f"  ESP32:  10.137.27.240  (sending every 30s)")
print(f"  API:    /api/sensor/push")
print(f"\n  ⏳ Waiting for incoming data...\n")

conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row
last_id = 0

# Get current max ID to only show NEW data
c = conn.cursor()
c.execute("SELECT MAX(id) FROM sensor_readings")
row = c.fetchone()
if row and row[0]:
    last_id = row[0]
    print(f"  (Skipping {last_id} old readings, showing only new ones)\n")

count = 0
try:
    while True:
        c = conn.cursor()
        c.execute(
            "SELECT * FROM sensor_readings WHERE id > ? ORDER BY id ASC",
            (last_id,)
        )
        rows = c.fetchall()

        for r in rows:
            count += 1
            last_id = r["id"]
            print(f"  ┌─ 📡 READING #{count} ─────────────────────────────")
            print(f"  │ Time:     {r['created_at']}")
            print(f"  │ Device:   {r['device_id']}")
            print(f"  │ 🌡️ Temp:   {r['temperature']}°C")
            print(f"  │ 💧 Humid:  {r['humidity']}%")
            print(f"  │ 🌾 Moist:  {r['moisture']}%")
            print(f"  │ ⚗️ pH:     {r['ph']}")
            print(f"  │ 🔋 Batt:   {r['battery_v']}V")
            print(f"  │ 📶 WiFi:   {r['wifi_rssi']} dBm")
            print(f"  └──────────────────────────────────────────────")
            print()

        time.sleep(2)
except KeyboardInterrupt:
    print(f"\n  ✋ Stopped. {count} new readings received.")
finally:
    conn.close()
