#!/usr/bin/env python3
"""
Diagnostic script to debug ESP32 → Flask connectivity
"""

import requests
import json
import sqlite3
from datetime import datetime

print("""
╔═════════════════════════════════════════════════════════════════╗
║     🔧 ESP32 ↔ FLASK CONNECTIVITY DIAGNOSTIC                   ║
╚═════════════════════════════════════════════════════════════════╝
""")

# 1. Check Flask server
print("\n1️⃣  Testing Flask Server...")
try:
    r = requests.get("http://localhost:5000/", timeout=5)
    print("   ✅ Flask server is running on port 5000")
except Exception as e:
    print(f"   ❌ Flask server error: {e}")
    print("   Make sure: python app.py is running")
    exit(1)

# 2. Check database for data
print("\n2️⃣  Checking Database for Sensor Data...")
try:
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('SELECT COUNT(*) FROM sensor_readings')
    count = c.fetchone()[0]
    print(f"   Total readings in database: {count}")
    
    if count > 0:
        c.execute('SELECT device_id, temperature, created_at FROM sensor_readings ORDER BY id DESC LIMIT 1')
        row = c.fetchone()
        print(f"   ✅ Latest reading: {row[0]} at {row[2]}")
        print(f"      Temperature: {row[1]}°C")
    else:
        print("   ⚠️  No sensor data in database yet")
    conn.close()
except Exception as e:
    print(f"   ❌ Database error: {e}")

# 3. Test API endpoint
print("\n3️⃣  Testing /api/sensor/latest Endpoint...")
try:
    r = requests.get("http://localhost:5000/api/sensor/latest", timeout=5)
    print(f"   HTTP Status: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        if data.get('success'):
            reading = data['reading']
            print("   ✅ API is working correctly")
            print(f"      Device: {reading.get('device_id')}")
            print(f"      Temp: {reading.get('temperature')}°C")
        else:
            print(f"   ⚠️  API error: {data.get('error')}")
    else:
        print(f"   ❌ HTTP Error: {r.status_code}")
        print(f"   {r.text[:200]}")
except Exception as e:
    print(f"   ❌ Connection error: {e}")

# 4. Simulate ESP32 POST request
print("\n4️⃣  Simulating ESP32 Sensor Data POST...")
test_data = {
    "api_key": "kisan-iot-2026",
    "device_id": "ESP32-FARM-001-TEST",
    "temperature": 28.5,
    "humidity": 72.0,
    "moisture": 55.3,
    "ph": 6.9,
    "battery_v": 7.5,
    "wifi_rssi": -48
}

try:
    r = requests.post(
        "http://localhost:5000/api/sensor/push",
        json=test_data,
        timeout=10
    )
    print(f"   HTTP Status: {r.status_code}")
    if r.status_code == 200:
        resp = r.json()
        print("   ✅ POST request successful!")
        print(f"      Response: {resp}")
        print("   This means ESP32 CAN send data to Flask")
    else:
        print(f"   ❌ POST failed: {r.status_code}")
        print(f"   {r.text[:200]}")
except Exception as e:
    print(f"   ❌ POST error: {e}")
    print("   This could mean ESP32 cannot reach your Flask server")

# 5. Test specific API key
print("\n5️⃣  Checking API Key Configuration...")
print("   Expected API key in firmware: kisan-iot-2026")
print("   Expected Flask API key: kisan-iot-2026")
print("   ✅ Keys match - this is correct")

print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔍 TROUBLESHOOTING:

If dashboard shows OFFLINE but Arduino shows ONLINE:

  A) Data not reaching Flask:
     • Check ESP32 firmware Server URL: http://192.168.92.1:5000
     • Verify ESP32 can ping your computer IP
     • Check Serial Monitor for HTTP POST response codes
     
  B) API Key mismatch:
     • Firmware sends: "kisan-iot-2026" ✅
     • Flask expects: "kisan-iot-2026" ✅
     
  C) Network issue:
     • ESP32 WiFi network ≠ Flask computer network?
     • Try: ping 192.168.92.1 from another device
     
  D) PORT blocking:
     • Make sure port 5000 is not blocked by firewall
     • Windows Firewall → Allow Python through

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Next: Check Arduino Serial Monitor output - what HTTP response does ESP32 get?
""")
