#!/usr/bin/env python3
"""
Real-time ESP32 WiFi Sensor Monitor
Watches for incoming sensor data from ESP32 over WiFi
"""

import sqlite3
import time
from datetime import datetime

print("""
╔═══════════════════════════════════════════════════════════════╗
║          🌾 KISAN SALAHKAR — ESP32 WiFi Monitor              ║
║                  Real-time Sensor Data                        ║
╚═══════════════════════════════════════════════════════════════╝
""")

print("⏳ Monitoring for sensor data from ESP32 over WiFi...")
print("   Waiting for first reading (should arrive within 30 seconds)...")
print()

conn = sqlite3.connect('users.db')
last_id = 0
reading_count = 0

try:
    while True:
        c = conn.cursor()
        c.execute('''
            SELECT id, device_id, temperature, humidity, moisture, ph, 
                   battery_v, wifi_rssi, created_at 
            FROM sensor_readings 
            WHERE id > ? 
            ORDER BY id DESC 
            LIMIT 1
        ''', (last_id,))
        
        row = c.fetchone()
        
        if row:
            last_id = row[0]
            reading_count += 1
            device_id, temp, humidity, moisture, ph, batt, rssi, timestamp = row[1:]
            
            print(f"\n{'='*60}")
            print(f"✅ READING #{reading_count} — {timestamp}")
            print(f"{'='*60}")
            print(f"\n📱 Device:        {device_id}")
            print(f"📡 WiFi Signal:   {rssi} dBm")
            print(f"🔋 Battery:       {batt}V")
            print(f"\n🌡️  Temperature:   {temp}°C")
            print(f"💧 Humidity:      {humidity}%")
            print(f"🌾 Soil Moisture: {moisture}%")
            print(f"⚗️  Soil pH:       {ph}")
            print()
            
            # Check thresholds
            print("⚠️  Status Alerts:")
            alerts = []
            
            if moisture < 25:
                alerts.append("  🌾 WATER LOW - Soil too dry, irrigate now!")
            if moisture > 85:
                alerts.append("  💧 Water HIGH - Soil waterlogged, risk of root rot")
            if temp > 40:
                alerts.append("  🔥 TOO HOT - Heat stress risk, provide shade")
            if temp < 5:
                alerts.append("  ❄️  FROST - Cold damage risk for seedlings")
            if ph < 4.5:
                alerts.append("  ⚗️  TOO ACIDIC - Apply lime to raise pH")
            if ph > 8.5:
                alerts.append("  ⚗️  TOO ALKALINE - Apply gypsum to lower pH")
            if humidity > 90:
                alerts.append("  🍄 FUNGAL RISK - High humidity, spray fungicide")
            if batt < 6.5:
                alerts.append("  🔋 LOW BATTERY - Charge ESP32 soon")
            
            if alerts:
                for alert in alerts:
                    print(alert)
            else:
                print("  ✅ All sensors in good range!")
            
            print(f"\n{'='*60}")
            print("📺 View full dashboard: http://localhost:5000/iot")
            print("="*60)
        
        time.sleep(2)

except KeyboardInterrupt:
    print("\n\n✋ Monitoring stopped.")
    print(f"\n📊 Total readings received: {reading_count}")
    if reading_count > 0:
        print("✅ WiFi connection working! Data received from ESP32.")
    else:
        print("⚠️  No data received yet. Check:")
        print("   • ESP32 is powered on")
        print("   • WiFi connection successful (check Serial Monitor)")
        print("   • Server IP in firmware matches: 192.168.92.1:5000")
finally:
    conn.close()
