import sqlite3

conn = sqlite3.connect('users.db')
conn.row_factory = sqlite3.Row
c = conn.cursor()
c.execute('SELECT * FROM sensor_readings ORDER BY id DESC LIMIT 5')
rows = c.fetchall()

print("\n" + "="*65)
print("  LIVE ESP32 DATA IN DATABASE")
print("="*65)

for r in rows:
    print(f"\n  [{r['created_at']}] {r['device_id']}")
    print(f"    🌡️  Temperature:  {r['temperature']}°C")
    print(f"    💧 Humidity:     {r['humidity']}%")
    print(f"    🌾 Soil Moisture: {r['moisture']}%")
    print(f"    ⚗️  Soil pH:      {r['ph']}")
    print(f"    🔋 Battery:      {r['battery_v']}V")
    print(f"    📶 WiFi Signal:  {r['wifi_rssi']} dBm")

c.execute("SELECT COUNT(*) FROM sensor_readings WHERE device_id='ESP32-FARM-001'")
total = c.fetchone()[0]
print(f"\n{'='*65}")
print(f"  Total readings from ESP32: {total}")
print(f"{'='*65}")

conn.close()
