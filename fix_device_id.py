import sqlite3
from datetime import datetime

conn = sqlite3.connect('users.db')
c = conn.cursor()

# Clear old test data with wrong device IDs
c.execute('DELETE FROM sensor_readings WHERE device_id != ?', ('ESP32-FARM-001',))
deleted = c.rowcount
print(f'✓ Cleared {deleted} old test records')

# Insert fresh test data with correct device ID
c.execute('''INSERT INTO sensor_readings 
             (device_id, temperature, humidity, moisture, ph, battery_v, wifi_rssi)
             VALUES (?, ?, ?, ?, ?, ?, ?)''',
    ('ESP32-FARM-001', 28.5, 72.0, 55.3, 6.9, 7.5, -48)
)
conn.commit()
print('✓ Inserted test data with device_id: ESP32-FARM-001')

# Verify
c.execute('SELECT device_id, temperature, created_at FROM sensor_readings ORDER BY id DESC LIMIT 1')
row = c.fetchone()
print(f'✓ Latest reading: {row[0]} → Temp={row[1]}°C at {row[2]}')

# Test API
import requests
r = requests.get("http://localhost:5000/api/sensor/latest")
if r.status_code == 200 and r.json()['success']:
    print('\n✅ Dashboard API now working!')
    print('   Device shows ONLINE on IoT dashboard')
else:
    print(f'\n❌ API still showing: {r.json()}')

conn.close()
