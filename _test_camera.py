import requests
import base64
from PIL import Image
import io
import numpy as np

# Create a small dummy image (100x100 RGB)
img = Image.new('RGB', (100, 100), color=(34, 139, 34))  # Green leaf-ish color
img_bytes = io.BytesIO()
img.save(img_bytes, format='JPEG')
img_bytes.seek(0)
img_b64 = base64.b64encode(img_bytes.read()).decode('utf-8')

# Send to camera endpoint
url = 'http://localhost:5000/api/sensor/camera'
data = {
    'api_key': 'kisan-iot-2026',
    'device_id': 'ESP32-CAM-001-TEST',
    'image': f'data:image/jpeg;base64,{img_b64}'
}

print('Testing ESP32-CAM image upload...')
print(f'Image size: {len(img_b64)} chars (base64)')

try:
    r = requests.post(url, json=data, timeout=10)
    print(f'\n📷 POST /api/sensor/camera: {r.status_code}')
    print(f'Response: {r.json()}')
    
    if r.status_code == 200:
        j = r.json()
        print(f"\n✅ Image stored!")
        print(f"   Diagnosis: {j.get('diagnosis')}")
        print(f"   Confidence: {j.get('confidence')}%")
        
        # Check database  
        import sqlite3
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute('SELECT COUNT(*) FROM sensor_images')
        count = c.fetchone()[0]
        print(f"\n📊 Total images in DB: {count}")
        c.execute('SELECT device_id, diagnosis FROM sensor_images ORDER BY created_at DESC LIMIT 1')
        row = c.fetchone()
        if row:
            print(f"   Latest: {row[0]} → {row[1]}")
        conn.close()
        
        print("\n=== CAMERA API WORKING ===")
    else:
        print(f"Error: {r.text}")
        
except Exception as e:
    print(f'Connection failed: {e}')
    print('\nMake sure Flask is running: python app.py')
