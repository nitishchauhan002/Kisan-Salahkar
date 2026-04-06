#!/usr/bin/env python3
"""
ESP32 IoT Dashboard Connector
Shows real-time sensor data from ESP32 on the web dashboard
"""

import requests
import time
from datetime import datetime

def check_flask_server():
    """Check if Flask server is running"""
    try:
        r = requests.get('http://localhost:5000/', timeout=2)
        return True
    except:
        return False

def get_latest_sensor_data():
    """Get latest sensor reading from database via API"""
    try:
        r = requests.get('http://localhost:5000/api/sensor/latest', timeout=5)
        if r.status_code == 200:
            data = r.json()
            if data.get('success'):
                return data.get('reading')
    except:
        pass
    return None

def display_sensor_data(reading):
    """Display sensor data in formatted output"""
    if not reading:
        print("❌ No sensor data available")
        return
    
    print("\n" + "="*60)
    print("  📊 ESP32 SENSOR DATA")
    print("="*60)
    print(f"\n  Device ID: {reading.get('device_id')}")
    print(f"  Last Update: {reading.get('created_at')}")
    print(f"\n  🌡️  Temperature:  {reading.get('temperature', '--')}°C")
    print(f"  💧 Humidity:     {reading.get('humidity', '--')}%")
    print(f"  🌾 Soil Moisture: {reading.get('moisture', '--')}%")
    print(f"  ⚗️  Soil pH:      {reading.get('ph', '--')}")
    print(f"  🔋 Battery:      {reading.get('battery_v', '--')}V")
    print(f"  📡 WiFi Signal:  {reading.get('wifi_rssi', '--')} dBm")
    print("\n" + "="*60)

def main():
    print("\n🌾 Kisan Salahkar — ESP32 IoT System")
    print("-" * 60)
    
    # Check Flask server
    print("\n⏳ Checking Flask server...")
    if not check_flask_server():
        print("❌ Flask server not running!")
        print("\n   To start Flask, open a new terminal and run:")
        print("   python app.py")
        return
    print("✅ Flask server is running!")
    
    # Get sensor data
    print("\n⏳ Fetching sensor data from ESP32...")
    reading = get_latest_sensor_data()
    
    if reading:
        display_sensor_data(reading)
        
        print("\n📺 To view the IoT Dashboard:")
        print("   1. Open: http://localhost:5000/iot")
        print("   2. Dashboard will auto-refresh every 30 seconds")
        print("   3. See live gauges with temperature, humidity, moisture, pH")
        print("   4. View WiFi signal strength and battery level")
        print("   5. Get smart advisories based on sensor thresholds")
        
        print("\n📡 ESP32 will send new readings every 30 seconds")
        print("   Keep Flask server running to receive data")
        
    else:
        print("\n⏳ Waiting for first data from ESP32...")
        print("   The ESP32 should send its first reading within 30 seconds")
        print("   Make sure:")
        print("   • ESP32 is powered on and connected to WiFi")
        print("   • Firmware has been uploaded to ESP32")
        print("   • Server URL in firmware matches your IP: 192.168.92.1:5000")
        print("\n   Once data arrives, view it at: http://localhost:5000/iot")

if __name__ == "__main__":
    main()
