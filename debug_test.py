# debug_test.py
import time

try:
    import pydevd_pycharm
    print("🔍 Attempting to connect to PyCharm debugger...")
    pydevd_pycharm.settrace(
        'localhost',
        port=5678,
        stdoutToServer=True,
        stderrToServer=True,
        suspend=False
    )
    print("✅ Connected to PyCharm debugger")
except Exception as e:
    print(f"❌ Failed to connect: {e}")

# Simulate work
for i in range(2):
    print(f"Working... {i}")
    time.sleep(1)
