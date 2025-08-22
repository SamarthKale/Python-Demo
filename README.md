# Python Demo

This project combines two main components:
1.  **Minecraft-like VR Game** 🧱 (built with the Ursina Engine)
2.  **Cyberpunk Hand & Face Tracker** 🤖 (built with Mediapipe + OpenCV)

This guide provides step-by-step instructions to help you install, run, and play, even if you are new to Python.

---

## 📦 1. Requirements

### ✅ Install Python
First, you need to download and install **Python 3.10.x**. This project is not compatible with Python 3.11 or newer versions.
- [**Python 3.10.11 Download Link**](https://www.python.org/downloads/release/python-31011/)

---

### ✅ Create a Virtual Environment
Using a virtual environment is highly recommended to manage project dependencies and avoid conflicts.

```bash
# 1. Navigate to your project folder in the terminal
cd path/to/your/project

# 2. Create the virtual environment
python -m venv .venv

# 3. Activate the environment
# On Windows (PowerShell/CMD):
.venv\Scripts\activate

# On macOS/Linux:
source .venv/bin/activate

#Run this command
pip install ursina==5.2.0 mediapipe==0.10.14 opencv-python==4.10.0.84 numpy==1.26.4 matplotlib==3.9.2 panda3d==1.10.14
