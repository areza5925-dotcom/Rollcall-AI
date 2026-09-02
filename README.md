# 📋 RollCall AI — Modular Face Recognition Attendance System

A reliable, modular, production-ready biometric attendance and roll call system built with Python, OpenCV, ace_recognition (dlib ResNet 128-d embeddings), Streamlit, and SQLite.

---

## 🌐 Sharing & Deployment Options

### Option 1: Streamlit Community Cloud (Free Public HTTPS URL)
1. Push this folder to a GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io/) and log in with GitHub.
3. Click **New app** -> Select your repository, branch (main), and set Main file path to pp.py.
4. The system automatically reads packages.txt (for Linux system dependencies) and equirements.txt.
5. Your app is live with a public HTTPS link (browser webcam permissions work smoothly on mobile and desktop).

---

### Option 2: Share on Local Wi-Fi Network (LAN)
Run the app bound to all network interfaces:
`ash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
`
Any phone, tablet, or laptop connected to the same Wi-Fi can open:
http://<YOUR_COMPUTER_IP>:8501

*(The sidebar inside the app automatically shows your computer's local IP address).*

---

### Option 3: Instant Public Demo Link (via ngrok or localtunnel)
If you want to share a live link from your local machine immediately:

**Using Localtunnel (No registration required):**
`ash
# Terminal 1: Run app
streamlit run app.py

# Terminal 2: Expose port 8501
npx localtunnel --port 8501
`

**Using Ngrok:**
`ash
ngrok http 8501
`

---

### Option 4: Docker Container Deployment
Run everywhere with a single command:
`ash
# Build and run with docker-compose
docker compose up -d

# Or using plain docker
docker build -t rollcall-ai .
docker run -d -p 8501:8501 --name rollcall-ai rollcall-ai
`

---

## 📁 Project Architecture

`
face_attendance/
├── .streamlit/
│   └── config.toml      # Server, theme, headless & CORS configurations
├── packages.txt         # Debian/Ubuntu C++ build dependencies for Streamlit Cloud
├── Dockerfile           # Multi-platform Linux container build
├── docker-compose.yml   # 1-command Docker service definition
├── app.py               # Streamlit web interface (Dashboard, Enrollment, Roll Call)
├── database.py          # SQLite schema, transaction handling, and 5-min cooldown
├── vision.py            # Computer vision engine (embedding extraction & face matcher)
├── kiosk.py             # High-performance 30 FPS OpenCV desktop roll-call kiosk
├── main.py              # Unified CLI and master application launcher
├── test_system.py       # Self-check test suite (verifies DB, cooldown, queries)
├── run.bat              # One-click Windows launcher
└── requirements.txt     # Python package dependencies
`

---

## 🖥️ Command-Line Interface (CLI)

| Task | Command |
|---|---|
| Launch Web UI | python main.py web |
| Launch Desktop Kiosk (OpenCV) | python main.py kiosk |
| Enroll Worker via CLI | python main.py enroll --name "Jane Doe" --image "jane.jpg" |
| Export Attendance to CSV | python main.py export --out "report.csv" --date "2026-09-02" |
| Run Test Suite | python main.py test |
| Seed Demo Data | python main.py seed |
