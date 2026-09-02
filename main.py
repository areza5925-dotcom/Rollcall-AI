import sys
import os
import argparse
import subprocess
import pandas as pd
from datetime import datetime
import database

def cmd_web(port: int = 8501):
    """Launches the Streamlit Web Application."""
    print(f"[*] Starting Streamlit Web App on port {port}...")
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", str(port)])
    except KeyboardInterrupt:
        print("\n[*] Web server stopped.")

def cmd_kiosk(camera: int = 0, tolerance: float = 0.50):
    """Launches the continuous OpenCV real-time kiosk."""
    from kiosk import run_kiosk
    run_kiosk(camera_index=camera, tolerance=tolerance)

def cmd_enroll(name: str, image_path: str):
    """Enrolls a worker from the command line."""
    import vision
    if not os.path.exists(image_path):
        print(f"[Error] Image path not found: {image_path}")
        return
        
    database.init_db()
    try:
        print(f"[*] Encoding face from '{image_path}' for '{name}'...")
        vector = vision.encode_face(image_path)
        worker_id = database.insert_worker(name, vector)
        print(f"[Success] Worker '{name}' registered successfully with Worker ID #{worker_id}!")
    except Exception as e:
        print(f"[Error] Enrollment failed: {e}")

def cmd_test():
    """Runs the self-check verification test suite."""
    from test_system import test_database_and_cooldown
    test_database_and_cooldown()

def cmd_export(output_file: str = "attendance_export.csv", date: str = None):
    """Exports attendance records to a CSV file."""
    database.init_db()
    logs = database.get_attendance_logs(date)
    if not logs:
        print(f"[*] No logs found for date filter: {date or 'ALL'}")
        return
    df = pd.DataFrame(logs)
    df.to_csv(output_file, index=False)
    print(f"[Success] Exported {len(df)} records to '{output_file}'.")

def cmd_seed():
    """Generates synthetic demo workers and attendance records for testing."""
    import numpy as np
    database.init_db()
    print("[*] Seeding database with demo workers and check-ins...")
    
    demo_names = ["Sarah Connor", "John Wick", "Ellen Ripley", "Tony Stark"]
    today_str = datetime.now().strftime("%Y-%m-%d")
    
    for name in demo_names:
        mock_vec = np.random.randn(128).astype(np.float64)
        worker_id = database.insert_worker(name, mock_vec)
        database.log_attendance(worker_id, cooldown_minutes=0)
        print(f"  - Created & checked-in: {name} (ID #{worker_id})")
        
    print("[Success] Demo data seeded successfully! You can now view them on the dashboard.")

def main():
    parser = argparse.ArgumentParser(
        description="RollCall AI - Complete Face Recognition Attendance System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py web              Launch the Streamlit Web Application (Default)
  python main.py kiosk            Launch the real-time continuous OpenCV camera kiosk
  python main.py enroll --name "Jane Doe" --image "jane.jpg"   Enroll from CLI
  python main.py export --out report.csv                        Export attendance to CSV
  python main.py test             Run validation test suite
  python main.py seed             Generate demo data for testing dashboard
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")
    
    # Subcommand: web
    p_web = subparsers.add_parser("web", help="Run Streamlit Web UI")
    p_web.add_argument("--port", type=int, default=8501, help="Port to serve Streamlit app on (default: 8501)")
    
    # Subcommand: kiosk
    p_kiosk = subparsers.add_parser("kiosk", help="Run real-time OpenCV desktop kiosk")
    p_kiosk.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0)")
    p_kiosk.add_argument("--tolerance", type=float, default=0.50, help="Recognition tolerance (default: 0.50)")
    
    # Subcommand: enroll
    p_enroll = subparsers.add_parser("enroll", help="Enroll a worker using a local image file")
    p_enroll.add_argument("--name", required=True, type=str, help="Worker full name")
    p_enroll.add_argument("--image", required=True, type=str, help="Path to reference photo (JPG/PNG)")
    
    # Subcommand: export
    p_export = subparsers.add_parser("export", help="Export attendance logs to CSV")
    p_export.add_argument("--out", type=str, default="attendance_export.csv", help="Output CSV path")
    p_export.add_argument("--date", type=str, default=None, help="Filter date (YYYY-MM-DD)")
    
    # Subcommand: test
    subparsers.add_parser("test", help="Run system test suite")
    
    # Subcommand: seed
    subparsers.add_parser("seed", help="Seed demo workers and attendance data")
    
    args = parser.parse_args()
    
    # Default to web if no command passed
    if args.command is None or args.command == "web":
        port = getattr(args, "port", 8501)
        cmd_web(port=port)
    elif args.command == "kiosk":
        cmd_kiosk(camera=args.camera, tolerance=args.tolerance)
    elif args.command == "enroll":
        cmd_enroll(name=args.name, image_path=args.image)
    elif args.command == "export":
        cmd_export(output_file=args.out, date=args.date)
    elif args.command == "test":
        cmd_test()
    elif args.command == "seed":
        cmd_seed()

if __name__ == "__main__":
    main()
