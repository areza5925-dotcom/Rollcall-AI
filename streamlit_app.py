import streamlit as st
import pandas as pd
from datetime import datetime
import socket
import numpy as np
import database
import vision

# --- Page Configuration ---
st.set_page_config(
    page_title="RollCall AI | Face Attendance System",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize database schema on startup
database.init_db()

def get_local_ip():
    """Helper to get local network IP for easy LAN sharing."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "localhost"

# --- Sidebar Navigation ---
st.sidebar.title("📋 RollCall AI")
st.sidebar.caption("Modular Face Recognition Attendance System")
st.sidebar.markdown("---")

menu = st.sidebar.radio(
    "Navigation Menu",
    ["📊 Dashboard", "➕ Enroll Worker", "📹 Live Roll Call"],
    index=0
)

# --- Sidebar System Stats & Sharing Widget ---
st.sidebar.markdown("---")
st.sidebar.subheader("System Status")
total_enrolled = database.get_worker_count()
today_str = datetime.now().strftime("%Y-%m-%d")
today_logs = database.get_attendance_logs(today_str)
unique_present_today = len(set(log["Worker ID"] for log in today_logs))

st.sidebar.metric(label="Enrolled Staff", value=total_enrolled)
st.sidebar.metric(label="Present Today", value=unique_present_today)
st.sidebar.info(f"📅 **Date:** {today_str}")

# Network Sharing Info
with st.sidebar.expander("🌐 Share on Local Network", expanded=False):
    local_ip = get_local_ip()
    st.markdown(f"""
    **Access from phones or tablets on same Wi-Fi:**  
    http://{local_ip}:8501
    """)

# Demo Sandbox Helper for Cloud Visitors
with st.sidebar.expander("🧪 Demo Sandbox Tools", expanded=False):
    st.caption("Quickly populate demo data for testing:")
    if st.button("🌱 Seed 4 Demo Profiles", use_container_width=True):
        demo_profiles = ["Sarah Connor", "John Wick", "Ellen Ripley", "Tony Stark"]
        for name in demo_profiles:
            mock_vec = np.random.randn(128).astype(np.float64)
            worker_id = database.insert_worker(name, mock_vec)
            database.log_attendance(worker_id, cooldown_minutes=0)
        st.success("Seeded demo workers and attendance!")
        st.rerun()

# ==============================================================================
# TAB 1: DASHBOARD
# ==============================================================================
if menu == "📊 Dashboard":
    st.title("📊 Attendance Dashboard")
    st.markdown("Monitor daily attendance logs, view check-in metrics, and export data.")
    
    col_date, col_refresh = st.columns([3, 1])
    with col_date:
        selected_date = st.date_input("Filter by Date", datetime.now())
        filter_date_str = selected_date.strftime("%Y-%m-%d")
    with col_refresh:
        st.write("")
        st.write("")
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.rerun()

    # Query attendance records for selected date
    records = database.get_attendance_logs(filter_date_str)
    
    # KPI Metrics Row
    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1:
        st.metric(label="Total Enrolled Staff", value=total_enrolled)
    with m_col2:
        date_unique_present = len(set(r["Worker ID"] for r in records))
        st.metric(label=f"Staff Present ({filter_date_str})", value=date_unique_present)
    with m_col3:
        st.metric(label="Total Check-in Events", value=len(records))

    st.markdown("---")
    st.subheader(f"Attendance Records for {filter_date_str}")
    
    if records:
        df = pd.DataFrame(records)
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Record ID": st.column_config.NumberColumn("Record ID", width="small"),
                "Worker ID": st.column_config.NumberColumn("Worker ID", width="small"),
                "Worker Name": st.column_config.TextColumn("Worker Name", width="medium"),
                "Timestamp": st.column_config.TextColumn("Check-in Timestamp", width="medium"),
                "Date": st.column_config.TextColumn("Date", width="small"),
            }
        )
        
        # CSV Export Button
        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Export Logs to CSV",
            data=csv_data,
            file_name=f"attendance_report_{filter_date_str}.csv",
            mime="text/csv",
            use_container_width=False
        )
    else:
        st.info(f"No attendance records found for {filter_date_str}.")

    # Collapsible Worker Directory
    with st.expander("👥 Registered Staff Directory", expanded=False):
        workers = database.get_all_workers()
        if workers:
            worker_df = pd.DataFrame([
                {"Worker ID": w["id"], "Full Name": w["name"], "Enrolled At": w["created_at"]}
                for w in workers
            ])
            st.dataframe(worker_df, use_container_width=True, hide_index=True)
            
            # Option to remove worker
            del_id = st.selectbox(
                "Delete Worker Profile", 
                options=[w["id"] for w in workers], 
                format_func=lambda x: f"ID #{x} - {next(w['name'] for w in workers if w['id'] == x)}"
            )
            if st.button("🗑️ Delete Selected Worker", type="secondary"):
                database.delete_worker(del_id)
                st.success(f"Worker #{del_id} deleted successfully.")
                st.rerun()
        else:
            st.caption("No registered workers found. Enroll workers using the '➕ Enroll Worker' tab.")

# ==============================================================================
# TAB 2: ENROLL WORKER
# ==============================================================================
elif menu == "➕ Enroll Worker":
    st.title("➕ Enroll New Staff Member")
    st.markdown("Register a worker into the database with their reference photograph and 128-d face embedding.")
    
    with st.form("enrollment_form", clear_on_submit=False):
        worker_name = st.text_input("Worker Full Name", placeholder="e.g., Alex Johnson")
        
        input_mode = st.radio("Photo Input Method", ["Upload Image File", "Use Webcam / Phone Camera"], horizontal=True)
        
        uploaded_image = None
        if input_mode == "Upload Image File":
            uploaded_image = st.file_uploader("Upload Clear Frontal Photo", type=["jpg", "jpeg", "png"])
        else:
            uploaded_image = st.camera_input("Take Enrollment Photo")
            
        submitted = st.form_submit_button("💾 Encode & Enroll Worker", use_container_width=True)
        
        if submitted:
            if not worker_name.strip():
                st.error("⚠️ Please enter a valid worker name.")
            elif uploaded_image is None:
                st.error("⚠️ Please provide a photo (upload file or camera snapshot).")
            else:
                with st.spinner("Analyzing image and extracting 128-d facial embedding..."):
                    try:
                        # Extract 128-d face embedding
                        face_vector = vision.encode_face(uploaded_image)
                        
                        # Store in SQLite
                        new_id = database.insert_worker(worker_name.strip(), face_vector)
                        
                        st.success(f"🎉 **{worker_name.strip()}** successfully registered with Worker ID **#{new_id}**!")
                        st.balloons()
                    except ValueError as ve:
                        st.error(f"❌ Validation Error: {ve}")
                    except Exception as e:
                        st.error(f"❌ Unexpected Error during enrollment: {e}")

# ==============================================================================
# TAB 3: LIVE ROLL CALL
# ==============================================================================
elif menu == "📹 Live Roll Call":
    st.title("📹 Live Roll Call & Attendance Verification")
    st.markdown("Capture a camera frame or upload an image to identify enrolled staff and automatically mark attendance.")
    
    # Configuration Bar
    col_settings, col_cooldown = st.columns([2, 1])
    with col_settings:
        tolerance = st.slider(
            "Recognition Strictness (Lower = Stricter, 0.50 recommended)",
            min_value=0.30,
            max_value=0.70,
            value=0.50,
            step=0.02,
            help="Tolerance controls false-positive rejection. 0.50 provides a balanced threshold for real-world lighting."
        )
    with col_cooldown:
        cooldown_mins = st.number_input(
            "Cooldown Window (Minutes)",
            min_value=1,
            max_value=60,
            value=5,
            help="Prevents duplicate check-ins for the same worker within this time window."
        )
        
    st.markdown("---")
    
    roll_mode = st.radio("Capture Mode", ["Webcam / Mobile Camera Snapshot", "Upload Image for Roll Call"], horizontal=True)
    
    roll_image = None
    if roll_mode == "Webcam / Mobile Camera Snapshot":
        roll_image = st.camera_input("Capture Roll Call Frame")
    else:
        roll_image = st.file_uploader("Upload Group/Individual Photo", type=["jpg", "jpeg", "png"])
        
    if roll_image is not None:
        # Load all enrolled workers
        known_workers = database.get_all_workers()
        
        if not known_workers:
            st.warning("⚠️ No workers are currently enrolled in the database. Please enroll workers first in the '➕ Enroll Worker' tab.")
        else:
            with st.spinner("Analyzing frame and matching faces..."):
                detections, annotated_img = vision.recognize_faces(
                    image_input=roll_image,
                    known_workers=known_workers,
                    tolerance=tolerance
                )
                
            col_img, col_results = st.columns([3, 2])
            
            with col_img:
                st.subheader("Visual Recognition")
                st.image(annotated_img, caption="Detected Faces with Bounding Boxes", use_container_width=True)
                
            with col_results:
                st.subheader("Attendance Log Actions")
                
                if not detections:
                    st.info("No faces detected in the image.")
                else:
                    for det in detections:
                        worker_id = det["worker_id"]
                        name = det["name"]
                        dist = det["distance"]
                        is_match = det["is_match"]
                        
                        if is_match and worker_id is not None:
                            # Attempt to log attendance with cooldown enforcement
                            logged, msg = database.log_attendance(
                                worker_id=worker_id,
                                cooldown_minutes=int(cooldown_mins)
                            )
                            if logged:
                                st.success(f"✅ **{name}** (ID #{worker_id})\n- Distance: {dist:.2f}\n- {msg}")
                            else:
                                st.info(f"⏳ **{name}** (ID #{worker_id})\n- Distance: {dist:.2f}\n- {msg}")
                        else:
                            st.error(f"❌ **Unknown Person**\n- Closest Distance: {dist:.2f} (Exceeds tolerance {tolerance:.2f})")
