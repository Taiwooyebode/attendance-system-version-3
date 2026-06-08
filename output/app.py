import os
import base64
import random
import string
import smtplib
import numpy as np
import face_recognition
from datetime import datetime
from functools import wraps
from flask import (
    Flask, render_template, request, redirect,
    url_for, flash, session, jsonify, Response
)
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import cv2
from math import radians, cos, sin, asin, sqrt
import  os
from dotenv import load_dotenv
load_dotenv()




# ── Liveness Detection (dlib 68-point landmarks)─

_LANDMARK_PATH = os.path.join(os.path.dirname(__file__), "shape_predictor_68_face_landmarks.dat")
try:
    _dlib_detector = dlib.get_frontal_face_detector()
    _dlib_predictor = dlib.shape_predictor(_LANDMARK_PATH)
except Exception as _e:
    _dlib_detector = None
    _dlib_predictor = None
    print(f"[liveness] Could not load landmark model: {_e}")

_LEFT_EYE  = list(range(42, 48))
_RIGHT_EYE = list(range(36, 42))
EAR_OPEN_THRESHOLD   = 0.25
EAR_CLOSED_THRESHOLD = 0.20


def _eye_aspect_ratio(landmarks, eye_idx):
    pts = [(landmarks.part(i).x, landmarks.part(i).y) for i in eye_idx]
    a = dist_metrics.euclidean(pts[1], pts[5])
    b = dist_metrics.euclidean(pts[2], pts[4])
    c = dist_metrics.euclidean(pts[0], pts[3])
    if c == 0:
        return 0.0
    return (a + b) / (2.0 * c)


def _avg_ear(rgb_img):
    """Average EAR for the largest detected face, or None."""
    if _dlib_detector is None or _dlib_predictor is None:
        return None
    gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
    faces = _dlib_detector(gray, 0)
    if len(faces) == 0:
        return None
    face = max(faces, key=lambda r: r.width() * r.height())
    shape = _dlib_predictor(gray, face)
    left = _eye_aspect_ratio(shape, _LEFT_EYE)
    right = _eye_aspect_ratio(shape, _RIGHT_EYE)
    return (left + right) / 2.0


# ── Helpers ───────────────────────────────────────────────────────

def decode_face_image(base64_string):
    if ',' in base64_string:
        base64_string = base64_string.split(',', 1)[1]
    image_bytes = base64.b64decode(base64_string)
    try:
        from PIL import Image
        import io
        pil_img = Image.open(io.BytesIO(image_bytes))
        pil_img = pil_img.convert('RGB')
        rgb_img = np.array(pil_img, dtype=np.uint8)
        return rgb_img, image_bytes
    except ImportError:
        pass
    nparr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rgb_img = np.ascontiguousarray(rgb_img, dtype=np.uint8)
    return rgb_img, image_bytes


def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return 6371000 * c


def generate_session_code():
    return ''.join(random.choices(string.digits, k=6))


def generate_default_password(length=10):
    """Random alphanumeric password for new lecturer accounts."""
    chars = string.ascii_letters + string.digits
    return ''.join(random.choices(chars, k=length))


from email.message import EmailMessage

def send_reset_password(to_email, name, temp_password, role):
    if not os.environ.get("SMTP_MAIL") or not os.environ.get("SMTP_PASSWORD"):
        return False, "SMTP not configured."
    msg = EmailMessage()
    msg["Subject"] = "Your FaceAttend Password Has Been Reset"
    msg["From"] = os.environ.get("SMTP_MAIL")
    msg["To"] = to_email
    msg.set_content(
        f"Hello {name},\n\n"
        f"An admin has reset your FaceAttend password.\n\n"
        f"Temporary password: {temp_password}\n\n"
        f"You will be required to set a new password on your next login.\n\n"
        f"— FaceAttend, Redeemer's University"
    )
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=10) as connection:
            connection.login(user=os.environ.get("SMTP_MAIL"), password=os.environ.get("SMTP_PASSWORD"))
            connection.send_message(msg)
        return True, None
    except Exception as e:
        return False, str(e)


def send_lecturer_credentials(to_email, name, default_password):
    if not os.environ.get("SMTP_MAIL") or not os.environ.get("SMTP_PASSWORD"):
        return False, "SMTP not configured. Set SMTP_MAIL and SMTP_PASSWORD."
    msg = EmailMessage()
    msg["Subject"] = "Your FaceAttend Lecturer Account"
    msg["From"] = os.environ.get("SMTP_MAIL")
    msg["To"] = to_email
    msg.set_content(
        f"Hello {name},\n\n"
        f"An admin has created a FaceAttend lecturer account for you.\n\n"
        f"Login email:    {to_email}\n"
        f"Default password: {default_password}\n\n"
        f"You will be required to set a new password on your first login.\n\n"
        f"— FaceAttend, Redeemer's University"
    )
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=10) as connection:
            connection.login(user=os.environ.get("SMTP_MAIL"), password=os.environ.get("SMTP_PASSWORD"))
            connection.send_message(msg)
        return True, None
    except Exception as e:
        return False, str(e)


# ── App Configuration ────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY")
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get("DATABASE_URL")
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'static', 'uploads', 'faces')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['GEO_RADIUS_METERS'] = 100

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
db = SQLAlchemy(app)


# ══════════════════════════════════════════════════════════════════
# MODELS
# ══════════════════════════════════════════════════════════════════

class Admin(db.Model):
    __tablename__ = 'admin'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

class Lecturer(db.Model):
    __tablename__ = 'lecturer'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    department = db.Column(db.String(120), nullable=True)
    password = db.Column(db.String(200), nullable=False)
    must_reset = db.Column(db.Boolean, default=True, nullable=False)
    courses = db.relationship('Course', backref='lecturer', lazy=True)

class Student(db.Model):
    __tablename__ = 'student'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    matric = db.Column(db.String(50), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    must_reset = db.Column(db.Boolean, default=False, nullable=False)
    face_encoding = db.Column(db.LargeBinary, nullable=True)
    enrollments = db.relationship('Enrollment', backref='student', lazy=True, cascade='all, delete-orphan')
    attendances = db.relationship('Attendance', backref='student', lazy=True, cascade='all, delete-orphan')

class Course(db.Model):
    __tablename__ = 'course'
    id = db.Column(db.Integer, primary_key=True)
    code = db.Column(db.String(20), unique=True, nullable=False)
    title = db.Column(db.String(200), nullable=False)
    lecturer_id = db.Column(db.Integer, db.ForeignKey('lecturer.id'), nullable=False)
    is_active = db.Column(db.Boolean, default=True, nullable=False)
    enrollments = db.relationship('Enrollment', backref='course', lazy=True, cascade='all, delete-orphan')
    attendances = db.relationship('Attendance', backref='course', lazy=True, cascade='all, delete-orphan')
    sessions = db.relationship('AttendanceSession', backref='course', lazy=True, cascade='all, delete-orphan')

class Enrollment(db.Model):
    __tablename__ = 'enrollment'
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey('student.id'), nullable=False)
    course_id = db.Column(db.Integer, db.ForeignKey('course.id'), nullable=False)
    __table_args__ = (db.UniqueConstraint('student_id', 'course_id'),)

class AttendanceSession(db.Model):
    __tablename__ = 'attendance_session'
    id = db.Column(db.Integer, primary_key=True)
    course_id = db.Column(db.Integer, db.ForeignKey('course.id'), nullable=False)
    lecturer_id = db.Column(db.Integer, db.ForeignKey('lecturer.id'), nullable=False)
    code = db.Column(db.String(6), nullable=False)
    is_open = db.Column(db.Boolean, default=True, nullable=False)
    latitude = db.Column(db.Float, nullable=True)
    longitude = db.Column(db.Float, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    closed_at = db.Column(db.DateTime, nullable=True)
    lecturer_ref = db.relationship('Lecturer', backref='sessions_created')
    attendances = db.relationship('Attendance', backref='att_session', lazy=True)

class Attendance(db.Model):
    __tablename__ = 'attendance'
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey('student.id'), nullable=False)
    course_id = db.Column(db.Integer, db.ForeignKey('course.id'), nullable=False)
    session_id = db.Column(db.Integer, db.ForeignKey('attendance_session.id'), nullable=True)
    lecturer_id = db.Column(db.Integer, db.ForeignKey('lecturer.id'), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    revoked = db.Column(db.Boolean, default=False, nullable=False)
    lecturer_ref = db.relationship('Lecturer', backref='attendances_recorded')





# ══════════════════════════════════════════════════════════════════
# AUTH DECORATORS
# ══════════════════════════════════════════════════════════════════

def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if session.get('role') != 'admin':
            flash('Admin login required.', 'error')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

def lecturer_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if session.get('role') != 'lecturer':
            flash('Lecturer login required.', 'error')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

def student_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if session.get('role') != 'student':
            flash('Student login required.', 'error')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated


# ══════════════════════════════════════════════════════════════════
# UNIFIED LOGIN / LOGOUT
# ══════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    role = session.get('role')
    if role == 'admin':
        return redirect(url_for('admin_dashboard'))
    elif role == 'lecturer':
        return redirect(url_for('lecturer_dashboard'))
    elif role == 'student':
        return redirect(url_for('student_dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if session.get('role'):
        return redirect(url_for('index'))
    if request.method == 'POST':
        identifier = request.form.get('identifier', '').strip()
        password = request.form.get('password', '')
        if not identifier or not password:
            flash('Please enter your credentials.', 'error')
            return redirect(url_for('login'))

        # Admin
        admin = Admin.query.filter_by(username=identifier).first()
        if admin and check_password_hash(admin.password, password):
            session.clear()
            session['admin_id'] = admin.id
            session['role'] = 'admin'
            session['user_name'] = 'Admin'
            return redirect(url_for('admin_dashboard'))

        # Lecturer
        lecturer = Lecturer.query.filter_by(email=identifier.lower()).first()
        if lecturer and check_password_hash(lecturer.password, password):
            session.clear()
            session['lecturer_id'] = lecturer.id
            session['lecturer_name'] = lecturer.name
            session['role'] = 'lecturer'
            session['user_name'] = lecturer.name
            if lecturer.must_reset:
                flash('Please set a new password before continuing.', 'info')
                return redirect(url_for('lecturer_reset_password'))
            return redirect(url_for('lecturer_dashboard'))

        # Student
        student = Student.query.filter_by(matric=identifier.upper()).first()
        if not student:
            student = Student.query.filter_by(matric=identifier).first()
        if student and check_password_hash(student.password, password):
            session.clear()
            session['student_id'] = student.id
            session['student_name'] = student.name
            session['student_matric'] = student.matric
            session['role'] = 'student'
            session['user_name'] = student.name
            if student.must_reset:
                flash('An admin has reset your password. Please set a new one to continue.', 'info')
                return redirect(url_for('student_reset_password'))
            return redirect(url_for('student_dashboard'))

        flash('Invalid credentials.', 'error')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


# ══════════════════════════════════════════════════════════════════
# PUBLIC SIGNUP (Student only — lecturers are admin-provisioned)
# ══════════════════════════════════════════════════════════════════

@app.route('/signup/student', methods=['GET', 'POST'])
def signup_student():
    if session.get('role'):
        return redirect(url_for('index'))
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        matric = request.form.get('matric', '').strip().upper()
        password = request.form.get('password', '')

        if not all([name, matric, password]):
            flash('Name, matric number, and password are required.', 'error')
            return redirect(url_for('signup_student'))
        if Student.query.filter_by(matric=matric).first():
            flash('Matric number already registered.', 'error')
            return redirect(url_for('signup_student'))

        student = Student(name=name, matric=matric, password=generate_password_hash(password))
        db.session.add(student)
        db.session.commit()

        # Auto-login and prompt face enrollment
        session.clear()
        session['student_id'] = student.id
        session['student_name'] = student.name
        session['student_matric'] = student.matric
        session['role'] = 'student'
        session['user_name'] = student.name
        flash('Account created. Please enroll your face to enable attendance.', 'success')
        return redirect(url_for('student_enroll_face'))
    return render_template('signup_student.html')


# ══════════════════════════════════════════════════════════════════
# ADMIN
# ══════════════════════════════════════════════════════════════════

@app.route('/admin/dashboard')
@admin_required
def admin_dashboard():
    return render_template('admin_dashboard.html',
        total_students=Student.query.count(),
        total_lecturers=Lecturer.query.count()
    )

@app.route('/admin/students')
@admin_required
def admin_students():
    students = Student.query.order_by(Student.name).all()
    return render_template('admin_students.html', students=students)

@app.route('/admin/students/<int:sid>/delete', methods=['POST'])
@admin_required
def admin_delete_student(sid):
    student = Student.query.get_or_404(sid)
    safe_matric = student.matric.replace('/', '_').replace('\\', '_')
    photo = os.path.join(app.config['UPLOAD_FOLDER'], f'{safe_matric}.jpg')
    if os.path.exists(photo):
        os.remove(photo)
    db.session.delete(student)
    db.session.commit()
    flash('Student deleted.', 'success')
    return redirect(url_for('admin_students'))

@app.route('/admin/lecturers')
@admin_required
def admin_lecturers():
    lecturers = Lecturer.query.order_by(Lecturer.name).all()
    return render_template('admin_lecturers.html', lecturers=lecturers)

@app.route('/admin/lecturers/create', methods=['GET', 'POST'])
@admin_required
def admin_create_lecturer():
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip().lower()
        department = request.form.get('department', '').strip()

        if not name or not email:
            flash('Name and email are required.', 'error')
            return redirect(url_for('admin_create_lecturer'))
        if Lecturer.query.filter_by(email=email).first():
            flash('A lecturer with that email already exists.', 'error')
            return redirect(url_for('admin_create_lecturer'))

        default_password = generate_default_password()
        lecturer = Lecturer(
            name=name,
            email=email,
            department=department or None,
            password=generate_password_hash(default_password),
            must_reset=True,
        )
        db.session.add(lecturer)
        db.session.commit()

        sent, err = send_lecturer_credentials(email, name, default_password)
        if sent:
            flash(f'Lecturer {name} created. Credentials emailed to {email}.', 'success')
        else:
            # Account was still created — surface the password so the admin can hand it over manually.
            flash(
                f'Lecturer {name} created, but email failed: {err}. '
                f'Default password: {default_password}',
                'error',
            )
        return redirect(url_for('admin_lecturers'))

    return render_template('admin_create_lecturer.html')

@app.route('/admin/lecturers/<int:lid>/delete', methods=['POST'])
@admin_required
def admin_delete_lecturer(lid):
    lecturer = Lecturer.query.get_or_404(lid)
    db.session.delete(lecturer)
    db.session.commit()
    flash('Lecturer deleted.', 'success')
    return redirect(url_for('admin_lecturers'))

@app.route("/reset-password", methods=["GET", "POST"])
def reset_password_request():
    if request.method == "POST":
        matric_or_email = request.form.get("matric_or_email", "").strip()
        receiving_gmail = request.form.get("receiving_gmail", "").strip()
        timestamp       = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

        if not os.environ.get("SMTP_MAIL") or not os.environ.get("SMTP_PASSWORD"):
            flash("SMTP not configured. Contact the administrator directly.", "error")
            return redirect(url_for("reset_password_request"))

        msg = EmailMessage()
        msg["Subject"] = "[FaceAttend] Password Reset Request"
        msg["From"]    = os.environ.get("SMTP_MAIL")
        msg["To"]      = os.environ.get("SMTP_MAIL")  # goes to the admin inbox
        msg.set_content(
            f"A user has submitted a password reset request.\n\n"
            f"Matric / Email  : {matric_or_email}\n"
            f"Receiving Gmail : {receiving_gmail}\n"
            f"Timestamp       : {timestamp} UTC\n\n"
            f"Please log in to the FaceAttend admin dashboard and use the\n"
            f"password reset form to generate a temporary password, then\n"
            f"send it to the receiving Gmail address above.\n\n"
            f"— FaceAttend, Redeemer's University"
        )

        try:
            with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=10) as connection:
                connection.login(user=os.environ.get("SMTP_MAIL"), password=os.environ.get("SMTP_PASSWORD"))
                connection.send_message(msg)
            flash("Reset request sent. Await your new password via Gmail.", "success")
        except Exception as e:
            app.logger.error(f"Reset email failed: {e}")
            flash("Failed to send request. Try again later.", "error")

        return redirect(url_for("login"))

    return render_template("form.html")

@app.route('/admin/reset-password', methods=['GET', 'POST'])
@admin_required
def admin_reset_password():
    if request.method == 'POST':
        identifier = request.form.get('identifier', '').strip()
        delivery_email = request.form.get('delivery_email', '').strip().lower()
        user_type = request.form.get('user_type', '')

        if not identifier or not delivery_email or user_type not in ('student', 'lecturer'):
            flash('All fields are required.', 'error')
            return redirect(url_for('admin_reset_password'))

        temp_password = generate_default_password()

        if user_type == 'student':
            user = Student.query.filter_by(matric=identifier.upper()).first()
            if not user:
                flash(f'No student found with matric number "{identifier}".', 'error')
                return redirect(url_for('admin_reset_password'))
            user.password = generate_password_hash(temp_password)
            user.must_reset = True
            db.session.commit()
            sent, err = send_reset_password(delivery_email, user.name, temp_password, 'student')
        else:
            user = Lecturer.query.filter_by(email=identifier.lower()).first()
            if not user:
                flash(f'No lecturer found with email "{identifier}".', 'error')
                return redirect(url_for('admin_reset_password'))
            user.password = generate_password_hash(temp_password)
            user.must_reset = True
            db.session.commit()
            sent, err = send_reset_password(delivery_email, user.name, temp_password, 'lecturer')

        if sent:
            flash(f'Password reset for {user.name}. Temporary password sent to {delivery_email}.', 'success')
        else:
            flash(
                f'Password reset for {user.name}, but email failed: {err}. '
                f'Temporary password: {temp_password}',
                'error',
            )
        return redirect(url_for('admin_reset_password'))

    return render_template('admin_reset_password.html')


# ══════════════════════════════════════════════════════════════════
# LECTURER
# ══════════════════════════════════════════════════════════════════

@app.before_request
def force_password_reset():
    """Redirect any user whose must_reset flag is set to the appropriate
    change-password page before they can access anything else."""
    role = session.get('role')

    if role == 'lecturer':
        lecturer = Lecturer.query.get(session.get('lecturer_id'))
        if not lecturer or not lecturer.must_reset:
            return
        allowed = {'lecturer_reset_password', 'logout', 'static'}
        if request.endpoint not in allowed:
            flash('Please set a new password before continuing.', 'info')
            return redirect(url_for('lecturer_reset_password'))

    elif role == 'student':
        student = Student.query.get(session.get('student_id'))
        if not student or not student.must_reset:
            return
        allowed = {'student_reset_password', 'logout', 'static'}
        if request.endpoint not in allowed:
            flash('An admin has reset your password. Please set a new one to continue.', 'info')
            return redirect(url_for('student_reset_password'))



@app.route('/lecturer/reset-password', methods=['GET', 'POST'])
@lecturer_required
def lecturer_reset_password():
    lecturer = Lecturer.query.get(session['lecturer_id'])
    if request.method == 'POST':
        current = request.form.get('current_password', '')
        new = request.form.get('new_password', '')
        confirm = request.form.get('confirm_password', '')

        if not all([current, new, confirm]):
            flash('All fields are required.', 'error')
            return redirect(url_for('lecturer_reset_password'))
        if not check_password_hash(lecturer.password, current):
            flash('Current password is incorrect.', 'error')
            return redirect(url_for('lecturer_reset_password'))
        if new != confirm:
            flash('New password and confirmation do not match.', 'error')
            return redirect(url_for('lecturer_reset_password'))
        if len(new) < 6:
            flash('New password must be at least 6 characters.', 'error')
            return redirect(url_for('lecturer_reset_password'))
        if new == current:
            flash('New password must be different from the default password.', 'error')
            return redirect(url_for('lecturer_reset_password'))

        lecturer.password = generate_password_hash(new)
        lecturer.must_reset = False
        db.session.commit()
        flash('Password updated. Welcome to FaceAttend.', 'success')
        return redirect(url_for('lecturer_dashboard'))

    return render_template('lecturer_reset_password.html', first_time=lecturer.must_reset)


@app.route('/lecturer/dashboard')
@lecturer_required
def lecturer_dashboard():
    lecturer = Lecturer.query.get(session['lecturer_id'])
    courses = Course.query.filter_by(lecturer_id=lecturer.id).all()
    active = [c for c in courses if c.is_active]
    terminated = [c for c in courses if not c.is_active]
    open_sessions = AttendanceSession.query.filter_by(lecturer_id=lecturer.id, is_open=True).all()
    return render_template('lecturer_dashboard.html',
        lecturer=lecturer, active_courses=active,
        terminated_courses=terminated, open_sessions=open_sessions
    )

@app.route('/lecturer/courses/create', methods=['GET', 'POST'])
@lecturer_required
def create_course():
    if request.method == 'POST':
        code = request.form.get('code', '').strip().upper()
        title = request.form.get('title', '').strip()
        if not code or not title:
            flash('Course code and title are required.', 'error')
            return redirect(url_for('create_course'))
        if Course.query.filter_by(code=code).first():
            flash('Course code already exists.', 'error')
            return redirect(url_for('create_course'))
        course = Course(code=code, title=title, lecturer_id=session['lecturer_id'], is_active=True)
        db.session.add(course)
        db.session.commit()
        flash(f'Course {code} created.', 'success')
        return redirect(url_for('lecturer_dashboard'))
    return render_template('create_course.html')

@app.route('/lecturer/courses/<int:cid>/terminate', methods=['POST'])
@lecturer_required
def terminate_course(cid):
    course = Course.query.get_or_404(cid)
    if course.lecturer_id != session['lecturer_id']:
        flash('Unauthorized.', 'error')
        return redirect(url_for('lecturer_dashboard'))
    course.is_active = False
    db.session.commit()
    flash(f'Course {course.code} terminated.', 'success')
    return redirect(url_for('lecturer_dashboard'))

@app.route('/lecturer/courses/<int:cid>/reactivate', methods=['POST'])
@lecturer_required
def reactivate_course(cid):
    course = Course.query.get_or_404(cid)
    if course.lecturer_id != session['lecturer_id']:
        flash('Unauthorized.', 'error')
        return redirect(url_for('lecturer_dashboard'))
    course.is_active = True
    db.session.commit()
    flash(f'Course {course.code} reactivated.', 'success')
    return redirect(url_for('lecturer_dashboard'))

# Roster (read-only — students self-enroll)
@app.route('/lecturer/courses/<int:cid>/roster')
@lecturer_required
def view_roster(cid):
    course = Course.query.get_or_404(cid)
    if course.lecturer_id != session['lecturer_id']:
        flash('Unauthorized.', 'error')
        return redirect(url_for('lecturer_dashboard'))
    enrollments = Enrollment.query.filter_by(course_id=course.id).all()
    return render_template('view_roster.html', course=course, enrollments=enrollments)

# Sessions
@app.route('/lecturer/session/start/<int:cid>', methods=['POST'])
@lecturer_required
def start_session(cid):
    course = Course.query.get_or_404(cid)
    if course.lecturer_id != session['lecturer_id']:
        return jsonify({'success': False, 'message': 'Unauthorized'}), 403
    # Close existing
    old = AttendanceSession.query.filter_by(course_id=course.id, is_open=True).first()
    if old:
        old.is_open = False
        old.closed_at = datetime.utcnow()

    lat = request.form.get('latitude', type=float)
    lon = request.form.get('longitude', type=float)
    code = generate_session_code()
    new_sess = AttendanceSession(course_id=course.id, lecturer_id=session['lecturer_id'],
                                  code=code, latitude=lat, longitude=lon, is_open=True)
    db.session.add(new_sess)
    db.session.commit()
    return jsonify({'success': True, 'session_id': new_sess.id, 'code': code})

@app.route('/lecturer/session/<int:sid>/end', methods=['POST'])
@lecturer_required
def end_session(sid):
    s = AttendanceSession.query.get_or_404(sid)
    if s.lecturer_id != session['lecturer_id']:
        flash('Unauthorized.', 'error')
        return redirect(url_for('lecturer_dashboard'))
    s.is_open = False
    s.closed_at = datetime.utcnow()
    db.session.commit()
    flash('Session closed.', 'success')
    return redirect(url_for('lecturer_dashboard'))

@app.route('/lecturer/session/<int:sid>/live')
@lecturer_required
def live_session(sid):
    s = AttendanceSession.query.get_or_404(sid)
    if s.lecturer_id != session['lecturer_id']:
        flash('Unauthorized.', 'error')
        return redirect(url_for('lecturer_dashboard'))
    attendances = Attendance.query.filter_by(session_id=sid, revoked=False).order_by(Attendance.timestamp.desc()).all()
    return render_template('live_session.html', att_session=s, attendances=attendances)

@app.route('/lecturer/session/<int:sid>/revoke/<int:aid>', methods=['POST'])
@lecturer_required
def revoke_attendance(sid, aid):
    att = Attendance.query.get_or_404(aid)
    s = AttendanceSession.query.get(sid)
    if not s or s.lecturer_id != session['lecturer_id'] or att.session_id != sid:
        return jsonify({'success': False}), 403
    att.revoked = True
    db.session.commit()
    return jsonify({'success': True, 'message': f'Revoked for {att.student.name}'})

@app.route('/api/session/<int:sid>/attendees')
@lecturer_required
def session_attendees(sid):
    s = AttendanceSession.query.get_or_404(sid)
    if s.lecturer_id != session['lecturer_id']:
        return jsonify({'error': 'Unauthorized'}), 403
    atts = Attendance.query.filter_by(session_id=sid, revoked=False).order_by(Attendance.timestamp.desc()).all()
    return jsonify({'count': len(atts), 'attendees': [
        {'id': a.id, 'name': a.student.name, 'matric': a.student.matric, 'time': a.timestamp.strftime('%H:%M:%S')}
        for a in atts
    ]})

# Records
    
@app.route('/lecturer/attendance')
@lecturer_required
def lecturer_attendance():
    lid = session['lecturer_id']
    courses = Course.query.filter_by(lecturer_id=lid).all()
    cid = request.args.get('course', type=int)
    date_filter = request.args.get('date', '')

    records = []
    summary = []
    total_sessions = 0

    if cid:
        # raw date-filtered rows (second tab)
        q = Attendance.query.filter_by(course_id=cid, lecturer_id=lid, revoked=False)
        if date_filter:
            try:
                d = datetime.strptime(date_filter, '%Y-%m-%d').date()
                q = q.filter(db.func.date(Attendance.timestamp) == d)
            except ValueError:
                pass
        records = q.order_by(Attendance.timestamp.desc()).all()

        # per-student percentage summary (primary tab)
        total_sessions = AttendanceSession.query.filter_by(course_id=cid).count()
        enrollments = Enrollment.query.filter_by(course_id=cid).all()
        for e in enrollments:
            attended = (
                db.session.query(Attendance.session_id)
                .filter_by(student_id=e.student_id, course_id=cid, revoked=False)
                .filter(Attendance.session_id.isnot(None))
                .distinct()
                .count()
            )
            pct = round((attended / total_sessions) * 100, 1) if total_sessions else 0.0
            summary.append({
                'student': e.student,
                'attended': attended,
                'total': total_sessions,
                'percentage': pct,
            })
        summary.sort(key=lambda r: r['percentage'], reverse=True)

    return render_template('lecturer_attendance.html',
                           courses=courses, records=records, summary=summary,
                           total_sessions=total_sessions,
                           selected_course=cid, selected_date=date_filter)

@app.route('/lecturer/attendance/export')
@lecturer_required
def export_attendance():
    import csv, io
    cid = request.args.get('course', type=int)
    if not cid:
        return jsonify({'error': 'Course required'}), 400
    course = Course.query.get_or_404(cid)
    if course.lecturer_id != session['lecturer_id']:
        return jsonify({'error': 'Unauthorized'}), 403
    records = Attendance.query.filter_by(course_id=cid, lecturer_id=session['lecturer_id'], revoked=False).order_by(Attendance.timestamp.desc()).all()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Student Name', 'Matric No', 'Course Code', 'Course Title', 'Date/Time'])
    for r in records:
        writer.writerow([r.student.name, r.student.matric, r.course.code, r.course.title, r.timestamp.strftime('%Y-%m-%d %H:%M:%S')])
    return Response(output.getvalue(), mimetype='text/csv',
                    headers={'Content-Disposition': f'attachment; filename=attendance_{course.code}.csv'})


# ══════════════════════════════════════════════════════════════════
# STUDENT
# ══════════════════════════════════════════════════════════════════

@app.route('/student/reset-password', methods=['GET', 'POST'])
@student_required
def student_reset_password():
    student = Student.query.get(session['student_id'])
    if request.method == 'POST':
        current = request.form.get('current_password', '')
        new = request.form.get('new_password', '')
        confirm = request.form.get('confirm_password', '')

        if not all([current, new, confirm]):
            flash('All fields are required.', 'error')
            return redirect(url_for('student_reset_password'))
        if not check_password_hash(student.password, current):
            flash('Temporary password is incorrect.', 'error')
            return redirect(url_for('student_reset_password'))
        if new != confirm:
            flash('New password and confirmation do not match.', 'error')
            return redirect(url_for('student_reset_password'))
        if len(new) < 6:
            flash('New password must be at least 6 characters.', 'error')
            return redirect(url_for('student_reset_password'))
        if new == current:
            flash('New password must be different from the temporary password.', 'error')
            return redirect(url_for('student_reset_password'))

        student.password = generate_password_hash(new)
        student.must_reset = False
        db.session.commit()
        flash('Password updated. Welcome back.', 'success')
        return redirect(url_for('student_dashboard'))

    return render_template('student_reset_password.html')



@app.route('/student/dashboard')
@student_required
def student_dashboard():
    student = Student.query.get(session['student_id'])
    enrollments = Enrollment.query.filter_by(student_id=student.id).all()
    course_stats = []
    for e in enrollments:
        c = e.course
        total = AttendanceSession.query.filter_by(course_id=c.id).count()
        attended = Attendance.query.filter_by(student_id=student.id, course_id=c.id, revoked=False).count()
        pct = round(attended / total * 100, 1) if total > 0 else 0
        course_stats.append({'course': c, 'attended': attended, 'total': total, 'percentage': pct})
    face_enrolled = student.face_encoding is not None
    return render_template('student_dashboard.html', student=student,
                           course_stats=course_stats, face_enrolled=face_enrolled)

@app.route('/student/enroll-face', methods=['GET', 'POST'])
@student_required
def student_enroll_face():
    student = Student.query.get(session['student_id'])
    if request.method == 'POST':
        face_image = request.form.get('face_image', '')
        if not face_image:
            flash('Face capture is required.', 'error')
            return redirect(url_for('student_enroll_face'))
        try:
            rgb_img, img_bytes = decode_face_image(face_image)
            locs = face_recognition.face_locations(rgb_img)
            if not locs:
                flash('No face detected. Try better lighting.', 'error')
                return redirect(url_for('student_enroll_face'))
            if len(locs) > 1:
                flash('Multiple faces detected. Capture alone.', 'error')
                return redirect(url_for('student_enroll_face'))
            new_enc = face_recognition.face_encodings(rgb_img, locs)[0]
        except Exception as e:
            flash(f'Face processing error: {str(e)}', 'error')
            return redirect(url_for('student_enroll_face'))

        # ── Face uniqueness check ─────────────────────────────────
        # Compare against every other student's stored encoding.
        DUPLICATE_THRESHOLD = 0.45  # same threshold used in verify_attendance

        others = Student.query.filter(
            Student.id != student.id,
            Student.face_encoding.isnot(None)
        ).all()

        if others:
            known_encodings = [
                np.frombuffer(s.face_encoding, dtype=np.float64) for s in others
            ]
            distances = face_recognition.face_distance(known_encodings, new_enc)
            min_idx = int(np.argmin(distances))
            min_dist = float(distances[min_idx])

            if min_dist < DUPLICATE_THRESHOLD:
                matched = others[min_idx]
                flash(
                    f'This face is already registered under matric '
                    f'{matched.matric}. Each student must enroll their own face.',
                    'error'
                )
                return redirect(url_for('student_enroll_face'))
        # ──────────────────────────────────────────────────────────

        student.face_encoding = new_enc.tobytes()
        db.session.commit()

        safe_matric = student.matric.replace('/', '_').replace('\\', '_')
        photo_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{safe_matric}.jpg')
        with open(photo_path, 'wb') as f:
            f.write(img_bytes)

        flash('Face enrolled. You can now mark attendance.', 'success')
        return redirect(url_for('student_dashboard'))
    already_enrolled = student.face_encoding is not None
    return render_template('student_enroll_face.html', already_enrolled=already_enrolled)

@app.route('/student/courses')
@student_required
def student_courses():
    student = Student.query.get(session['student_id'])
    enrolled_ids = {e.course_id for e in Enrollment.query.filter_by(student_id=student.id).all()}
    active_courses = Course.query.filter_by(is_active=True).order_by(Course.code).all()
    return render_template('student_courses.html',
                           courses=active_courses, enrolled_ids=enrolled_ids)


@app.route('/student/courses/<int:cid>/enroll', methods=['POST'])
@student_required
def student_self_enroll(cid):
    course = Course.query.get_or_404(cid)
    if not course.is_active:
        flash('This course is not active.', 'error')
        return redirect(url_for('student_courses'))
    sid = session['student_id']
    if Enrollment.query.filter_by(student_id=sid, course_id=cid).first():
        flash(f'Already enrolled in {course.code}.', 'error')
    else:
        db.session.add(Enrollment(student_id=sid, course_id=cid))
        db.session.commit()
        flash(f'Enrolled in {course.code}.', 'success')
    return redirect(url_for('student_courses'))


@app.route('/student/courses/<int:cid>/unenroll', methods=['POST'])
@student_required
def student_self_unenroll(cid):
    sid = session['student_id']
    e = Enrollment.query.filter_by(student_id=sid, course_id=cid).first_or_404()
    course = Course.query.get(cid)
    db.session.delete(e)
    db.session.commit()
    flash(f'Unenrolled from {course.code}.', 'success')
    return redirect(url_for('student_courses'))

@app.route('/student/mark-attendance')
@student_required
def student_mark_attendance():
    student = Student.query.get(session['student_id'])
    if not student.face_encoding:
        flash('Please enroll your face before marking attendance.', 'error')
        return redirect(url_for('student_enroll_face'))
    return render_template('student_mark_attendance.html')

@app.route('/api/verify-attendance', methods=['POST'])
@student_required
def verify_attendance():
    data = request.get_json()
    if not data:
        return jsonify({'success': False, 'message': 'No data received.'}), 400

    code = data.get('session_code', '').strip()
    face_image = data.get('image', '')          # eyes OPEN frame (identity)
    blink_image = data.get('blink_image', '')   # eyes CLOSED frame (liveness)
    slat = data.get('latitude')
    slon = data.get('longitude')

    if not code:
        return jsonify({'success': False, 'message': 'Enter the session code.'}), 400
    if not face_image or not blink_image:
        return jsonify({'success': False, 'message': 'Both face captures required (open + blink).'}), 400

    att_session = AttendanceSession.query.filter_by(code=code, is_open=True).first()
    if not att_session:
        return jsonify({'success': False, 'message': 'Invalid or expired session code.'}), 400

    course = Course.query.get(att_session.course_id)
    student = Student.query.get(session['student_id'])

    if not Enrollment.query.filter_by(student_id=student.id, course_id=course.id).first():
        return jsonify({'success': False, 'message': f'You are not enrolled in {course.code}.'}), 400

    # Geolocation check
    if att_session.latitude is not None and att_session.longitude is not None:
        if slat is None or slon is None:
            return jsonify({'success': False, 'message': 'Location access required. Enable GPS.'}), 400
        dist = haversine(att_session.latitude, att_session.longitude, slat, slon)
        if dist > app.config['GEO_RADIUS_METERS']:
            return jsonify({'success': False, 'message': f'Too far from class ({int(dist)}m). Move closer.'}), 400

    # Decode both frames
    try:
        open_rgb, _ = decode_face_image(face_image)
        blink_rgb, _ = decode_face_image(blink_image)
    except Exception as e:
        return jsonify({'success': False, 'message': f'Image error: {str(e)}'}), 400

    # ---- LIVENESS: blink detection BEFORE face matching ----
    if _dlib_predictor is None:
        return jsonify({'success': False, 'message': 'Liveness model unavailable. Contact admin.'}), 500

    open_ear = _avg_ear(open_rgb)
    blink_ear = _avg_ear(blink_rgb)
    if open_ear is None or blink_ear is None:
        return jsonify({'success': False, 'message': 'Could not detect eyes in one of the frames. Retry.'}), 400
    if open_ear < EAR_OPEN_THRESHOLD:
        return jsonify({'success': False, 'message': 'Keep your eyes open for the first capture.'}), 400
    if blink_ear > EAR_CLOSED_THRESHOLD:
        return jsonify({'success': False, 'message': 'Blink not detected. Please blink when prompted.'}), 400
    # eyes were open, then closed -> live person confirmed

    # ---- FACE MATCH on the open-eyes frame ----
    locs = face_recognition.face_locations(open_rgb)
    if len(locs) == 0:
        return jsonify({'success': False, 'message': 'No face detected.'}), 400
    if len(locs) > 1:
        return jsonify({'success': False, 'message': 'Multiple faces detected.'}), 400

    encs = face_recognition.face_encodings(open_rgb, locs)
    if not encs:
        return jsonify({'success': False, 'message': 'Could not process face.'}), 400

    if not student.face_encoding:
        return jsonify({'success': False, 'message': 'Face not enrolled. Enroll your face from your dashboard.'}), 400

    known = np.frombuffer(student.face_encoding, dtype=np.float64)
    d = face_recognition.face_distance([known], encs[0])[0]

    if d >= 0.45:
        return jsonify({'success': False, 'message': 'Face verification failed.'}), 400

    if Attendance.query.filter_by(student_id=student.id, session_id=att_session.id, revoked=False).first():
        return jsonify({'success': True, 'already_marked': True, 'message': 'Already recorded for this session.'})

    att = Attendance(student_id=student.id, course_id=course.id, session_id=att_session.id, lecturer_id=att_session.lecturer_id)
    db.session.add(att)
    db.session.commit()
    return jsonify({'success': True, 'already_marked': False,
                    'message': f'Attendance marked for {course.code}!',
                    'confidence': round((1 - d) * 100, 1)})

@app.route('/student/history')
@student_required
def student_history():
    student = Student.query.get(session['student_id'])
    enrollments = Enrollment.query.filter_by(student_id=student.id).all()
    courses = [e.course for e in enrollments]
    cid = request.args.get('course', type=int)
    records = []
    if cid:
        records = Attendance.query.filter_by(student_id=student.id, course_id=cid, revoked=False).order_by(Attendance.timestamp.desc()).all()
    return render_template('student_history.html', courses=courses, records=records, selected_course=cid)


# ══════════════════════════════════════════════════════════════════
# DB INIT
# ══════════════════════════════════════════════════════════════════

with app.app_context():
    db.create_all()
    if not Admin.query.filter_by(username='Beluga').first():
        db.session.add(Admin(username=os.environ.get("ADMIN_USERNAME"), password=generate_password_hash(os.environ.get("ADMIN_PASSWORD"))))
        db.session.commit()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
