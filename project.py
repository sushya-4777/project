# library_ml_app.py
"""
Library Management System with simple ML prediction - migrated & fixed

Features:
- Safe DB migration: adds 'code' columns to books & users if missing and populates existing rows
- Admin: add books, create students, assign/issue books, APPROVE REQUESTS, REMOVE BOOKS
- Students: view assigned loans, return books, REQUEST BOOKS
- Auto-generated codes: books b1,b2... and students s1,s2...
- ML: train RandomForest to predict days kept (uses historical returned transactions)
- Reports + CSV export
- NEW: Book Request workflow (Student requests, Admin approves/issues)
- NEW: Transaction status now shows 'overdue' based on predicted_days

Run:
pip install streamlit pandas scikit-learn joblib passlib
streamlit run library_ml_app.py
"""
import streamlit as st
import pandas as pd
import sqlite3
from datetime import datetime, date
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from passlib.hash import pbkdf2_sha256

DB_PATH = "library.db"
MODEL_PATH = "loan_days_model.joblib"


# ----------------- Database helpers -----------------
def get_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def table_has_column(conn, table_name: str, column_name: str) -> bool:
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table_name})")
    cols = [r[1] for r in cur.fetchall()]
    return column_name in cols


def migrate_add_code_columns(conn):
    """
    Add 'code' column to books and users if missing and populate for existing rows.
    Safe to run multiple times.
    """
    cur = conn.cursor()
    # BOOKS: add column if missing
    if not table_has_column(conn, "books", "code"):
        cur.execute("ALTER TABLE books ADD COLUMN code TEXT")
        cur.execute("SELECT id FROM books")
        rows = cur.fetchall()
        for r in rows:
            bid = r[0]
            cur.execute("UPDATE books SET code=? WHERE id=?", (f"b{bid}", bid))
        conn.commit()
    else:
        # fill nulls if any
        cur.execute("SELECT id, code FROM books")
        for r in cur.fetchall():
            if r[1] is None:
                cur.execute("UPDATE books SET code=? WHERE id=?", (f"b{r[0]}", r[0]))
        conn.commit()

    # USERS: add column if missing
    if not table_has_column(conn, "users", "code"):
        cur.execute("ALTER TABLE users ADD COLUMN code TEXT")
        cur.execute("SELECT id, role FROM users")
        rows = cur.fetchall()
        for r in rows:
            uid, role = r
            if role == "admin":
                code = "admin"
            else:
                code = f"s{uid}"
            cur.execute("UPDATE users SET code=? WHERE id=?", (code, uid))
        conn.commit()
    else:
        cur.execute("SELECT id, code, role FROM users")
        for r in cur.fetchall():
            uid, code, role = r
            if code is None:
                code = "admin" if role == "admin" else f"s{uid}"
                cur.execute("UPDATE users SET code=? WHERE id=?", (code, uid))
        conn.commit()


def init_db():
    conn = get_connection()
    cur = conn.cursor()
    # create base tables (do not assume code exists; migration will add/populate)
    cur.executescript(
        """
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        full_name TEXT,
        role TEXT CHECK(role IN ('admin','student')),
        password_hash TEXT
    );
    CREATE TABLE IF NOT EXISTS books (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT,
        author TEXT,
        total_copies INTEGER DEFAULT 1,
        available_copies INTEGER DEFAULT 1
    );
    CREATE TABLE IF NOT EXISTS transactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        book_id INTEGER,
        student_id INTEGER,
        issue_date TEXT,
        return_date TEXT,
        predicted_days INTEGER,
        actual_days INTEGER,
        status TEXT CHECK(status IN ('issued','returned')),
        FOREIGN KEY(book_id) REFERENCES books(id),
        FOREIGN KEY(student_id) REFERENCES users(id)
    );
    CREATE TABLE IF NOT EXISTS book_requests (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        book_id INTEGER,
        student_id INTEGER,
        request_date TEXT,
        status TEXT CHECK(status IN ('pending','approved','denied')),
        FOREIGN KEY(book_id) REFERENCES books(id),
        FOREIGN KEY(student_id) REFERENCES users(id),
        UNIQUE(book_id, student_id, status) -- Prevent duplicate pending requests
    );
    """
    )
    conn.commit()

    # ensure at least one admin exists
    cur.execute("SELECT COUNT(*) FROM users WHERE role='admin'")
    if cur.fetchone()[0] == 0:
        pw = pbkdf2_sha256.hash("admin123")
        cur.execute(
            "INSERT INTO users(username, full_name, role, password_hash) VALUES (?,?,?,?)",
            ("admin", "Administrator", "admin", pw),
        )
        conn.commit()

    # run migration to ensure 'code' fields present and populated
    migrate_add_code_columns(conn)
    conn.close()


# utilities to generate next code (avoid collisions by scanning existing codes)
def next_student_code(conn):
    cur = conn.cursor()
    cur.execute("SELECT code FROM users WHERE role='student' ORDER BY id DESC")
    rows = [r[0] for r in cur.fetchall() if r[0]]
    if not rows:
        return "s1"
    nums = []
    for c in rows:
        try:
            nums.append(int(str(c).lstrip("s")))
        except Exception:
            pass
    next_n = max(nums) + 1 if nums else len(rows) + 1
    return f"s{next_n}"


def next_book_code(conn):
    cur = conn.cursor()
    cur.execute("SELECT code FROM books ORDER BY id DESC")
    rows = [r[0] for r in cur.fetchall() if r[0]]
    if not rows:
        return "b1"
    nums = []
    for c in rows:
        try:
            nums.append(int(str(c).lstrip("b")))
        except Exception:
            pass
    next_n = max(nums) + 1 if nums else len(rows) + 1
    return f"b{next_n}"


# ----------------- Auth & user management -----------------
def verify_user(username, password):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE username=?", (username,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    # row includes 'code' after migration, 'role' etc.
    if row["password_hash"] and pbkdf2_sha256.verify(password, row["password_hash"]):
        return dict(row)
    return None


def create_student(username, full_name, password):
    conn = get_connection()
    cur = conn.cursor()
    pw = pbkdf2_sha256.hash(password)
    code = next_student_code(conn)
    try:
        cur.execute(
            "INSERT INTO users(code, username, full_name, role, password_hash) VALUES (?,?,?,?,?)",
            (code, username, full_name, "student", pw),
        )
        conn.commit()
        conn.close()
        return True, code
    except sqlite3.IntegrityError as e:
        conn.close()
        return False, str(e)


def admin_create_student(username, full_name, password):
    return create_student(username, full_name, password)


# ----------------- Book CRUD -----------------
def add_book(title, author, copies=1):
    conn = get_connection()
    cur = conn.cursor()
    code = next_book_code(conn)
    cur.execute(
        "INSERT INTO books(code, title, author, total_copies, available_copies) VALUES (?,?,?,?,?)",
        (code, title, author, copies, copies),
    )
    conn.commit()
    conn.close()
    return code


def remove_book(book_id):
    """
    Removes a book if it has no active issued loans.
    Also cleans up associated book requests.
    """
    conn = get_connection()
    cur = conn.cursor()
    # 1. Check for active issued transactions
    cur.execute("SELECT COUNT(*) FROM transactions WHERE book_id=? AND status='issued'", (book_id,))
    if cur.fetchone()[0] > 0:
        conn.close()
        return False, "Cannot remove book. It currently has active issued loans."

    # 2. Check if book exists
    cur.execute("SELECT id FROM books WHERE id=?", (book_id,))
    if cur.fetchone() is None:
        conn.close()
        return False, "Book not found."

    # 3. Delete related requests (cleanup)
    cur.execute("DELETE FROM book_requests WHERE book_id=?", (book_id,))
    # 4. Delete the book
    cur.execute("DELETE FROM books WHERE id=?", (book_id,))
    conn.commit()
    conn.close()
    return True, "Book successfully removed."


# ----------------- Models & helpers -----------------
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None


def save_model(model, enc_book=None, enc_student=None):
    obj = {"model": model, "enc_book": enc_book, "enc_student": enc_student}
    joblib.dump(obj, MODEL_PATH)


def prepare_training_data():
    conn = get_connection()
    df_tx = pd.read_sql_query("SELECT * FROM transactions WHERE actual_days IS NOT NULL", conn)
    conn.close()
    if df_tx.empty:
        return None
    df = df_tx.copy()
    df["issue_date"] = pd.to_datetime(df["issue_date"])
    df["dow"] = df["issue_date"].dt.dayofweek
    df["is_weekend"] = df["dow"].isin([5, 6]).astype(int)
    enc_book = LabelEncoder()
    enc_student = LabelEncoder()
    df["book_enc"] = enc_book.fit_transform(df["book_id"].astype(str))
    df["stu_enc"] = enc_student.fit_transform(df["student_id"].astype(str))
    stud_avg = df.groupby("student_id")["actual_days"].mean().to_dict()
    book_avg = df.groupby("book_id")["actual_days"].mean().to_dict()
    df["stu_avg_prev"] = df["student_id"].map(stud_avg)
    df["book_avg_prev"] = df["book_id"].map(book_avg)
    features = df[["book_enc", "stu_enc", "dow", "is_weekend", "stu_avg_prev", "book_avg_prev"]].fillna(0)
    target = df["actual_days"]
    return features, target, enc_book, enc_student


def train_and_save_model():
    data = prepare_training_data()
    if data is None:
        return False, "Not enough historical returned transactions to train a model"
    X, y, enc_book, enc_student = data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    save_model(model, enc_book, enc_student)
    score = model.score(X_test, y_test)
    return True, f"Model trained. R^2 on test set: {score:.3f}"


def predict_days_for_issue(book_id, student_id, issue_date_str):
    model_bundle = load_model()
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM transactions WHERE actual_days IS NOT NULL", conn)
    conn.close()
    # fallback default if no data or model
    if df.empty or model_bundle is None:
        return 7
    issue_date = pd.to_datetime(issue_date_str)
    dow = issue_date.dayofweek
    is_weekend = int(dow in [5, 6])
    stu_avg = df.groupby("student_id")["actual_days"].mean().to_dict().get(student_id, 7)
    book_avg = df.groupby("book_id")["actual_days"].mean().to_dict().get(book_id, 7)
    enc_book = model_bundle["enc_book"]
    enc_student = model_bundle["enc_student"]
    model = model_bundle["model"]
    try:
        b_enc = enc_book.transform([str(book_id)])[0]
    except Exception:
        b_enc = 0
    try:
        s_enc = enc_student.transform([str(student_id)])[0]
    except Exception:
        s_enc = 0
    feat = [[b_enc, s_enc, dow, is_weekend, stu_avg, book_avg]]
    pred = model.predict(feat)[0]
    return max(1, int(round(pred)))


# ----------------- Admin functions: manual issue/assign -----------------
def admin_issue_book_to_student(book_id, student_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT available_copies FROM books WHERE id=?", (book_id,))
    r = cur.fetchone()
    if not r or r[0] <= 0:
        conn.close()
        return False, "No copies available"
    issue_date = date.today().isoformat()
    predicted = predict_days_for_issue(book_id, student_id, issue_date)
    cur.execute(
        "INSERT INTO transactions(book_id, student_id, issue_date, status, predicted_days) VALUES (?,?,?,?,?)",
        (book_id, student_id, issue_date, "issued", predicted),
    )
    cur.execute("UPDATE books SET available_copies=available_copies-1 WHERE id=?", (book_id,))
    conn.commit()
    conn.close()
    return True, predicted

# ----------------- Request/Approval functions -----------------
def student_request_book(student_id, book_id):
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            "INSERT INTO book_requests(student_id, book_id, request_date, status) VALUES (?,?,?,?)",
            (student_id, book_id, date.today().isoformat(), "pending"),
        )
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        # User already has a pending request for this book
        conn.close()
        return False


def admin_approve_request(request_id):
    conn = get_connection()
    cur = conn.cursor()

    # Find book & student
    cur.execute("SELECT student_id, book_id FROM book_requests WHERE id=?", (request_id,))
    row = cur.fetchone()
    if not row:
        conn.close()
        return False, "Request not found"

    student_id, book_id = row

    # Assign book using existing function
    # NOTE: This function handles availability check and updates available_copies
    ok, msg = admin_issue_book_to_student(book_id, student_id)
    if not ok:
        conn.close()
        # Deny the request if the issue failed (e.g., no copies)
        cur.execute("UPDATE book_requests SET status='denied' WHERE id=?", (request_id,))
        conn.commit()
        return False, f"Could not issue book: {msg}. Request status set to 'denied'."

    # Update request
    cur.execute("UPDATE book_requests SET status='approved' WHERE id=?", (request_id,))
    conn.commit()
    conn.close()
    return True, f"Book issued. Predicted days: {msg}"


# ----------------- Initialize DB -----------------
init_db()


# ----------------- Streamlit UI -----------------
st.set_page_config(page_title="Library + ML", layout="wide")
st.title("Library Management System with ML models")

if "user" not in st.session_state:
    st.session_state.user = None

# Sidebar: Login / Create student
with st.sidebar:
    if st.session_state.user is None:
        st.subheader("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            user = verify_user(username, password)
            if user:
                st.session_state.user = user
                st.rerun()
            else:
                st.error("Invalid credentials")

        st.markdown("---")
        st.subheader("Create student account")
        newu = st.text_input("New username", key="newu")
        newname = st.text_input("Full name", key="newname")
        newpw = st.text_input("Password", type="password", key="newpw")
        if st.button("Create student"):
            ok, msg = create_student(newu, newname, newpw)
            if ok:
                st.success(f"Student created. Assigned student code: {msg}. Please login.")
            else:
                st.error(msg)
    else:
        st.markdown(f"**Signed in as:** {st.session_state.user.get('full_name')} ({st.session_state.user.get('role')})")
        if st.button("Logout"):
            st.session_state.user = None
            st.rerun()

# require login
if st.session_state.user is None:
    st.info("Please login or create a student account from the sidebar to continue.")
    st.stop()

user = st.session_state.user
is_admin = user.get("role") == "admin"

# Admin dashboard
if is_admin:
    st.header("Admin dashboard")
    # Tabs modification: Added 'Requests' at index 2
    tabs = st.tabs(["Books", "Transactions", "Requests", "Train Model", "Reports", "Users"])

    # Books
    with tabs[0]:
        st.subheader("Manage books")
        conn = get_connection()
        if st.button("Add sample books"):
            sample = [
                ("C Programming", "Dennis Ritchie", 3),
                ("Concrete Technology", "M.S. Shetty", 2),
                ("Machine Learning", "Tom Mitchell", 4),
                ("Atomic Habits", "James Clear", 3),
            ]
            codes = []
            for t, a, c in sample:
                codes.append(add_book(t, a, c))
            st.success("Sample books added: " + ", ".join(codes))
            st.rerun()

        with st.form("add_book_admin"):
            t = st.text_input("Title")
            a = st.text_input("Author")
            c = st.number_input("Copies", min_value=1, value=1)
            if st.form_submit_button("Add book"):
                code = add_book(t, a, c)
                st.success(f"Book added with code {code}")
                st.rerun()
        
        st.markdown("---")
        st.subheader("Remove a book")
        
        try:
            conn = get_connection()
            all_books = pd.read_sql_query("SELECT id, code, title FROM books", conn)
            conn.close()
            
            if all_books.empty:
                st.info("No books to remove.")
            else:
                # Need a unique key for the selectbox here
                remove_list = (all_books["title"] + " (" + all_books["code"] + ")").tolist()
                book_sel = st.selectbox("Select book to remove", remove_list, key="remove_book_sel")
                
                # Retrieve the ID of the selected book
                book_row = all_books[(all_books["title"] + " (" + all_books["code"] + ")") == book_sel].iloc[0]
                book_id_to_remove = int(book_row["id"])
                
                if st.button("Permanently Remove Book"):
                    ok, msg = remove_book(book_id_to_remove)
                    if ok:
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)
        except Exception as e:
            st.error("Error setting up remove book form: " + str(e))
            
        st.markdown("---")
        # show books table
        try:
            conn = get_connection()
            df = pd.read_sql_query("SELECT id, code, title, author, total_copies, available_copies FROM books", conn)
            conn.close()
            st.dataframe(df)
        except Exception as e:
            st.error("Error reading books: " + str(e))

    # Transactions (assign/issue)
    with tabs[1]:
        st.subheader("Transactions (Admin: manual assign/issue books)")
        st.markdown("Assign a book to a student. Students cannot self-issue — they will only see assigned loans.")
        conn = get_connection()
        books_df = pd.read_sql_query("SELECT id, code, title, available_copies FROM books WHERE available_copies>0", conn)
        users_df = pd.read_sql_query("SELECT id, code, full_name FROM users WHERE role='student'", conn)
        conn.close()

        with st.form("issue_form"):
            if books_df.empty:
                st.info("No books available to assign.")
            else:
                book_options = (books_df["title"] + " (" + books_df["code"] + ")").tolist()
                book_sel = st.selectbox("Select book (available)", book_options, key="issue_book_sel")
                book_row = books_df[(books_df["title"] + " (" + books_df["code"] + ")") == book_sel].iloc[0]

            if users_df.empty:
                st.info("No students registered.")
            else:
                stu_options = (users_df["full_name"] + " (" + users_df["code"] + ")").tolist()
                stu_sel = st.selectbox("Select student to assign", stu_options, key="issue_stu_sel")
                stu_row = users_df[(users_df["full_name"] + " (" + users_df["code"] + ")") == stu_sel].iloc[0]

            if st.form_submit_button("Assign book"):
                ok, res = admin_issue_book_to_student(int(book_row["id"]), int(stu_row["id"]))
                if ok:
                    st.success(f"Book assigned to {stu_row['full_name']}. Predicted days: {res}")
                else:
                    st.error(res)
                    st.rerun()

        st.markdown("---")
        try:
            conn = get_connection()
            # STEP 5 - Admin: Transactions/Reports: Overdue Logic
            tx = pd.read_sql_query(
                """
                SELECT 
                    t.id, b.code as book_code, b.title, u.code as student_code, u.full_name as student, 
                    t.issue_date, t.return_date, t.predicted_days, t.actual_days, 
                    CASE 
                        WHEN t.status='issued' 
                            AND julianday('now') - julianday(t.issue_date) > t.predicted_days 
                        THEN 'overdue' 
                        ELSE t.status 
                    END AS status
                FROM transactions t 
                LEFT JOIN books b ON t.book_id=b.id 
                LEFT JOIN users u ON t.student_id=u.id
                """,
                conn,
            )
            conn.close()
            if tx.empty:
                st.info("No transactions yet.")
            else:
                st.dataframe(tx)
        except Exception as e:
            st.error("Error reading transactions: " + str(e))

    # Requests
    # STEP 4 - Admin: New Requests Tab
    with tabs[2]:
        st.subheader("Book Requests from Students")

        conn = get_connection()
        req_df = pd.read_sql_query(
            "SELECT r.id, u.full_name AS student, b.title AS book, b.code AS book_code, "
            "r.request_date, r.status "
            "FROM book_requests r "
            "LEFT JOIN users u ON r.student_id=u.id "
            "LEFT JOIN books b ON r.book_id=b.id",
            conn,
        )
        conn.close()

        st.dataframe(req_df)

        pending = req_df[req_df["status"] == "pending"]
        if not pending.empty:
            st.subheader("Approve a request")
            options = (pending["student"] + " → " + pending["book"]).tolist()
            sel = st.selectbox("Select request", options)
            req_row = pending[(pending["student"] + " → " + pending["book"]) == sel].iloc[0]
            req_id = int(req_row["id"])

            if st.button("Approve Request"):
                ok, msg = admin_approve_request(req_id)
                if ok:
                    st.success("Approved. " + str(msg))
                    st.rerun()
                else:
                    st.error(msg)


    # Train model
    with tabs[3]: # Note: Index changed due to new 'Requests' tab
        st.subheader("Train ML model")
        st.write("Train a model to predict how many days a student will keep a book (requires past returns).")
        if st.button("Train model now"):
            ok, msg = train_and_save_model()
            if ok:
                st.success(msg)
            else:
                st.error(msg)
        if load_model():
            st.info("Model is available for predictions.")
        else:
            st.warning("No trained model found. Train if you have historical returned transactions.")

    # Reports
    with tabs[4]: # Note: Index changed due to new 'Requests' tab
        st.subheader("Reports & Export")
        try:
            conn = get_connection()
            # STEP 5 - Admin: Reports: Overdue Logic
            tx = pd.read_sql_query(
                """
                SELECT 
                    t.id, b.code as book_code, b.title, u.code as student_code, u.full_name as student, 
                    t.issue_date, t.return_date, t.predicted_days, t.actual_days, 
                    CASE 
                        WHEN t.status='issued' 
                            AND julianday('now') - julianday(t.issue_date) > t.predicted_days 
                        THEN 'overdue' 
                        ELSE t.status 
                    END AS status
                FROM transactions t 
                LEFT JOIN books b ON t.book_id=b.id 
                LEFT JOIN users u ON t.student_id=u.id
                """,
                conn,
            )
            conn.close()
            st.dataframe(tx)
            csv = tx.to_csv(index=False).encode("utf-8")
            st.download_button("Download transactions CSV", data=csv, file_name="transactions.csv", mime="text/csv")

            if not tx.empty:
                # The logic for 'Potential overdue' needs to check the 'status' column
                overdue = tx[tx["status"] == "overdue"]
                st.markdown("**Overdue Loans (issued and passed predicted return date)**")
                st.dataframe(overdue)
        except Exception as e:
            st.error("Error generating reports: " + str(e))

    # Users
    with tabs[5]: # Note: Index changed due to new 'Requests' tab
        st.subheader("User list and admin student creation")
        try:
            conn = get_connection()
            users_df = pd.read_sql_query("SELECT id, code, username, full_name, role FROM users", conn)
            conn.close()
            st.dataframe(users_df)
        except Exception as e:
            st.error("Error reading users: " + str(e))

        st.markdown("---")
        st.subheader("Create student (admin)")
        with st.form("admin_create_student"):
            uname = st.text_input("Username")
            fname = st.text_input("Full name")
            pwd = st.text_input("Password", type="password")
            if st.form_submit_button("Create student"):
                ok, res = admin_create_student(uname, fname, pwd)
                if ok:
                    st.success(f"Student created with code {res}")
                else:
                    st.error(res)

# Student dashboard
else:
    st.header("Student Dashboard")
    st.subheader(f"Welcome {user.get('full_name')} ({user.get('code')})")
    tabs = st.tabs(["All Books & Request", "My Loans", "Return Book", "My Reports"]) # Renamed tab 0

    # All Books & Request
    # STEP 3 - Student: Modify "All Books" to include Request feature
    with tabs[0]:
        st.subheader("Available books")
        try:
            conn = get_connection()
            books = pd.read_sql_query(
                "SELECT id, code, title, author, available_copies FROM books", conn
            )
            conn.close()
            st.dataframe(books)

            st.subheader("Request a book")

            # Check for current pending requests to inform the user
            conn = get_connection()
            pending_requests = pd.read_sql_query(
                "SELECT book_id FROM book_requests WHERE student_id=? AND status='pending'",
                conn,
                params=(user["id"],),
            )["book_id"].tolist()
            conn.close()

            # Filter books to only show those not already requested (pending)
            books_to_request = books[~books['id'].isin(pending_requests)]

            if books_to_request.empty:
                st.info("No books available to request, or all available books are already pending a request.")
            else:
                request_list = (books_to_request["title"] + " (" + books_to_request["code"] + ")").tolist()
                sel = st.selectbox("Select book to request", request_list, key="student_request_sel")
                book_row = books_to_request[(books_to_request["title"] + " (" + books_to_request["code"] + ")") == sel].iloc[0]

                if st.button("Request this book"):
                    ok = student_request_book(user["id"], int(book_row["id"]))
                    if ok:
                        st.success("Book request sent to admin.")
                    else:
                        st.error("You already have a pending request for this book.")
                    st.rerun()

        except Exception as e:
            st.error("Error reading books or handling request: " + str(e))

    with tabs[1]:
        st.subheader("My active and past loans")
        try:
            conn = get_connection()
            # STEP 5 - Student: My Loans: Overdue Logic
            mytx = pd.read_sql_query(
                """
                SELECT 
                    t.id, b.code as book_code, b.title, t.issue_date, t.return_date, 
                    t.predicted_days, t.actual_days,
                    CASE 
                        WHEN t.status='issued' 
                            AND julianday('now') - julianday(t.issue_date) > t.predicted_days 
                        THEN 'overdue' 
                        ELSE t.status 
                    END AS status
                FROM transactions t 
                LEFT JOIN books b ON t.book_id=b.id 
                WHERE t.student_id=?
                ORDER BY t.id DESC
                """,
                conn,
                params=(user["id"],),
            )
            conn.close()
            st.dataframe(mytx)
        except Exception as e:
            st.error("Error reading your transactions: " + str(e))

    with tabs[2]:
        st.subheader("Return a book")
        try:
            conn = get_connection()
            my_issued = pd.read_sql_query(
                "SELECT t.id, b.code as book_code, b.title, t.issue_date, t.book_id FROM transactions t LEFT JOIN books b ON t.book_id=b.id "
                "WHERE t.student_id=? AND t.status='issued'",
                conn,
                params=(user["id"],),
            )
            conn.close()
            if my_issued.empty:
                st.info("No books assigned to you currently.")
            else:
                loan_options = (my_issued["title"] + " (" + my_issued["book_code"] + ")").tolist()
                opt = st.selectbox("Select loan to return", loan_options)
                tr = my_issued[(my_issued["title"] + " (" + my_issued["book_code"] + ")") == opt].iloc[0]
                if st.button("Return selected book"):
                    ret_date = date.today().isoformat()
                    issue_dt = pd.to_datetime(tr["issue_date"]).date()
                    actual = (datetime.fromisoformat(ret_date).date() - issue_dt).days
                    conn = get_connection()
                    cur = conn.cursor()
                    cur.execute(
                        "UPDATE transactions SET return_date=?, actual_days=?, status='returned' WHERE id=?",
                        (ret_date, actual, tr["id"]),
                    )
                    # Use the book_id from the transaction row for the update
                    cur.execute(
                        "UPDATE books SET available_copies=available_copies+1 WHERE id=?",
                        (tr["book_id"],),
                    )
                    conn.commit()
                    conn.close()
                    st.success(f"Returned. Actual days kept: {actual}")
                    st.rerun()
        except Exception as e:
            st.error("Error during return: " + str(e))

    with tabs[3]:
        st.subheader("My reports")
        try:
            conn = get_connection()
            # STEP 5 - Student: My Reports: Overdue Logic
            mytx = pd.read_sql_query(
                """
                SELECT 
                    t.id, b.code as book_code, b.title, t.issue_date, t.return_date, 
                    t.predicted_days, t.actual_days,
                    CASE 
                        WHEN t.status='issued' 
                            AND julianday('now') - julianday(t.issue_date) > t.predicted_days 
                        THEN 'overdue' 
                        ELSE t.status 
                    END AS status
                FROM transactions t 
                LEFT JOIN books b ON t.book_id=b.id 
                WHERE t.student_id=?
                ORDER BY t.id DESC
                """,
                conn,
                params=(user["id"],),
            )
            conn.close()
            st.dataframe(mytx)
            csv = mytx.to_csv(index=False).encode("utf-8")
            st.download_button("Download my transactions CSV", data=csv, file_name="my_transactions.csv", mime="text/csv")
        except Exception as e:
            st.error("Error reading your reports: " + str(e))

st.markdown("\n---\n*Project by student — extend as needed.*")