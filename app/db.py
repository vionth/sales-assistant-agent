"""SQLite storage for the sales CRM demo data."""
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "sales.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS customers (
    id          INTEGER PRIMARY KEY,
    name        TEXT NOT NULL,
    company     TEXT NOT NULL,
    email       TEXT NOT NULL,
    phone       TEXT,
    city        TEXT,
    owner       TEXT
);

CREATE TABLE IF NOT EXISTS invoices (
    id            INTEGER PRIMARY KEY,
    customer_id   INTEGER NOT NULL REFERENCES customers(id),
    amount        REAL NOT NULL,
    currency      TEXT NOT NULL DEFAULT 'INR',
    issued_on     TEXT NOT NULL,
    due_on        TEXT NOT NULL,
    status        TEXT NOT NULL CHECK (status IN ('paid', 'unpaid', 'overdue'))
);
"""

SEED_CUSTOMERS = [
    (1, "Kumar Raja", "Raja Traders", "kumar@rajatraders.in", "9840011221", "Chennai", "Vinoth"),
    (2, "Meena Iyer", "Iyer Textiles", "meena@iyertextiles.in", "9840033445", "Coimbatore", "Vinoth"),
    (3, "Arun Prasad", "AP Electricals", "arun@apelectricals.in", "9840055667", "Madurai", "Divya"),
    (4, "Fatima Noor", "Noor Logistics", "fatima@noorlogistics.in", "9840077889", "Bengaluru", "Vinoth"),
    (5, "Suresh Babu", "SB Hardware", "suresh@sbhardware.in", "9840099001", "Salem", "Divya"),
]

SEED_INVOICES = [
    (101, 1, 48500.00, "INR", "2026-08-02", "2026-09-01", "paid"),
    (102, 1, 72000.00, "INR", "2026-09-05", "2026-10-05", "overdue"),
    (103, 2, 15750.00, "INR", "2026-09-12", "2026-10-12", "unpaid"),
    (104, 3, 96000.00, "INR", "2026-07-20", "2026-08-19", "paid"),
    (105, 4, 134500.00, "INR", "2026-09-28", "2026-10-28", "unpaid"),
    (106, 4, 22000.00, "INR", "2026-08-15", "2026-09-14", "overdue"),
    (107, 5, 58900.00, "INR", "2026-10-01", "2026-10-31", "unpaid"),
    (108, 2, 41200.00, "INR", "2026-06-10", "2026-07-10", "paid"),
]


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(force: bool = False) -> None:
    """Create the schema and seed demo rows. Idempotent unless force=True."""
    with connect() as conn:
        if force:
            conn.executescript("DROP TABLE IF EXISTS invoices; DROP TABLE IF EXISTS customers;")
        conn.executescript(SCHEMA)
        if conn.execute("SELECT COUNT(*) FROM customers").fetchone()[0] == 0:
            conn.executemany("INSERT INTO customers VALUES (?,?,?,?,?,?,?)", SEED_CUSTOMERS)
            conn.executemany("INSERT INTO invoices VALUES (?,?,?,?,?,?,?)", SEED_INVOICES)
        conn.commit()


if __name__ == "__main__":
    init_db(force=True)
    print(f"Initialised {DB_PATH} with {len(SEED_CUSTOMERS)} customers and {len(SEED_INVOICES)} invoices.")
