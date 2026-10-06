"""Project 03 — SQL Database Agent: demo SQLite database + schema helpers.

Builds a deterministic synthetic e-commerce database in an in-memory SQLite
instance exposed through a SQLAlchemy engine (so LangChain's ``SQLDatabase`` can
introspect it) and exposes helpers for schema text and row counts.

Tables: customers, products, orders, order_items.
"""
from __future__ import annotations

import random
from datetime import date, timedelta

from sqlalchemy import create_engine, text
from sqlalchemy.pool import StaticPool

SCHEMA_SQL = """
CREATE TABLE customers (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT NOT NULL,
    country TEXT NOT NULL,
    plan TEXT NOT NULL,           -- free | starter | pro | enterprise
    signup_date TEXT NOT NULL
);
CREATE TABLE products (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    category TEXT NOT NULL,       -- hardware | software | subscription | accessory
    price REAL NOT NULL
);
CREATE TABLE orders (
    id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL,
    order_date TEXT NOT NULL,
    status TEXT NOT NULL,         -- pending | paid | shipped | delivered | refunded
    total REAL NOT NULL,
    FOREIGN KEY (customer_id) REFERENCES customers(id)
);
CREATE TABLE order_items (
    id INTEGER PRIMARY KEY,
    order_id INTEGER NOT NULL,
    product_id INTEGER NOT NULL,
    quantity INTEGER NOT NULL,
    unit_price REAL NOT NULL,
    FOREIGN KEY (order_id) REFERENCES orders(id),
    FOREIGN KEY (product_id) REFERENCES products(id)
);
"""

COUNTRIES = ["Germany", "USA", "UK", "France", "India", "Brazil", "Canada", "Japan", "Spain", "Australia"]
PLANS = ["free", "starter", "starter", "pro", "pro", "enterprise"]
CATEGORIES = {
    "hardware": ["Router X1", "Mesh Node Pro", "USB-C Hub", "4K Webcam", "Mechanical Keyboard"],
    "software": ["Studio License", "Analytics Add-on", "Backup Suite", "Design Toolkit"],
    "subscription": ["Cloud Basic", "Cloud Pro", "Cloud Enterprise", "Support Plan"],
    "accessory": ["Laptop Sleeve", "Desk Mat", "Cable Kit", "Monitor Stand"],
}
STATUSES = ["pending", "paid", "paid", "shipped", "shipped", "delivered", "delivered", "delivered", "refunded"]
FIRST = ["Alex", "Maria", "Liam", "Sofia", "Noah", "Emma", "Yuki", "Omar", "Priya", "Lucas", "Hana", "Diego",
         "Nina", "Ethan", "Zara", "Ken", "Aisha", "Ravi", "Elena", "Tom", "Mia", "Jonas", "Lea", "Finn"]
LAST = ["Meyer", "Silva", "Okafor", "Khan", "Rossi", "Novak", "Tanaka", "Haddad", "Patel", "Costa",
        "Kim", "Alvarez", "Petrov", "Brown", "Aziz", "Yamamoto", "Diallo", "Iyer", "Fischer", "Wolff",
        "Berg", "Nagy", "Lund", "Reyes"]


def build_engine(seed: int = 42):
    """Create an in-memory SQLite engine populated with deterministic demo data."""
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    rng = random.Random(seed)
    with engine.begin() as conn:
        for stmt in SCHEMA_SQL.split(";"):
            if stmt.strip():
                conn.execute(text(stmt))

        for i in range(1, 61):
            name = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
            email = name.lower().replace(" ", ".") + f"{i}@example.com"
            signup = date(2023, 1, 1) + timedelta(days=rng.randint(0, 900))
            conn.execute(
                text("INSERT INTO customers (id,name,email,country,plan,signup_date) VALUES (:i,:n,:e,:c,:p,:s)"),
                {"i": i, "n": name, "e": email, "c": rng.choice(COUNTRIES),
                 "p": rng.choice(PLANS), "s": signup.isoformat()},
            )

        products: dict[int, float] = {}
        pid = 1
        for category, items in CATEGORIES.items():
            for nm in items:
                base = {"hardware": 120, "software": 60, "subscription": 25, "accessory": 20}[category]
                price = round(base * rng.uniform(0.6, 2.2), 2)
                conn.execute(
                    text("INSERT INTO products (id,name,category,price) VALUES (:i,:n,:c,:p)"),
                    {"i": pid, "n": nm, "c": category, "p": price},
                )
                products[pid] = price
                pid += 1

        item_id = 1
        for oid in range(1, 201):
            cust = rng.randint(1, 60)
            odate = date(2024, 1, 1) + timedelta(days=rng.randint(0, 500))
            status = rng.choice(STATUSES)
            total = 0.0
            for _ in range(rng.randint(1, 4)):
                prod = rng.randint(1, pid - 1)
                price = products[prod]
                qty = rng.randint(1, 5)
                total += price * qty
                conn.execute(
                    text("INSERT INTO order_items (id,order_id,product_id,quantity,unit_price) "
                         "VALUES (:i,:o,:p,:q,:u)"),
                    {"i": item_id, "o": oid, "p": prod, "q": qty, "u": price},
                )
                item_id += 1
            conn.execute(
                text("INSERT INTO orders (id,customer_id,order_date,status,total) VALUES (:i,:c,:d,:s,:t)"),
                {"i": oid, "c": cust, "d": odate.isoformat(), "s": status, "t": round(total, 2)},
            )
    return engine


def get_langchain_db(engine, sample_rows: int = 2):
    """Wrap the engine in a LangChain ``SQLDatabase`` for schema introspection."""
    from langchain_community.utilities import SQLDatabase

    return SQLDatabase(
        engine,
        include_tables=["customers", "products", "orders", "order_items"],
        sample_rows_in_table_info=sample_rows,
    )


def schema_text(db) -> str:
    return db.get_table_info()


def table_stats(engine) -> dict[str, int]:
    out: dict[str, int] = {}
    with engine.connect() as conn:
        for t in ("customers", "products", "orders", "order_items"):
            try:
                out[t] = int(conn.execute(text(f"SELECT COUNT(*) FROM {t}")).scalar() or 0)
            except Exception:
                out[t] = 0
    return out
