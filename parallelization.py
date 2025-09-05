#!/usr/bin/env python3
import socket
import threading
import pickle
import pandas as pd
import sqlite3
import sys
import time
import logging
import numpy as np
import struct

# -----------------------
# Config
# -----------------------
DB_NAME = "sales_analysis.db"
HOST = "0.0.0.0"
PORT = 5000
CHUNKS_PER_WORKER = 10

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s")

# -----------------------
# Helpers for sending/receiving full messages
# -----------------------
def send_msg(sock, data):
    payload = pickle.dumps(data)
    length = struct.pack("!I", len(payload))  # 4 bytes length
    sock.sendall(length + payload)

def recv_msg(sock):
    raw_len = recvall(sock, 4)
    if not raw_len:
        return None
    msg_len = struct.unpack("!I", raw_len)[0]
    data = recvall(sock, msg_len)
    return pickle.loads(data)

def recvall(sock, n):
    """Receive n bytes exactly"""
    buf = b""
    while len(buf) < n:
        packet = sock.recv(n - len(buf))
        if not packet:
            return None
        buf += packet
    return buf

# -----------------------
# Database Manager
# -----------------------
class DatabaseManager:
    def __init__(self, db_name=DB_NAME):
        self.db_name = db_name
        self.init_db()

    def init_db(self):
        with sqlite3.connect(self.db_name) as conn:
            c = conn.cursor()
            # Drop old tables to avoid schema mismatch
            c.execute("DROP TABLE IF EXISTS worker_results")
            c.execute("DROP TABLE IF EXISTS final_results")

            c.execute(
                """CREATE TABLE worker_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    worker_id TEXT,
                    rows_processed INTEGER,
                    total_sales_amount REAL,
                    min_price REAL,
                    max_price REAL,
                    avg_price REAL,
                    timestamp TEXT
                )"""
            )
            c.execute(
                """CREATE TABLE final_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    total_rows INTEGER,
                    total_sales_amount REAL,
                    min_price REAL,
                    max_price REAL,
                    avg_price REAL,
                    processing_time REAL,
                    timestamp TEXT,
                    workers_used INTEGER,
                    chunks_processed INTEGER
                )"""
            )
            conn.commit()

    def insert_worker_result(self, result):
        with sqlite3.connect(self.db_name) as conn:
            c = conn.cursor()
            c.execute(
                """INSERT INTO worker_results 
                   (worker_id, rows_processed, total_sales_amount, min_price, max_price, avg_price, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    result["worker_id"],
                    result["rows_processed"],
                    result["total_sales_amount"],
                    result["min_price"],
                    result["max_price"],
                    result["avg_price"],
                    result["timestamp"],
                ),
            )
            conn.commit()

    def aggregate_results(self, dataset_rows, num_workers, chunks_per_worker, start_time):
        with sqlite3.connect(self.db_name) as conn:
            c = conn.cursor()
            c.execute("""SELECT SUM(total_sales_amount), MIN(min_price), MAX(max_price) FROM worker_results""")
            total_sales, min_price, max_price = c.fetchone()

            c.execute("""SELECT SUM(rows_processed * avg_price), SUM(rows_processed) FROM worker_results""")
            weighted_sum, total_rows_counted = c.fetchone()
            avg_price = weighted_sum / total_rows_counted if total_rows_counted else 0

            processing_time = time.time() - start_time
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            chunks_processed = num_workers * chunks_per_worker

            c.execute(
                """INSERT INTO final_results 
                   (total_rows, total_sales_amount, min_price, max_price, avg_price, processing_time, timestamp, workers_used, chunks_processed)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    dataset_rows,
                    total_sales,
                    min_price,
                    max_price,
                    avg_price,
                    processing_time,
                    timestamp,
                    num_workers,
                    chunks_processed,
                ),
            )
            conn.commit()

        # Pretty print
        print("\n" + "=" * 70)
        print(" FINAL AGGREGATED RESULTS ".center(70, "="))
        print(f" Timestamp              : {timestamp}")
        print(f" Dataset Rows           : {dataset_rows:,}")
        print(f" Workers Used           : {num_workers}")
        print(f" Chunks per Worker      : {chunks_per_worker}")
        print(f" Total Chunks Processed : {chunks_processed}")
        print("-" * 70)
        print(f" Total Rows Processed   : {dataset_rows:,}")
        print(f" Total Sales Amount     : {total_sales:,.2f}")
        print(f" Min Price              : {min_price:.2f}")
        print(f" Max Price              : {max_price:.2f}")
        print(f" Avg Price              : {avg_price:.2f}")
        print(f" Processing Time        : {processing_time:.2f} seconds")
        print("=" * 70 + "\n")

# -----------------------
# Server
# -----------------------
class SalesDataServer:
    def __init__(self, csv_file, num_workers):
        self.csv_file = csv_file
        self.num_workers = num_workers
        self.db = DatabaseManager()
        self.start_time = None
        self.chunks = []
        self.dataset_rows = 0

    def split_data(self):
        df = pd.read_csv(self.csv_file)
        self.dataset_rows = len(df)
        logging.info(f"Dataset loaded: {self.dataset_rows} rows")
        total_chunks = self.num_workers * CHUNKS_PER_WORKER
        self.chunks = np.array_split(df, total_chunks)
        logging.info(f"Split into {total_chunks} chunks")

    def handle_worker(self, conn, addr, worker_id):
        logging.info(f"Worker {worker_id} connected: {addr}")
        while True:
            try:
                if not self.chunks:
                    send_msg(conn, "STOP")
                    break
                chunk = self.chunks.pop(0)
                send_msg(conn, chunk)
                result = recv_msg(conn)
                if result:
                    self.db.insert_worker_result(result)
                    logging.info(f"Received result from {worker_id}")
            except Exception as e:
                logging.error(f"Worker {worker_id} error: {e}")
                break
        conn.close()
        logging.info(f"Worker {worker_id} finished")

    def start(self):
        self.split_data()
        self.start_time = time.time()
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind((HOST, PORT))
        server_socket.listen(self.num_workers)
        logging.info(f"Server started on {HOST}:{PORT}, waiting for {self.num_workers} workers...")

        threads = []
        for i in range(self.num_workers):
            conn, addr = server_socket.accept()
            worker_id = f"Worker-{i+1}"
            t = threading.Thread(target=self.handle_worker, args=(conn, addr, worker_id))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()
        self.db.aggregate_results(self.dataset_rows, self.num_workers, CHUNKS_PER_WORKER, self.start_time)

# -----------------------
# Worker
# -----------------------
def process_chunk(df, worker_id):
    sales_amount = (df["Price"] * df["Quantity"]).sum()
    return {
        "worker_id": worker_id,
        "rows_processed": len(df),
        "total_sales_amount": float(sales_amount),
        "min_price": float(df["Price"].min()),
        "max_price": float(df["Price"].max()),
        "avg_price": float(df["Price"].mean()),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

def start_worker():
    worker_id = f"Worker-{int(time.time())}"
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((HOST, PORT))
    logging.info(f"{worker_id} connected, waiting for chunks...")

    while True:
        chunk = recv_msg(sock)
        if isinstance(chunk, str) and chunk == "STOP":
            logging.info(f"{worker_id} received STOP signal, exiting.")
            break
        result = process_chunk(chunk, worker_id)
        send_msg(sock, result)

    sock.close()
    logging.info(f"{worker_id} closed connection.")

# -----------------------
# Entry Point
# -----------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python distributed_sales_system.py server <num_workers>")
        print("  python distributed_sales_system.py worker")
        sys.exit(1)

    mode = sys.argv[1]
    if mode == "server":
        if len(sys.argv) < 3:
            print("Usage: python distributed_sales_system.py server <num_workers>")
            sys.exit(1)
        num_workers = int(sys.argv[2])
        server = SalesDataServer("sales_data_5m.csv", num_workers)
        server.start()
    elif mode == "worker":
        start_worker()
    else:

        print("Invalid mode. Use 'server' or 'worker'.")
