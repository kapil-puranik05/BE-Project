import os

import numpy as np
import pandas as pd

SEED = 42
DATA_DIR = "data"

EXPERIMENT_ID = "FL_EXP_001"
DATASET = "CIFAR-100"
ALGORITHM = "FedAvg"
NUM_CLIENTS = ["C1", "C2", "C3", "C4", "C5"]
NUM_ROUNDS = 10
ANOMALY_ROUNDS = [6, 7]
STRAGGLER = "C3"

BASELINE = {
    "C1": {"cpu_usage": 62, "memory_usage": 65, "network_bandwidth": 8.2, "training_time": 17.3, "local_loss": 0.42, "local_accuracy": 84.1},
    "C2": {"cpu_usage": 71, "memory_usage": 68, "network_bandwidth": 7.8, "training_time": 18.1, "local_loss": 0.39, "local_accuracy": 85.2},
    "C3": {"cpu_usage": 91, "memory_usage": 89, "network_bandwidth": 2.1, "training_time": 41.5, "local_loss": 0.61, "local_accuracy": 78.3},
    "C4": {"cpu_usage": 55, "memory_usage": 62, "network_bandwidth": 9.1, "training_time": 16.2, "local_loss": 0.40, "local_accuracy": 84.7},
    "C5": {"cpu_usage": 68, "memory_usage": 70, "network_bandwidth": 8.5, "training_time": 19.0, "local_loss": 0.43, "local_accuracy": 83.8},
}

ANOMALY = {
    "cpu_usage": 94,
    "memory_usage": 91,
    "network_bandwidth": 1.7,
    "training_time": 43.0,
    "local_loss": 0.74,
    "local_accuracy": 70.9,
}

GLOBAL_ACCURACY = [72.3, 75.1, 77.8, 79.9, 82.1, 83.4, 80.8, 84.0, 85.2, 86.4]
GLOBAL_LOSS = [0.81, 0.72, 0.65, 0.61, 0.56, 0.53, 0.62, 0.51, 0.47, 0.44]
COMMUNICATION_MB = [18.2, 19.1, 20.0, 19.6, 20.4, 20.9, 21.3, 20.7, 21.0, 21.5]
ROUND_TRAINING_TIME = [84, 91, 87, 89, 93, 96, 104, 92, 95, 98]

EXPERIMENTS = [
    {"experiment_id": "EXP001", "algorithm": "FedAvg", "data_distribution": "IID", "final_accuracy": 89.2, "total_training_time": 120, "communication_mb": 41.5, "stragglers": 0},
    {"experiment_id": "EXP002", "algorithm": "FedAvg", "data_distribution": "Non-IID", "final_accuracy": 82.4, "total_training_time": 147, "communication_mb": 45.3, "stragglers": 2},
    {"experiment_id": "EXP003", "algorithm": "FedProx", "data_distribution": "Non-IID", "final_accuracy": 86.7, "total_training_time": 138, "communication_mb": 43.1, "stragglers": 1},
]


def simulate_clients(rng):
    rows = []
    for r in range(1, NUM_ROUNDS + 1):
        for cid in NUM_CLIENTS:
            base = BASELINE[cid]
            is_anomaly = cid == STRAGGLER and r in ANOMALY_ROUNDS
            src = ANOMALY if is_anomaly else base
            progress = (r - 1) / (NUM_ROUNDS - 1)
            jitter = rng.normal(0, 1)
            if is_anomaly:
                cpu = src["cpu_usage"]
                mem = src["memory_usage"]
                bw = src["network_bandwidth"]
                tt = src["training_time"]
                loss = src["local_loss"] + abs(rng.normal(0, 0.01))
                acc = src["local_accuracy"] - abs(rng.normal(0, 0.3))
            else:
                cpu = np.clip(src["cpu_usage"] + jitter * 2.5, 5, 100)
                mem = np.clip(src["memory_usage"] + jitter * 2.0, 5, 100)
                bw = max(0.5, src["network_bandwidth"] + rng.normal(0, 0.3))
                tt = src["training_time"] * (1 + rng.normal(0, 0.05))
                loss = max(0.05, src["local_loss"] - 0.02 * progress + rng.normal(0, 0.01))
                acc = min(99.0, src["local_accuracy"] + 2.5 * progress + rng.normal(0, 0.4))
            status = "straggler" if cid == STRAGGLER else "healthy"
            rows.append({
                "client_id": cid,
                "round": r,
                "cpu_usage": round(cpu, 1),
                "memory_usage": round(mem, 1),
                "network_bandwidth": round(bw, 1),
                "training_time": round(tt, 1),
                "local_loss": round(loss, 3),
                "local_accuracy": round(acc, 1),
                "status": status,
            })
    return pd.DataFrame(rows)


def simulate_rounds():
    return pd.DataFrame({
        "round": range(1, NUM_ROUNDS + 1),
        "global_accuracy": GLOBAL_ACCURACY,
        "global_loss": GLOBAL_LOSS,
        "communication_mb": COMMUNICATION_MB,
        "training_time": ROUND_TRAINING_TIME,
        "participating_clients": [len(NUM_CLIENTS)] * NUM_ROUNDS,
    })


def simulate_experiments():
    return pd.DataFrame(EXPERIMENTS)


def main():
    rng = np.random.default_rng(SEED)
    os.makedirs(DATA_DIR, exist_ok=True)

    clients = simulate_clients(rng)
    rounds = simulate_rounds()
    experiments = simulate_experiments()

    clients.to_csv(os.path.join(DATA_DIR, "clients.csv"), index=False)
    rounds.to_csv(os.path.join(DATA_DIR, "rounds.csv"), index=False)
    experiments.to_csv(os.path.join(DATA_DIR, "experiments.csv"), index=False)

    print(f"Experiment : {EXPERIMENT_ID}")
    print(f"Dataset    : {DATASET}")
    print(f"Algorithm  : {ALGORITHM}")
    print(f"Clients    : {len(NUM_CLIENTS)}")
    print(f"Rounds     : {NUM_ROUNDS}")
    print(f"Anomaly    : {STRAGGLER} in rounds {ANOMALY_ROUNDS}")
    print(f"Wrote {len(clients)} rows -> data/clients.csv")
    print(f"Wrote {len(rounds)} rows -> data/rounds.csv")
    print(f"Wrote {len(experiments)} rows -> data/experiments.csv")


if __name__ == "__main__":
    main()
