# FedLineage — Federated MLOps Observability Platform

**Final Year B.E. Project — Review 3 Simulation Proof-of-Concept**

---

## 📌 Project Overview
**FedLineage** is an observability and model lineage platform designed for Federated Learning (FL) workflows. While traditional FL platforms track only final model accuracy, FedLineage captures deep execution metadata — including client CPU/memory/network telemetry, communication overhead, training latency, and directed provenance graphs (`Client → Local Update → Round → Global Model Version`).

This repository contains the complete **Review 3 simulation proof-of-concept**, fulfilling all 3 workloads:
1. **Simulated FL Backend & Metadata Generation** (`simulation.py`, `data/`)
2. **Streamlit Observability Dashboard & Analytics** (`app.py`, `analytics.py`)
3. **Model Lineage Graph & Root-Cause Investigation** (`lineage.py`, `data/lineage.json`)

---

## 📁 Repository Structure

```text
FedLineage/
│
├── app.py              # Streamlit multi-page dashboard UI
├── analytics.py        # Plotly chart generation and visualization helpers
├── lineage.py          # NetworkX lineage graph generator & root-cause tracer
├── simulation.py       # FL metadata simulation engine (clients, rounds, experiments)
│
├── data/
│   ├── clients.csv     # 50 rows: 5 clients × 10 rounds telemetry (CPU, RAM, BW, Time, Loss, Acc, Status)
│   ├── rounds.csv      # 10 rows: Global accuracy, loss, communication MB, round duration
│   ├── experiments.csv # 3 experiments: FedAvg vs FedProx across IID and Non-IID distributions
│   └── lineage.json    # Complete directed graph relationships & anomaly metadata
│
├── requirements.txt    # Python package dependencies
└── README.md           # Documentation and demo guide
```

---

## ⚙️ Installation & Setup

### 1. Clone & Setup Environment
```bash
# Clone the repository
git clone <repo-url>
cd BE-Project

# Create and activate virtual environment (optional)
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

### 2. Generate Simulated Metadata
Run the simulation backend to produce all CSV files and lineage JSON:
```bash
python simulation.py
python lineage.py
```

### 3. Launch the Observability Dashboard
```bash
streamlit run app.py
```
The dashboard will open automatically in your browser at `http://localhost:8501`.

---

## 🖥️ Review 3 Final Demo Flow & Script

Use this exact step-by-step script during your Review 3 demonstration:

### Step 1: Opening
* **Action:** Open `http://localhost:8501` to the FedLineage dashboard.
* **Script:**  
  > *"This is our current proof-of-concept implementation of the FedLineage observability layer for Federated Learning."*

### Step 2: Show Overview Page
* **Action:** Point to the 5 top KPI cards (Clients: 5, Rounds: 10, Final Accuracy: 86.4%, Communication: 207.7 MB, Stragglers: 1) and the accuracy/loss/communication charts.
* **Script:**  
  > *"Instead of tracking only model accuracy, we also capture execution metadata such as communication cost, training time, and client participation."*

### Step 3: Go to Client Health Page
* **Action:** Switch to `💻 Client Health` in the sidebar and show Round 7. Point to Client C3 in the table and the red anomaly detection box.
* **Script:**  
  > *"Here we can observe the operational state of individual clients. In our simulated scenario, Client C3 has unusually high resource utilization (CPU 94%, RAM 91%), low network bandwidth (1.7 MB/s), and significantly higher training latency (43s)."*

### Step 4: Go to Model Lineage Page
* **Action:** Switch to `🧬 Model Lineage` in the sidebar. Select Global Model V7. Show the directed provenance graph: `Client -> Local Update -> Round 7 -> Global Model V7` with C3 highlighted in red (⚠).
* **Script:**  
  > *"The key innovation of FedLineage is that we don't just know that Model V7 degraded (dropping to 80.8%). We can reconstruct the provenance chain and trace the model version back through its participating rounds and client updates."*

### Step 5: Show Root-Cause Panel
* **Action:** Scroll down to the `Model V7 Investigation` panel.
* **Script:**  
  > *"The system identifies Client C3 as a potential contributing straggler based on its telemetry. This is currently a simulated proof-of-concept; the full automated root-cause engine will be expanded in future work."*

### Step 6: Show Experiment Comparison Page
* **Action:** Switch to `🧪 Experiment Comparison` in the sidebar. Show the comparison table and the bar/scatter charts comparing FedAvg (IID vs Non-IID) and FedProx.
* **Script:**  
  > *"Finally, the platform allows operators to compare different FL configurations using multiple dimensions — accuracy, training duration, and communication cost — rather than accuracy alone."*

---

## 🎯 Review 3 Success Statement
> *"We have implemented a working proof-of-concept of the FedLineage observability layer. We simulate FL execution, collect execution metadata, visualize client health and experiment performance, and reconstruct model lineage to trace a simulated anomaly back to its client and training round."*
