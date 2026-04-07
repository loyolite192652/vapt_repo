# AI/ML-Based Intelligent Vulnerability Assessment and Penetration Testing Using Nmap in Kali Linux

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2%2B-orange?logo=scikit-learn&logoColor=white)
![Nmap](https://img.shields.io/badge/Nmap-7.94-green?logo=nmap)
![Kali Linux](https://img.shields.io/badge/Kali%20Linux-2025.4-blue?logo=kalilinux&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Research%20Paper-red)



**Authors:** Naveen Kumar Bandla¹ · Dr. Y. Nasir Ahmed²

¹ Email Security Specialist, Centific, Hyderabad, India
² Department of Computer Science, Chaitanya Deemed To Be University, Hyderabad, India

</div>

---

## Table of Contents

- [Overview](#overview)
- [Is This Real AI/ML?](#is-this-real-aiml)
- [Pipeline Architecture](#pipeline-architecture)
- [How It Works — Step by Step](#how-it-works--step-by-step)
- [Features Explained](#features-explained)
- [ML Models Explained](#ml-models-explained)
- [Installation](#installation)
- [Usage](#usage)
- [Output Explained](#output-explained)
- [Risk Score Formulas](#risk-score-formulas)
- [Results](#results)
- [Repository Structure](#repository-structure)
- [Citation](#citation)

---

## Overview

This repository contains the complete implementation of an **AI/ML-driven network vulnerability assessment pipeline** that automatically processes **Nmap XML scan output** and produces a prioritised, scored vulnerability report — without requiring any manually labelled real-world training data.

The system solves two fundamental problems in traditional penetration testing:

| Problem | Traditional Approach | This Pipeline |
|---------|---------------------|---------------|
| Raw Nmap XML is unstructured and verbose | Manual analyst review | Automated XML parsing + feature extraction |
| No predictive risk score | Manual CVSS lookup | Random Forest predicts severity tier |
| Can't detect unusual service behaviour | Rule-based checklists | Isolation Forest detects anomalies |
| Slow, inconsistent at scale | Analyst fatigue | Fully automated, reproducible |

---

## Is This Real AI/ML?

**Yes.** Here is an honest breakdown for reviewers and users:

| Component | Type | Genuine AI/ML? |
|-----------|------|----------------|
| Nmap XML Parsing (Stage 0) | Automation | ❌ Not ML |
| Feature Engineering — PRS, VES, PEF, PSC, VKF (Stage 1) | Domain-informed rules | ❌ Not ML |
| **Random Forest Classifier (Stage 2)** | **Supervised ML** | ✅ **Yes — Real ML** |
| **Isolation Forest (Stage 3)** | **Unsupervised ML** | ✅ **Yes — Real ML** |
| GWVS / NEF / ARS Score (Stage 4) | Mathematical formula | ❌ Not ML |

### Why the rule-based features are still valid

The feature engineering stage (PRS, VES, PEF, PSC, VKF) provides **domain-informed numerical inputs** to the ML models. This is standard practice in applied ML — raw text/metadata must be converted to numbers before a model can learn from them. The **learning and classification** is done entirely by the Random Forest; the feature engineering does not perform classification.

Think of it this way: in sentiment analysis, converting words to TF-IDF vectors is not the "AI" — the classifier that learns from those vectors is. Same principle here.

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    INPUT: Nmap XML File                         │
│              (nmap -sV -O -oX scan_results.xml <IP>)           │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 0 — Data Ingestion & XML Parsing                         │
│  • Parses Nmap XML using xmltodict                              │
│  • Extracts: port_id, protocol, service_name,                  │
│              service_version, host_ip                           │
│  • Filters: only port_state = "open" retained                  │
│  • Output: structured pandas DataFrame                          │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 1 — Feature Engineering                                  │
│  Constructs 5-feature matrix X ∈ R^(n×5):                      │
│                                                                 │
│  PRS  Port Risk Score         float [0,1]                       │
│  VES  Version Entropy Score   float [0,1]                       │
│  PEF  Protocol Encryption Flag  int {0,1}                       │
│  PSC  Port Service Category     int {0..5}                      │
│  VKF  Version Known Flag        int {0,1}                       │
└─────────────────────────┬───────────────────────────────────────┘
                          │
              ┌───────────┴────────────┐
              │                        │
              ▼                        ▼
┌─────────────────────┐   ┌───────────────────────────────────────┐
│  STAGE 2            │   │  STAGE 3                              │
│  Supervised ML      │   │  Unsupervised ML                      │
│                     │   │                                       │
│  Random Forest      │   │  Isolation Forest                     │
│  Classifier         │   │  Anomaly Detection                    │
│                     │   │                                       │
│  Trained on 200     │   │  Fits on scan feature matrix          │
│  NVD-aligned records│   │  Computes anomaly score s(x,n)        │
│                     │   │  Labels: ANOMALY / Normal             │
│  Output:            │   │                                       │
│  predicted_tier     │   │  Output:                              │
│  rf_confidence      │   │  anomaly_score                        │
│  prob_Critical      │   │  anomaly_label                        │
│  prob_High          │   │                                       │
│  prob_Medium        │   │                                       │
│  prob_Low           │   │                                       │
└──────────┬──────────┘   └────────────────┬──────────────────────┘
           │                               │
           └───────────────┬───────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 4 — Risk Score Computation                               │
│  • GWVS — Global Weighted Vulnerability Score                   │
│  • NEF  — Network Exposure Factor                               │
│  • ARS  — Adjusted Risk Score = GWVS × NEF                     │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 5 — Unified Vulnerability Report                         │
│  • Global Dashboard (GWVS, NEF, ARS, tier counts)              │
│  • High-Priority Items Table (Critical + Anomalies)            │
│  • Full Service Assessment (all ports, ranked by CRI)          │
│  • Remediation Recommendations (per port)                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## How It Works — Step by Step

### Step 0: Run Nmap and Get XML

Before running the pipeline, scan your target network with Nmap:

```bash
# Basic service + OS scan → XML output
nmap -sV -O -oX scan_results.xml 192.168.1.1

# More thorough: include vulnerability scripts
nmap -sV -O --script vuln -oX scan_results.xml 192.168.1.1

# Aggressive scan (all detection modes)
nmap -A -oX scan_results.xml 192.168.1.1
```

The `-oX` flag saves results in XML format, which this pipeline reads.

---

### Step 1: XML Parsing (Stage 0)

The pipeline opens the XML file and navigates its structure:

```
nmaprun
  └── host
        ├── address (@addrtype="ipv4")     → host_ip
        └── ports
              └── port [@portid, @protocol]
                    ├── state [@state]     → filter: keep "open" only
                    └── service [@name, @version]
```

Only **open ports** are extracted. Closed and filtered ports are discarded because they do not represent active attack surface.

**What you get:** A clean DataFrame like this:

```
Port  Proto  Service   Version          Host IP
21    tcp    ftp       vsftpd 1.4.1     192.168.1.1
22    tcp    ssh       OpenSSH 7.4      192.168.1.1
23    tcp    telnet    unknown          192.168.1.1
53    tcp    domain    dnsmasq 2.80     192.168.1.1
80    tcp    http      lighttpd 0.93.15 192.168.1.1
443   tcp    https     lighttpd 0.93.15 192.168.1.1
```

---

### Step 2: Feature Engineering (Stage 1)

Five numerical features are computed for each service:

#### PRS — Port Risk Score
- Looked up from a knowledge base grounded in **NIST NVD CVE frequency data**
- Telnet (port 23) = 0.95 — highest CVE density, always critical
- HTTPS (port 443) = 0.45 — encrypted, much lower base risk
- Unknown ports get default PRS = 0.55

#### VES — Version Entropy Score
```
VES(i) = 1 - len(version_string_i) / (max_len + ε)
```
- If version is "unknown" → VES = 1.0 (maximum — can't assess risk)
- Longer, specific version strings → lower VES (more information = less entropy)
- Telnet with no version detected → VES = 1.0

#### PEF — Protocol Encryption Flag
- 1 = unencrypted transport (FTP, Telnet, HTTP, SMTP...)
- 0 = encrypted transport (HTTPS, SSH, FTPS, SFTP...)

#### PSC — Port Service Category
- Groups services: Web=0, Legacy=1, File=2, Remote=3, Database=4, Other=5
- Provides the model with service-type context beyond port number alone

#### VKF — Version Known Flag
- 1 = version string detected, 0 = version unknown
- Distinct from VES: VES measures how much version info exists; VKF is simply binary

**Combined feature matrix for the paper's experimental scan:**

```
Port  Service  PRS   VES   PEF  PSC  VKF  CRI
23    telnet   0.95  1.00   1    1    0   2.95  ← highest risk
21    ftp      0.90  0.25   1    1    1   2.15
80    http     0.75  0.00   1    0    1   1.75
53    domain   0.60  0.25   0    5    1   0.85
22    ssh      0.55  0.31   0    3    1   0.86
443   https    0.45  0.00   0    0    1   0.45  ← lowest risk
```

---

### Step 3: Random Forest Classifier (Stage 2) ✅ Real AI/ML

This is the **first genuine ML component**.

#### Training Data
A 200-record dataset constructed to match real NIST NVD CVE severity distributions:

| Severity | Count | Represents |
|----------|-------|-----------|
| Critical | 70 | CVSS ≥ 9.0 services (Telnet, FTP, RDP, SMB, Redis) |
| High | 60 | CVSS 7.0–8.9 (HTTP, old SSH, MySQL, DNS) |
| Medium | 40 | CVSS 4.0–6.9 (HTTPS, IMAP/TLS, admin panels) |
| Low | 30 | CVSS < 4.0 (modern SSH, patched HTTPS) |

#### Model Configuration
```python
RandomForestClassifier(
    n_estimators=200,     # 200 decision trees in ensemble
    max_depth=8,          # prevents overfitting
    class_weight="balanced",  # handles class imbalance
    random_state=42
)
```

#### Validation
- **5-fold Stratified Cross-Validation** on the training dataset
- CV Accuracy ≈ **89%** ± 2.5%
- CV F1-Macro ≈ **87%** ± 3%

#### Output per service
```
Port  Service  RF Tier   Confidence  P(Critical)  P(High)  P(Medium)  P(Low)
23    telnet   Critical  1.0000      1.0000       0.0000   0.0000     0.0000
21    ftp      Critical  0.9974      0.9974       0.0026   0.0000     0.0000
80    http     High      0.9493      0.0050       0.9493   0.0457     0.0000
443   https    Medium    0.5030      0.0000       0.0079   0.5030     0.4891
22    ssh      High      0.9384      0.0161       0.9384   0.0456     0.0000
53    domain   High      0.7507      0.0100       0.7507   0.2062     0.0332
```

The model gives **probability scores for every severity class**, not just a single label — which is far more informative than a rule-based lookup.

---

### Step 4: Isolation Forest Anomaly Detection (Stage 3) ✅ Real AI/ML

This is the **second genuine ML component**.

#### How Isolation Forest Works

The algorithm builds an ensemble of **random binary trees**. Anomalous points are isolated (reach a leaf node) in **fewer splits** than normal points, because they occupy sparse regions of the feature space.

The anomaly score for each service is:

```
s(x, n) = 2^( -E[h(x)] / c(n) )

where:
  E[h(x)] = expected path length across all isolation trees
  c(n)    = 2*H(n-1) - 2*(n-1)/n    (normalisation constant)
  H(i)    = ln(i) + 0.5772156649    (Euler-Mascheroni constant)

Scores → 1.0 : highly anomalous
Scores → 0.5 : normal behaviour
```

#### Why Telnet is Flagged as ANOMALY

Telnet has a **unique combination** that no other service shares:
- Maximum PRS (0.95) — top of the risk scale
- Maximum VES (1.00) — completely unknown version
- PEF = 1 — unencrypted

This multivariate combination is so far from the cluster of other services that the Isolation Forest isolates it almost immediately, giving it an anomaly score of **1.0**.

FTP also scores Critical via the Random Forest, but it has a known version string (VES = 0.25), so its feature vector is less isolated — the IF scores it as Normal. This demonstrates the **complementary** value of the two ML stages.

#### Output
```
Port  Service  Anomaly Score s(x,n)  Label
23    telnet   1.0000                ANOMALY  ← flagged
21    ftp      0.0000                Normal
80    http     0.1689                Normal
443   https    0.3950                Normal
22    ssh      0.0221                Normal
53    domain   0.2426                Normal
```

---

### Step 5: Risk Score Computation (Stage 4)

Three scores are computed using the RF-predicted tiers:

#### GWVS — Global Weighted Vulnerability Score
```
GWVS = [Σ w_Tier(i) × PRS_i] / [n × w_max] × 100

Severity weights: Critical=4, High=3, Medium=2, Low=1
w_max = 4 (Critical weight)
```

#### NEF — Network Exposure Factor
```
NEF = (H + P + S) / (H_max + P_max + S_max)

H = detected hosts, P = open ports, S = unique services
```

#### ARS — Adjusted Risk Score
```
ARS = GWVS × NEF
```

The NEF adjustment ensures that a network with more exposed hosts and services receives a higher final score than one with identical service vulnerabilities but a smaller footprint.

---

## Features Explained

| Feature | Full Name | Type | Range | Description |
|---------|-----------|------|-------|-------------|
| PRS | Port Risk Score | float | 0–1 | Historical CVE exposure likelihood from NIST NVD |
| VES | Version Entropy Score | float | 0–1 | Version information density; 1.0 = completely unknown |
| PEF | Protocol Encryption Flag | int | 0,1 | 1 = unencrypted transport (FTP/Telnet/HTTP) |
| PSC | Port Service Category | int | 0–5 | Service type: Web/Legacy/File/Remote/Database/Other |
| VKF | Version Known Flag | int | 0,1 | 1 = version string detected, 0 = unknown |
| CRI | Composite Risk Indicator | float | 0–3 | PRS+VES+PEF — display only, not used by ML |

---

## ML Models Explained

### Random Forest (Stage 2)

| Parameter | Value | Reason |
|-----------|-------|--------|
| n_estimators | 200 | More trees = more stable predictions |
| max_depth | 8 | Prevents overfitting on training data |
| class_weight | balanced | Handles Critical/Low class imbalance |
| Features | PRS, VES, PEF, PSC, VKF | 5 domain-informed features |
| Training records | 200 | NVD CVE distribution-matched |
| CV Accuracy | ~89% | 5-fold stratified cross-validation |
| CV F1-Macro | ~87% | Balanced across all 4 severity classes |

### Isolation Forest (Stage 3)

| Parameter | Value | Reason |
|-----------|-------|--------|
| n_estimators | 100 | Sufficient for stable anomaly scores |
| contamination | 0.40 | Lee & Park (2023): ~40% anomalous in misconfigured networks |
| Anomaly threshold τ | 0.60 | Sensitivity analysis on experimental data |
| Features | PRS, VES, PEF, PSC, VKF | Same 5-feature space |
| Training data | None required | Fully unsupervised |

---

## Installation

### On Kali Linux (Recommended)

```bash
# Clone the repository
git clone https://github.com/loyolite192652/vapt_repo.git
cd vapt_repo

# Install Python dependencies
pip install -r requirements.txt

# OR install individually
pip install pandas scikit-learn xmltodict numpy tabulate
```

### Optional: Virtual Environment

```bash
python3 -m venv vapt_env
source vapt_env/bin/activate
pip install -r requirements.txt
```

---

## Usage

### Step 1: Generate Nmap Scan

```bash
# Basic service + OS detection
sudo nmap -sV -O -oX scan_results.xml 192.168.1.1

# Aggressive scan (recommended for thorough assessment)
sudo nmap -A -oX scan_results.xml 192.168.1.1

# Scan entire subnet
sudo nmap -sV -O -oX scan_results.xml 192.168.1.0/24

# Include vulnerability scripts
sudo nmap -sV -O --script vuln -oX scan_results.xml 192.168.1.1
```

### Step 2: Run the AI/ML Pipeline

```bash
# Basic usage
python3 ml_analysis.py --xml scan_results.xml

# With full network exposure parameters (recommended for accurate ARS)
python3 ml_analysis.py --xml scan_results.xml \
    --hosts 5 \
    --max_hosts 10 \
    --max_ports 20 \
    --max_services 15

# Using the provided sample XML
python3 ml_analysis.py --xml examples/scan_results.xml
```

### Command-Line Arguments

| Argument | Required | Description | Default |
|----------|----------|-------------|---------|
| `--xml` | ✅ Yes | Path to Nmap XML file | — |
| `--hosts` | No | Number of active hosts detected | Count unique IPs in XML |
| `--max_hosts` | No | Maximum possible hosts in subnet | = detected hosts |
| `--max_ports` | No | Maximum possible open ports | = detected ports |
| `--max_services` | No | Maximum possible unique services | = detected services |

---

## Output Explained

The pipeline prints five sections to the terminal:

### Section 1: Stage Progress Logs
Each stage prints `[SUCCESS]`, `[INFO]`, or `[WARNING]` messages showing what was processed.

### Section 2: Feature Matrix Table
Shows computed PRS, VES, PEF, PSC, VKF, CRI for every open port.

### Section 3: Random Forest Results
- Cross-validation accuracy and F1 scores
- Per-class precision, recall, F1 on training data
- Feature importance bar chart
- Predicted tier + probability scores for every scan record

### Section 4: Isolation Forest Results
- Anomaly score s(x,n) for every service
- ANOMALY / Normal label per service

### Section 5: Final Vulnerability Report
```
════════════════════════════════════════════════════════════════════════
   FINAL AI-ASSISTED VULNERABILITY REPORT
════════════════════════════════════════════════════════════════════════

  OVERALL VULNERABILITY SCORE (GWVS)  :   XX.XX%
  NETWORK EXPOSURE FACTOR     (NEF)   :   X.XXXX
  ADJUSTED RISK SCORE         (ARS)   :   XX.XX%

  Critical  : X service(s)
  High      : X service(s)
  Medium    : X service(s)
  Low       : X service(s)
  Anomalies : X service(s)

HIGH-PRIORITY ITEMS  (Critical OR Anomaly)
  [table of critical/anomalous services with scores]

FULL SERVICE ASSESSMENT
  [all open ports ranked by CRI]

REMEDIATION RECOMMENDATIONS
  Port 23 (telnet): Disable Telnet IMMEDIATELY. Replace with SSH.
  Port 21 (ftp):    Disable FTP. Replace with SFTP or FTPS.
  ...
```

---

## Risk Score Formulas

```
GWVS = [Σ w_Tier(i) × PRS_i] / [n × 4] × 100

NEF  = (H + P + S) / (H_max + P_max + S_max)

ARS  = GWVS × NEF

Anomaly Score:  s(x, n) = 2^( -E[h(x)] / c(n) )
                c(n)    = 2×H(n-1) - 2×(n-1)/n
                H(i)    = ln(i) + 0.5772156649

VES(i) = 1 - len(version_i) / (max_len + ε)

CVSS v3.1 Qualitative ARS Rating:
  ARS ≥ 70%  →  HIGH SEVERITY
  ARS ≥ 40%  →  MEDIUM SEVERITY
  ARS < 40%  →  LOW SEVERITY
```

---

## Results

Experimental evaluation on a controlled network (6 open ports, 192.168.1.1):

| Metric | Value |
|--------|-------|
| Random Forest CV Accuracy | 89.00% ± 2.55% |
| Random Forest CV F1-Macro | 87.37% ± 3.06% |
| Risk Tier Classification Accuracy (vs NIST NVD GT) | 100% |
| Precision | 1.00 |
| Recall | 1.00 |
| F1-Score | 1.00 |
| Anomaly Detection (Telnet) | Correctly flagged (score = 1.0) |
| GWVS | 79.17% |
| NEF | 0.941 |
| ARS | 74.51% (HIGH SEVERITY) |

---

## Repository Structure

```
vapt_repo/
│
├── ml_analysis.py          ← Main pipeline (Stage 0–5)
├── requirements.txt        ← Python dependencies
├── README.md               ← This file
│
├── examples/
│   └── scan_results.xml    ← Sample Nmap XML (paper's experimental scan)
│
├── docs/
│   └── pipeline_explained.md  ← Detailed technical documentation
│
└── tests/
    └── test_pipeline.py    ← Unit tests for all pipeline functions
```





## Contact

**Naveen Kumar Bandla** — bandlanaveenkumar2000@gmail.com
**Dr. Y. Nasir Ahmed** — nasirahmed@chaitanya.edu.in

Department of Computer Science, Chaitanya Deemed To Be University, Hyderabad, India
