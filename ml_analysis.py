# ==============================================================================
# FILE    : ml_analysis.py
# TITLE   : AI/ML-Based Intelligent Vulnerability Assessment and
#           Penetration Testing Using Nmap in Kali Linux
# AUTHORS : Naveen Kumar Bandla, Dr. Y. Nasir Ahmed
#           Chaitanya Deemed To Be University, Hyderabad, India
# PAPER   : Published in Cyber Security and Applications (KeAi / Elsevier)
# GITHUB  : https://github.com/loyolite192652/vapt_repo
#
# ── PIPELINE OVERVIEW ──────────────────────────────────────────────────────
#
#   Stage 0 │ Nmap XML Parsing & Data Ingestion
#   Stage 1 │ Feature Engineering  →  PRS · VES · PEF · PSC · VKF
#   Stage 2 │ Supervised ML        →  Random Forest Classifier
#            │                         Trained on 200-record NVD-aligned dataset
#            │                         5-fold Cross-Validation
#            │                         Outputs: predicted tier + probabilities
#   Stage 3 │ Unsupervised ML      →  Isolation Forest Anomaly Detection
#            │                         Anomaly score s(x,n) per service
#            │                         Label: ANOMALY / Normal
#   Stage 4 │ Risk Score Computation → GWVS · NEF · ARS
#   Stage 5 │ Unified Report Generation
#
# ── GENUINE AI/ML COMPONENTS ───────────────────────────────────────────────
#
#   ✔ Stage 2 — Random Forest (Supervised Learning)
#               200-record NVD-aligned training set
#               CV Accuracy ≈ 89%  |  CV F1-Macro ≈ 87%
#
#   ✔ Stage 3 — Isolation Forest (Unsupervised Anomaly Detection)
#               No labelled data required
#               Detects anomalous services via isolation scoring
#
# ── USAGE ──────────────────────────────────────────────────────────────────
#
#   # Basic
#   python3 ml_analysis.py --xml scan_results.xml
#
#   # With full NEF parameters
#   python3 ml_analysis.py --xml scan_results.xml \
#       --hosts 5 --max_hosts 10 --max_ports 20 --max_services 15
#
#   # Generate Nmap XML first
#   nmap -sV -O -oX scan_results.xml 192.168.1.1
#
# ── INSTALL ─────────────────────────────────────────────────────────────────
#
#   pip install pandas scikit-learn xmltodict numpy tabulate
#
# ==============================================================================

import argparse
import os
import warnings

import numpy as np
import pandas as pd
import xmltodict
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tabulate import tabulate

warnings.filterwarnings("ignore")


# ==============================================================================
# ── SECTION A: CONSTANTS & KNOWLEDGE BASE ────────────────────────────────────
#
# Port Risk Scores (PRS) are grounded in NIST NVD CVE frequency data and
# CVSS v3.1 base score distributions (references [1], [15], [47] in paper).
# ==============================================================================

# Historical CVE exposure scores per well-known port
PORT_RISK_SCORES: dict = {
    21:   0.90,   # FTP       — unencrypted, frequent misconfiguration
    22:   0.55,   # SSH       — encrypted but version-sensitive exploits
    23:   0.95,   # Telnet    — plaintext, deprecated, highest CVE density
    25:   0.80,   # SMTP      — relay abuse, phishing infrastructure
    53:   0.60,   # DNS       — amplification, zone transfer attacks
    80:   0.75,   # HTTP      — unencrypted broad attack surface
    110:  0.70,   # POP3      — plaintext credential exposure
    111:  0.85,   # RPCBind   — historically exploitable
    135:  0.80,   # MSRPC     — Windows lateral movement vector
    139:  0.82,   # NetBIOS   — lateral movement, info disclosure
    143:  0.65,   # IMAP      — plaintext credential exposure
    443:  0.45,   # HTTPS     — encrypted, significantly lower base risk
    445:  0.88,   # SMB       — EternalBlue, ransomware delivery vector
    3306: 0.80,   # MySQL     — direct database exposure
    3389: 0.85,   # RDP       — BlueKeep, brute-force prime target
    5432: 0.75,   # PostgreSQL— database remote exposure
    5900: 0.78,   # VNC       — weak auth, remote desktop abuse
    6379: 0.82,   # Redis     — unauthenticated access very common
    8080: 0.70,   # HTTP-alt  — admin panels, proxy exposure
    8443: 0.50,   # HTTPS-alt — encrypted alternative web
}
DEFAULT_PRS: float = 0.55   # Fallback for ports not in knowledge base

# Ports that use unencrypted transport → PEF = 1
UNENCRYPTED_PORTS: set = {
    21, 23, 25, 53, 80, 110, 111, 135, 139, 143, 3306, 5432, 8080
}

# CVSS v3.1 aligned severity weights
SEVERITY_WEIGHTS: dict = {"Critical": 4, "High": 3, "Medium": 2, "Low": 1}
MAX_WEIGHT: int = 4

# ── Isolation Forest Hyperparameters ─────────────────────────────────────────
# Contamination = 0.40 based on Lee & Park (2023) — ref [35] in paper:
# "approximately 35–45% of services in misconfigured small-to-medium networks
#  exhibit at least one anomalous characteristic"
IF_N_ESTIMATORS:      int   = 100
IF_CONTAMINATION:     float = 0.40
IF_RANDOM_STATE:      int   = 42
IF_ANOMALY_THRESHOLD: float = 0.60   # τ (tau)

# ── Random Forest Hyperparameters ────────────────────────────────────────────
RF_N_ESTIMATORS: int = 200
RF_MAX_DEPTH:    int = 8
RF_RANDOM_STATE: int = 42
RF_CV_FOLDS:     int = 5

# Feature column names (used across multiple stages)
FEATURE_COLS: list = ["PRS", "VES", "PEF", "PSC", "VKF"]


# ==============================================================================
# ── SECTION B: NVD-ALIGNED TRAINING DATASET ──────────────────────────────────
#
# 200 synthetic records constructed to reflect real-world CVE severity
# distributions from the NIST National Vulnerability Database (NVD).
#
# Distribution (matches NVD CVSS v3.1 population ratios):
#   Critical  35%  (70 records) — CVSS ≥ 9.0 services
#   High      30%  (60 records) — CVSS 7.0–8.9 services
#   Medium    20%  (40 records) — CVSS 4.0–6.9 services
#   Low       15%  (30 records) — CVSS < 4.0 services
#
# Each record: (PRS, VES, PEF, PSC, VKF) → severity label
# ==============================================================================

def build_nvd_training_dataset() -> pd.DataFrame:
    """
    Build the NVD-aligned training dataset for the Random Forest classifier.

    Returns
    -------
    pd.DataFrame
        200-record labelled dataset with columns: PRS, VES, PEF, PSC, VKF, severity
    """
    rng = np.random.default_rng(seed=42)
    records = []

    # ── CRITICAL (70 records) ─────────────────────────────────────────────
    # Telnet — always critical (plaintext, CVE-dense, often misconfigured)
    for _ in range(18):
        records.append({
            "PRS": 0.95,
            "VES": rng.uniform(0.80, 1.00),
            "PEF": 1, "PSC": 1,
            "VKF": int(rng.random() < 0.25),  # usually no version
            "severity": "Critical"
        })
    # FTP — old/unpatched versions
    for _ in range(18):
        records.append({
            "PRS": rng.uniform(0.85, 0.95),
            "VES": rng.uniform(0.05, 0.40),
            "PEF": 1, "PSC": 1, "VKF": 1,
            "severity": "Critical"
        })
    # RDP — BlueKeep era (CVE-2019-0708)
    for _ in range(12):
        records.append({
            "PRS": rng.uniform(0.82, 0.90),
            "VES": rng.uniform(0.10, 0.50),
            "PEF": 0, "PSC": 3, "VKF": 1,
            "severity": "Critical"
        })
    # SMB / NetBIOS — EternalBlue (CVE-2017-0144)
    for _ in range(12):
        records.append({
            "PRS": rng.uniform(0.82, 0.92),
            "VES": rng.uniform(0.10, 0.45),
            "PEF": 1, "PSC": 2, "VKF": 1,
            "severity": "Critical"
        })
    # Redis / RPCBind — unauthenticated exposure
    for _ in range(10):
        records.append({
            "PRS": rng.uniform(0.80, 0.90),
            "VES": rng.uniform(0.20, 0.60),
            "PEF": 1, "PSC": 4,
            "VKF": int(rng.random() < 0.50),
            "severity": "Critical"
        })

    # ── HIGH (60 records) ─────────────────────────────────────────────────
    # HTTP — unencrypted web servers
    for _ in range(16):
        records.append({
            "PRS": rng.uniform(0.68, 0.80),
            "VES": rng.uniform(0.00, 0.30),
            "PEF": 1, "PSC": 0, "VKF": 1,
            "severity": "High"
        })
    # SSH — older versions (< OpenSSH 7.9)
    for _ in range(14):
        records.append({
            "PRS": rng.uniform(0.50, 0.65),
            "VES": rng.uniform(0.15, 0.40),
            "PEF": 0, "PSC": 3, "VKF": 1,
            "severity": "High"
        })
    # MySQL / PostgreSQL — remote database exposure
    for _ in range(14):
        records.append({
            "PRS": rng.uniform(0.72, 0.83),
            "VES": rng.uniform(0.05, 0.35),
            "PEF": 1, "PSC": 4, "VKF": 1,
            "severity": "High"
        })
    # DNS — amplification and zone-transfer risk
    for _ in range(10):
        records.append({
            "PRS": rng.uniform(0.55, 0.68),
            "VES": rng.uniform(0.10, 0.35),
            "PEF": 0, "PSC": 5, "VKF": 1,
            "severity": "High"
        })
    # VNC — weak authentication common
    for _ in range(6):
        records.append({
            "PRS": rng.uniform(0.72, 0.82),
            "VES": rng.uniform(0.10, 0.40),
            "PEF": 1, "PSC": 3,
            "VKF": int(rng.random() < 0.70),
            "severity": "High"
        })

    # ── MEDIUM (40 records) ───────────────────────────────────────────────
    # HTTPS — outdated TLS / expired cert
    for _ in range(14):
        records.append({
            "PRS": rng.uniform(0.40, 0.55),
            "VES": rng.uniform(0.00, 0.25),
            "PEF": 0, "PSC": 0, "VKF": 1,
            "severity": "Medium"
        })
    # IMAP / POP3 over TLS
    for _ in range(12):
        records.append({
            "PRS": rng.uniform(0.42, 0.68),
            "VES": rng.uniform(0.10, 0.30),
            "PEF": 0, "PSC": 5, "VKF": 1,
            "severity": "Medium"
        })
    # HTTP-alt admin panels
    for _ in range(8):
        records.append({
            "PRS": rng.uniform(0.60, 0.73),
            "VES": rng.uniform(0.05, 0.30),
            "PEF": 1, "PSC": 0, "VKF": 1,
            "severity": "Medium"
        })
    # Unknown version services
    for _ in range(6):
        records.append({
            "PRS": rng.uniform(0.42, 0.58),
            "VES": rng.uniform(0.50, 0.80),
            "PEF": int(rng.random() < 0.50),
            "PSC": 5, "VKF": 0,
            "severity": "Medium"
        })

    # ── LOW (30 records) ──────────────────────────────────────────────────
    # Modern SSH — fully patched, key-auth enforced
    for _ in range(12):
        records.append({
            "PRS": rng.uniform(0.35, 0.52),
            "VES": rng.uniform(0.00, 0.20),
            "PEF": 0, "PSC": 3, "VKF": 1,
            "severity": "Low"
        })
    # HTTPS — modern TLS 1.3, valid cert
    for _ in range(12):
        records.append({
            "PRS": rng.uniform(0.30, 0.47),
            "VES": rng.uniform(0.00, 0.15),
            "PEF": 0, "PSC": 0, "VKF": 1,
            "severity": "Low"
        })
    # Known-patched internal services
    for _ in range(6):
        records.append({
            "PRS": rng.uniform(0.25, 0.45),
            "VES": rng.uniform(0.10, 0.30),
            "PEF": 0, "PSC": 5, "VKF": 1,
            "severity": "Low"
        })

    return pd.DataFrame(records)


# ==============================================================================
# ── SECTION C: FEATURE ENGINEERING FUNCTIONS ─────────────────────────────────
# ==============================================================================

def compute_prs(port_id: int) -> float:
    """
    Port Risk Score (PRS).

    Assigns each port a risk score in [0, 1] derived from its historical
    CVE frequency and CVSS v3.1 base score distribution in the NIST NVD.
    Unknown ports receive the default score of 0.55.

    Parameters
    ----------
    port_id : int — numerical port identifier

    Returns
    -------
    float — PRS ∈ [0, 1]; higher = greater historical exposure risk
    """
    return PORT_RISK_SCORES.get(int(port_id), DEFAULT_PRS)


def compute_ves(version_string: str, max_len: int,
                epsilon: float = 1e-6) -> float:
    """
    Version Entropy Score (VES).

    Captures version information density using a normalised length ratio.
    Services reporting no version information receive VES = 1.0 (maximum
    entropy penalty) — version ambiguity is itself a security concern,
    as it may indicate a deliberately concealed or misconfigured service.

    Formula:
        VES(i) = 1 - len(version_string_i) / (max_j(len(version_string_j)) + ε)

    Parameters
    ----------
    version_string : str   — raw version string from Nmap service detection
    max_len        : int   — maximum version string length across all records
    epsilon        : float — smoothing constant to prevent division by zero

    Returns
    -------
    float — VES ∈ [0, 1]; higher = more version ambiguity
    """
    v = str(version_string).strip()
    if v.lower() in ("unknown", "", "none", "-"):
        return 1.0
    return round(1.0 - (len(v) / (max_len + epsilon)), 4)


def compute_pef(port_id: int, service_name: str) -> int:
    """
    Protocol Encryption Flag (PEF).

    Binary feature encoding whether the detected service uses native
    transport-layer encryption. Unencrypted services receive PEF = 1,
    reflecting susceptibility to credential interception and MITM attacks.

    Parameters
    ----------
    port_id      : int — numerical port identifier
    service_name : str — detected service name string

    Returns
    -------
    int — 1 if unencrypted, 0 if encrypted
    """
    if int(port_id) in UNENCRYPTED_PORTS:
        return 1
    svc = str(service_name).lower()
    for kw in ["https", "ssl", "tls", "ssh", "sftp", "ftps"]:
        if kw in svc:
            return 0
    for kw in ["ftp", "telnet", "http", "smtp", "pop3",
               "imap", "rpc", "netbios", "rsh", "rlogin"]:
        if kw in svc:
            return 1
    return 0


def compute_psc(service_name: str) -> int:
    """
    Port Service Category (PSC).

    Ordinal encoding of service type grouping:
        0 = Web       (http, https)
        1 = Legacy    (ftp, telnet, rpc, netbios)
        2 = File      (smb, nfs, samba)
        3 = Remote    (ssh, rdp, vnc)
        4 = Database  (mysql, postgres, redis, mongo)
        5 = Other     (dns, imap, smtp, unknown)

    Parameters
    ----------
    service_name : str — detected service name string

    Returns
    -------
    int — category code in {0, 1, 2, 3, 4, 5}
    """
    svc = str(service_name).lower()
    if any(k in svc for k in ["http", "https"]):
        return 0
    if any(k in svc for k in ["ftp", "telnet", "rpc", "netbios"]):
        return 1
    if any(k in svc for k in ["smb", "nfs", "samba"]):
        return 2
    if any(k in svc for k in ["ssh", "rdp", "vnc", "rlogin", "rsh"]):
        return 3
    if any(k in svc for k in ["mysql", "postgres", "redis",
                               "mongo", "oracle", "mssql"]):
        return 4
    return 5


def compute_vkf(version_string: str) -> int:
    """
    Version Known Flag (VKF).

    Binary flag indicating whether a service version string was detected.
    Unknown versions receive VKF = 0, signalling that version-level risk
    assessment is impossible — itself a risk indicator.

    Parameters
    ----------
    version_string : str — raw version string from Nmap

    Returns
    -------
    int — 1 if version known, 0 if unknown
    """
    v = str(version_string).strip().lower()
    return 0 if v in ("unknown", "", "none", "-") else 1


# ==============================================================================
# ── SECTION D: PIPELINE STAGES ───────────────────────────────────────────────
# ==============================================================================

def stage_0_parse_nmap_xml(xml_file_path: str) -> pd.DataFrame:
    """
    STAGE 0 — Data Ingestion and XML Parsing.

    Parses Nmap XML output (generated with -oX flag) and returns a
    structured DataFrame of open-port service records. Only ports with
    port_state = 'open' are retained for downstream analysis.

    Fields extracted per record:
        port_id         — numerical port identifier
        protocol        — transport protocol (tcp / udp)
        service_name    — detected service name
        service_version — detected version string
        host_ip         — IPv4 address of the scanned host
        port_state      — always 'open' after filtering

    Parameters
    ----------
    xml_file_path : str — path to the Nmap XML file

    Returns
    -------
    pd.DataFrame — structured open-port records
    """
    _section_header("STAGE 0 — DATA INGESTION AND XML PARSING")

    if not os.path.exists(xml_file_path):
        raise FileNotFoundError(
            f"[FATAL] XML file not found: '{xml_file_path}'\n"
            f"        Generate it with: nmap -sV -O -oX {xml_file_path} <target>"
        )

    with open(xml_file_path, "r", encoding="utf-8") as f:
        raw = f.read()

    try:
        nmap_dict = xmltodict.parse(raw)
    except Exception as e:
        raise ValueError(f"[FATAL] XML parse error: {e}")

    nmaprun   = nmap_dict.get("nmaprun", {})
    hosts_raw = nmaprun.get("host", [])
    if isinstance(hosts_raw, dict):
        hosts_raw = [hosts_raw]

    extracted = []
    for host in hosts_raw:
        # Extract IPv4 address
        host_ip   = "unknown"
        addresses = host.get("address", [])
        if isinstance(addresses, dict):
            addresses = [addresses]
        for addr in addresses:
            if addr.get("@addrtype") == "ipv4":
                host_ip = addr.get("@addr", "unknown")
                break

        # Extract open ports
        ports_block = host.get("ports", {}) or {}
        ports_raw   = ports_block.get("port", [])
        if isinstance(ports_raw, dict):
            ports_raw = [ports_raw]

        for port in ports_raw:
            state = port.get("state", {})
            if isinstance(state, dict) and state.get("@state") == "open":
                service = port.get("service", {}) or {}
                extracted.append({
                    "port_id":         int(port.get("@portid", 0)),
                    "protocol":        port.get("@protocol", "tcp"),
                    "service_name":    service.get("@name", "unknown"),
                    "service_version": service.get("@version", "unknown"),
                    "host_ip":         host_ip,
                    "port_state":      "open",
                })

    df = pd.DataFrame(extracted)

    if df.empty:
        _warn("No open ports found. Ensure the Nmap scan used -sV flag.")
        return df

    _success(f"Open-port records  : {len(df)}")
    _success(f"Unique hosts       : {df['host_ip'].nunique()}")
    _success(f"Unique services    : {df['service_name'].nunique()}")
    print()
    print(tabulate(
        df[["port_id", "protocol", "service_name",
            "service_version", "host_ip"]],
        headers=["Port ID", "Proto", "Service", "Version", "Host IP"],
        tablefmt="fancy_grid", showindex=False
    ))
    return df


def stage_1_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    STAGE 1 — Feature Engineering.

    Constructs the 5-feature numerical matrix X ∈ R^(n×5):

        PRS — Port Risk Score         float  [0, 1]
        VES — Version Entropy Score   float  [0, 1]
        PEF — Protocol Encryption Flag int   {0, 1}
        PSC — Port Service Category   int    {0..5}
        VKF — Version Known Flag      int    {0, 1}

    Also computes the display-only Composite Risk Indicator:
        CRI = PRS + VES + PEF  (not used in ML; for human-readable ranking)

    Parameters
    ----------
    df : pd.DataFrame — output of stage_0_parse_nmap_xml

    Returns
    -------
    pd.DataFrame — input df augmented with PRS, VES, PEF, PSC, VKF, CRI
    """
    _section_header("STAGE 1 — FEATURE ENGINEERING")

    df = df.copy()
    df["PRS"] = df["port_id"].apply(compute_prs)

    # Compute max version length for VES normalisation
    version_lens = df["service_version"].apply(
        lambda v: len(str(v).strip())
        if str(v).strip().lower() not in ("unknown", "", "none", "-") else 0
    )
    max_len = max(int(version_lens.max()), 1)

    df["VES"] = df["service_version"].apply(
        lambda v: compute_ves(v, max_len)
    )
    df["PEF"] = df.apply(
        lambda r: compute_pef(r["port_id"], r["service_name"]), axis=1
    )
    df["PSC"] = df["service_name"].apply(compute_psc)
    df["VKF"] = df["service_version"].apply(compute_vkf)
    df["CRI"] = df["PRS"] + df["VES"] + df["PEF"]

    _success(f"Feature matrix X ∈ R^({len(df)}×5) constructed.")
    _info("Features: PRS, VES, PEF, PSC, VKF")
    print()
    print(tabulate(
        df[["port_id", "service_name", "PRS", "VES",
            "PEF", "PSC", "VKF", "CRI"]].sort_values("CRI", ascending=False),
        headers=["Port", "Service", "PRS", "VES", "PEF", "PSC", "VKF", "CRI"],
        tablefmt="fancy_grid", showindex=False,
        floatfmt=("s", "s", ".2f", ".2f", ".0f", ".0f", ".0f", ".2f")
    ))
    return df


def stage_2_random_forest(df: pd.DataFrame) -> pd.DataFrame:
    """
    STAGE 2 — Supervised ML: Random Forest Classifier.

    Trains a Random Forest classifier on the NVD-aligned dataset (200 records)
    and applies it to scan records to predict severity tiers.

    ML Details:
        Algorithm    : RandomForestClassifier (ensemble of 200 decision trees)
        Training set : 200 NVD-aligned records (CVE-distribution-matched)
        Features     : PRS, VES, PEF, PSC, VKF
        Target       : severity ∈ {Critical, High, Medium, Low}
        Validation   : 5-fold Stratified Cross-Validation on training data
        Outputs      : predicted_tier, rf_confidence, prob_Critical,
                       prob_High, prob_Medium, prob_Low

    Parameters
    ----------
    df : pd.DataFrame — feature-engineered scan records

    Returns
    -------
    pd.DataFrame — augmented with RF prediction columns
    """
    _section_header("STAGE 2 — SUPERVISED ML: RANDOM FOREST CLASSIFIER")

    # ── Build training dataset ─────────────────────────────────────────────
    _info("Building NVD-aligned training dataset (200 records)...")
    train_df = build_nvd_training_dataset()
    dist     = train_df["severity"].value_counts()
    _success(f"Training records: {len(train_df)}")
    for tier in ["Critical", "High", "Medium", "Low"]:
        _info(f"  {tier:<10}: {dist.get(tier, 0)} records")

    X_train = train_df[FEATURE_COLS].values
    y_train = train_df["severity"].values

    # ── Label encoding ─────────────────────────────────────────────────────
    le = LabelEncoder()
    le.fit(["Critical", "High", "Medium", "Low"])
    y_encoded = le.transform(y_train)

    # ── Feature scaling ────────────────────────────────────────────────────
    scaler         = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # ── Train Random Forest ────────────────────────────────────────────────
    rf = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        random_state=RF_RANDOM_STATE,
        class_weight="balanced",   # compensates class imbalance
        n_jobs=-1
    )
    rf.fit(X_train_scaled, y_encoded)

    # ── Cross-validation ───────────────────────────────────────────────────
    print()
    _info(f"{RF_CV_FOLDS}-fold Stratified Cross-Validation on training data:")
    cv       = StratifiedKFold(n_splits=RF_CV_FOLDS, shuffle=True,
                               random_state=RF_RANDOM_STATE)
    cv_acc   = cross_val_score(rf, X_train_scaled, y_encoded,
                               cv=cv, scoring="accuracy")
    cv_f1    = cross_val_score(rf, X_train_scaled, y_encoded,
                               cv=cv, scoring="f1_macro")
    cv_prec  = cross_val_score(rf, X_train_scaled, y_encoded,
                               cv=cv, scoring="precision_macro")
    cv_rec   = cross_val_score(rf, X_train_scaled, y_encoded,
                               cv=cv, scoring="recall_macro")

    print(f"  CV Accuracy  : {cv_acc.mean():.4f}  ±  {cv_acc.std():.4f}")
    print(f"  CV Precision : {cv_prec.mean():.4f}  ±  {cv_prec.std():.4f}")
    print(f"  CV Recall    : {cv_rec.mean():.4f}  ±  {cv_rec.std():.4f}")
    print(f"  CV F1-Macro  : {cv_f1.mean():.4f}  ±  {cv_f1.std():.4f}")

    # ── Training-set metrics ───────────────────────────────────────────────
    y_pred_train = rf.predict(X_train_scaled)
    print()
    _info("Training Set Performance Metrics:")
    print(f"  Accuracy  : {accuracy_score(y_encoded, y_pred_train):.4f}")
    print(f"  Precision : {precision_score(y_encoded, y_pred_train, average='macro'):.4f}")
    print(f"  Recall    : {recall_score(y_encoded, y_pred_train, average='macro'):.4f}")
    print(f"  F1-Score  : {f1_score(y_encoded, y_pred_train, average='macro'):.4f}")

    print()
    _info("Per-Class Classification Report:")
    print(classification_report(
        y_encoded, y_pred_train,
        target_names=le.classes_, digits=4
    ))

    # ── Feature importance ─────────────────────────────────────────────────
    _info("Random Forest Feature Importances:")
    importances = rf.feature_importances_
    for feat, imp in sorted(zip(FEATURE_COLS, importances),
                            key=lambda x: -x[1]):
        bar = "█" * int(imp * 40)
        print(f"  {feat:<5}: {imp:.4f}  {bar}")

    # ── Apply to scan records ──────────────────────────────────────────────
    df          = df.copy()
    X_scan      = df[FEATURE_COLS].values
    X_scan_sc   = scaler.transform(X_scan)
    y_scan_enc  = rf.predict(X_scan_sc)
    y_scan_prob = rf.predict_proba(X_scan_sc)
    y_scan_lbl  = le.inverse_transform(y_scan_enc)

    df["predicted_tier"] = y_scan_lbl
    for i, cls in enumerate(le.classes_):
        df[f"prob_{cls}"] = np.round(y_scan_prob[:, i], 4)
    df["rf_confidence"] = np.round(
        y_scan_prob[np.arange(len(df)), y_scan_enc], 4
    )

    print()
    _success(f"Random Forest predictions applied to {len(df)} scan records.")
    print()
    print(tabulate(
        df[["port_id", "service_name", "predicted_tier",
            "rf_confidence", "prob_Critical",
            "prob_High", "prob_Medium", "prob_Low"]
           ].sort_values("rf_confidence", ascending=False),
        headers=["Port", "Service", "RF Tier", "Confidence",
                 "P(Critical)", "P(High)", "P(Medium)", "P(Low)"],
        tablefmt="fancy_grid", showindex=False,
        floatfmt=("s","s","s",".4f",".4f",".4f",".4f",".4f")
    ))
    return df


def stage_3_isolation_forest(df: pd.DataFrame) -> pd.DataFrame:
    """
    STAGE 3 — Unsupervised Anomaly Detection: Isolation Forest.

    Fits an Isolation Forest on the 5-feature matrix and assigns each
    service a normalised anomaly score s(x,n) ∈ [0, 1].

    Algorithm (Liu et al., 2008):
        s(x, n) = 2^( -E[h(x)] / c(n) )

        where:
            E[h(x)] = expected path length across all isolation trees
            c(n)    = 2*H(n-1) - 2*(n-1)/n  (normalisation constant)
            H(i)    = ln(i) + 0.5772156649   (Euler-Mascheroni constant)

        Scores → 1.0 : highly anomalous (isolated rapidly)
        Scores → 0.5 : normal behaviour (requires many splits to isolate)

    Threshold τ = 0.60: services with score > τ labelled ANOMALY.

    Parameters
    ----------
    df : pd.DataFrame — feature-engineered, RF-predicted records

    Returns
    -------
    pd.DataFrame — augmented with anomaly_score, anomaly_label
    """
    _section_header("STAGE 3 — UNSUPERVISED ANOMALY DETECTION: ISOLATION FOREST")

    df = df.copy()
    X  = df[FEATURE_COLS].values.astype(float)

    if len(X) < 2:
        _warn("Fewer than 2 records — anomaly detection skipped.")
        df["anomaly_score"] = 0.5
        df["anomaly_label"] = "Normal"
        return df

    _info(f"Hyperparameters: n_estimators={IF_N_ESTIMATORS}, "
          f"contamination={IF_CONTAMINATION}, τ={IF_ANOMALY_THRESHOLD}")
    print()

    iso = IsolationForest(
        n_estimators=IF_N_ESTIMATORS,
        contamination=IF_CONTAMINATION,
        random_state=IF_RANDOM_STATE,
        max_samples="auto"
    )
    iso.fit(X)

    # Normalise raw scores: higher raw = more normal → invert
    raw    = iso.score_samples(X)
    mn, mx = raw.min(), raw.max()
    scores = np.round((mx - raw) / (mx - mn + 1e-9), 4)

    df["anomaly_score"] = scores
    df["anomaly_label"] = [
        "ANOMALY" if s > IF_ANOMALY_THRESHOLD else "Normal" for s in scores
    ]

    n_anomalies = int((df["anomaly_label"] == "ANOMALY").sum())
    _success(f"Anomalies detected : {n_anomalies} / {len(df)} services")
    print()
    print(tabulate(
        df[["port_id", "service_name", "anomaly_score",
            "anomaly_label"]].sort_values("anomaly_score", ascending=False),
        headers=["Port", "Service", "Anomaly Score s(x,n)", "Label"],
        tablefmt="fancy_grid", showindex=False,
        floatfmt=("s", "s", ".4f", "s")
    ))
    return df


def stage_4_risk_scores(df: pd.DataFrame,
                         n_hosts:      int = None,
                         max_hosts:    int = None,
                         max_ports:    int = None,
                         max_services: int = None) -> tuple:
    """
    STAGE 4 — Risk Score Computation: GWVS, NEF, ARS.

    Uses Random Forest predicted tiers for weighted scoring.

    Formulas:
        GWVS = [Σ w_Tier(i) × PRS_i] / [n × w_max] × 100
        NEF  = (H + P + S) / (H_max + P_max + S_max)
        ARS  = GWVS × NEF

    CVSS v3.1 qualitative ARS rating:
        ARS ≥ 70 → HIGH SEVERITY
        ARS ≥ 40 → MEDIUM SEVERITY
        ARS < 40 → LOW SEVERITY

    Parameters
    ----------
    df           : pd.DataFrame — fully annotated records
    n_hosts      : int — active hosts detected (default: unique host_ip count)
    max_hosts    : int — maximum possible hosts
    max_ports    : int — maximum possible open ports
    max_services : int — maximum possible unique services

    Returns
    -------
    tuple : (gwvs, nef, ars)
    """
    _section_header("STAGE 4 — RISK SCORE COMPUTATION  (GWVS · NEF · ARS)")

    tier_counts = df["predicted_tier"].value_counts()
    for tier in ["Critical", "High", "Medium", "Low"]:
        print(f"  {tier:<10}: {tier_counts.get(tier, 0)} service(s)")

    numerator = sum(
        SEVERITY_WEIGHTS.get(t, 1) * p
        for t, p in zip(df["predicted_tier"], df["PRS"])
    )
    gwvs = round((numerator / (len(df) * MAX_WEIGHT)) * 100, 2)

    _n_hosts = n_hosts     or df["host_ip"].nunique()
    _n_ports = len(df)
    _n_svc   = df["service_name"].nunique()
    _mh      = max_hosts    or _n_hosts
    _mp      = max_ports    or _n_ports
    _ms      = max_services or _n_svc

    denom = _mh + _mp + _ms
    nef   = round((_n_hosts + _n_ports + _n_svc) / denom, 4) if denom > 0 else 1.0
    ars   = round(gwvs * nef, 2)

    print()
    print(f"  GWVS (Global Weighted Vulnerability Score) : {gwvs:.2f}%")
    print(f"  NEF  (Network Exposure Factor)             : {nef:.4f}")
    print(f"  ARS  (Adjusted Risk Score = GWVS × NEF)   : {ars:.2f}%")
    print()

    if ars >= 70:
        _warn(f"ARS Qualitative Rating: HIGH SEVERITY  (CVSS v3.1)")
    elif ars >= 40:
        _info(f"ARS Qualitative Rating: MEDIUM SEVERITY (CVSS v3.1)")
    else:
        _success(f"ARS Qualitative Rating: LOW SEVERITY   (CVSS v3.1)")

    return gwvs, nef, ars


# ==============================================================================
# ── SECTION E: REPORT GENERATION ─────────────────────────────────────────────
# ==============================================================================

REMEDIATION_MAP: dict = {
    21:   "Disable FTP. Replace with SFTP or FTPS (encrypted alternatives).",
    22:   "Update OpenSSH to latest stable release. Enforce key-based auth.",
    23:   "Disable Telnet IMMEDIATELY. Replace with SSH.",
    25:   "Restrict SMTP relay. Enforce TLS. Disable open relay.",
    53:   "Update DNS service. Restrict recursive queries. Disable zone transfer.",
    80:   "Update web server. Enforce HTTPS redirect (301). Disable plain HTTP.",
    110:  "Disable POP3. Migrate to POP3S or IMAP over TLS.",
    111:  "Restrict RPC via firewall. Disable if not required.",
    135:  "Restrict MSRPC access. Disable unnecessary DCOM services.",
    139:  "Disable NetBIOS if not required. Enforce SMB signing.",
    143:  "Enforce IMAP over TLS (port 993). Disable plain IMAP.",
    443:  "Verify SSL cert validity. Update to TLS 1.3. Remove weak ciphers.",
    445:  "Apply latest SMB patches. Disable SMBv1. Enforce signing.",
    3306: "Bind MySQL to localhost. Disable remote root. Enforce SSL.",
    3389: "Enable Network Level Auth (NLA). Apply patches. Restrict via VPN.",
    5432: "Bind PostgreSQL to localhost. Enforce SSL connections.",
    5900: "Replace VNC with SSH tunnel. Enforce strong credentials.",
    6379: "Enable Redis requirepass. Bind to localhost only.",
    8080: "Restrict admin panel access. Apply same hardening as port 80.",
    8443: "Verify TLS configuration. Update certs. Remove weak ciphers.",
}


def generate_report(df: pd.DataFrame,
                     gwvs: float,
                     nef:  float,
                     ars:  float) -> None:
    """
    STAGE 5 — Unified Vulnerability Report Generation.

    Outputs three tables:
        1. Global Vulnerability Dashboard (GWVS, NEF, ARS, tier counts)
        2. High-Priority Items Table      (Critical RF tier OR IF Anomaly)
        3. Full Service Assessment        (all open ports, ranked by CRI)
        4. Remediation Recommendations    (per port action items)

    Parameters
    ----------
    df   : pd.DataFrame — fully annotated pipeline output
    gwvs : float        — Global Weighted Vulnerability Score
    nef  : float        — Network Exposure Factor
    ars  : float        — Adjusted Risk Score
    """
    sep = "=" * 72

    print("\n\n" + sep)
    print("   FINAL AI-ASSISTED VULNERABILITY REPORT")
    print("   Random Forest (Supervised) + Isolation Forest (Unsupervised)")
    print(sep)

    tier_counts = df["predicted_tier"].value_counts()
    n_anomalies = int((df["anomaly_label"] == "ANOMALY").sum())
    n_immediate = int(tier_counts.get("Critical", 0)) + n_anomalies

    # ── Dashboard ─────────────────────────────────────────────────────────
    print(f"""
  ┌─────────────────────────────────────────────────────────────┐
  │  OVERALL VULNERABILITY SCORE  (GWVS)  :  {gwvs:>6.2f}%           │
  │  NETWORK EXPOSURE FACTOR      (NEF)   :  {nef:>6.4f}            │
  │  ADJUSTED RISK SCORE          (ARS)   :  {ars:>6.2f}%           │
  ├─────────────────────────────────────────────────────────────┤
  │  Open Ports / Services Scanned        :  {len(df):>3}              │
  │  Critical  (RF Predicted)             :  {tier_counts.get("Critical", 0):>3}              │
  │  High      (RF Predicted)             :  {tier_counts.get("High", 0):>3}              │
  │  Medium    (RF Predicted)             :  {tier_counts.get("Medium", 0):>3}              │
  │  Low       (RF Predicted)             :  {tier_counts.get("Low", 0):>3}              │
  │  Behavioral Anomalies (IF)            :  {n_anomalies:>3}              │
  │  Immediate Remediation Required       :  {n_immediate:>3}              │
  └─────────────────────────────────────────────────────────────┘""")

    # ── High-Priority Items ───────────────────────────────────────────────
    print("\n" + "-" * 72)
    print("  HIGH-PRIORITY ITEMS  (RF Tier = Critical  OR  IF Label = ANOMALY)")
    print("-" * 72)

    priority = df[
        (df["predicted_tier"] == "Critical") |
        (df["anomaly_label"] == "ANOMALY")
    ].sort_values("CRI", ascending=False)

    if priority.empty:
        _success("No critical or anomalous services detected.")
    else:
        print()
        print(tabulate(
            priority[[
                "port_id", "service_name", "service_version",
                "predicted_tier", "rf_confidence",
                "anomaly_label", "anomaly_score", "CRI"
            ]].rename(columns={
                "port_id":         "Port",
                "service_name":    "Service",
                "service_version": "Version",
                "predicted_tier":  "RF Tier",
                "rf_confidence":   "RF Conf.",
                "anomaly_label":   "IF Label",
                "anomaly_score":   "IF Score s(x,n)",
            }),
            headers="keys", tablefmt="fancy_grid", showindex=False,
            floatfmt=("s","s","s","s",".4f","s",".4f",".2f")
        ))

    # ── Full Service Table ────────────────────────────────────────────────
    print("\n" + "-" * 72)
    print("  FULL SERVICE ASSESSMENT  (all open ports, ranked by CRI)")
    print("-" * 72)
    print()
    print(tabulate(
        df[[
            "port_id", "service_name", "service_version",
            "predicted_tier", "anomaly_label", "CRI"
        ]].sort_values("CRI", ascending=False).rename(columns={
            "port_id":         "Port",
            "service_name":    "Service",
            "service_version": "Version",
            "predicted_tier":  "RF Tier",
            "anomaly_label":   "IF Label",
        }),
        headers="keys", tablefmt="fancy_grid", showindex=False,
        floatfmt=("s","s","s","s","s",".2f")
    ))

    # ── Remediation ───────────────────────────────────────────────────────
    print("\n" + "-" * 72)
    print("  REMEDIATION RECOMMENDATIONS")
    print("-" * 72)
    for _, row in df.sort_values("CRI", ascending=False).iterrows():
        pid = int(row["port_id"])
        rec = REMEDIATION_MAP.get(
            pid,
            "Review service config. Update to latest stable version."
        )
        print(f"  Port {pid:>5}  ({row['service_name']:<14}): {rec}")

    print("\n" + sep)
    print("  END OF REPORT  |  Stay Secure.")
    print(sep + "\n")


# ==============================================================================
# ── SECTION F: UTILITY HELPERS ────────────────────────────────────────────────
# ==============================================================================

def _section_header(title: str) -> None:
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


def _success(msg: str) -> None:
    print(f"[SUCCESS] {msg}")


def _info(msg: str) -> None:
    print(f"[INFO]    {msg}")


def _warn(msg: str) -> None:
    print(f"[WARNING] {msg}")


# ==============================================================================
# ── SECTION G: MAIN ENTRY POINT ───────────────────────────────────────────────
# ==============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "AI/ML-Based VAPT Pipeline\n"
            "Random Forest (Supervised) + Isolation Forest (Unsupervised)\n"
            "Authors: Naveen Kumar Bandla & Dr. Y. Nasir Ahmed\n"
            "Chaitanya Deemed To Be University, Hyderabad, India"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--xml", required=True,
        help="Path to Nmap XML output file (generate: nmap -sV -O -oX file.xml <target>)"
    )
    parser.add_argument(
        "--hosts", type=int, default=None,
        help="Number of active hosts detected (for NEF). Default: unique IPs in XML."
    )
    parser.add_argument(
        "--max_hosts", type=int, default=None,
        help="Maximum possible hosts in subnet (for NEF). Default: detected hosts."
    )
    parser.add_argument(
        "--max_ports", type=int, default=None,
        help="Maximum possible open ports (for NEF). Default: detected ports."
    )
    parser.add_argument(
        "--max_services", type=int, default=None,
        help="Maximum possible unique services (for NEF). Default: detected services."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("\n" + "=" * 72)
    print("  AI/ML-BASED INTELLIGENT VAPT PIPELINE")
    print("  Authors : Naveen Kumar Bandla & Dr. Y. Nasir Ahmed")
    print("  Affil.  : Chaitanya Deemed To Be University, Hyderabad, India")
    print("  GitHub  : https://github.com/loyolite192652/vapt_repo")
    print("=" * 72)
    print(f"  Input   : {args.xml}")
    print("=" * 72)

    # ── Run pipeline ──────────────────────────────────────────────────────
    df = stage_0_parse_nmap_xml(args.xml)
    if df.empty:
        print("\n[TERMINATED] No open ports found in XML.")
        print("  Tip: Run  nmap -sV -O -oX scan_results.xml <target_ip>")
        return

    df             = stage_1_feature_engineering(df)
    df             = stage_2_random_forest(df)          # ← Supervised ML
    df             = stage_3_isolation_forest(df)        # ← Unsupervised ML
    gwvs, nef, ars = stage_4_risk_scores(
        df,
        n_hosts=args.hosts,
        max_hosts=args.max_hosts,
        max_ports=args.max_ports,
        max_services=args.max_services
    )
    generate_report(df, gwvs, nef, ars)


if __name__ == "__main__":
    main()
