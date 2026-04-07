# Pipeline Technical Documentation

## docs/pipeline_explained.md

This document provides deep technical explanations of every component in `ml_analysis.py` for readers who want to understand exactly how the pipeline works internally.

---

## Stage 0 — XML Parsing

Nmap saves scan results in XML using the `-oX` flag. The structure is:

```xml
<nmaprun>
  <host>
    <address addr="192.168.1.1" addrtype="ipv4"/>
    <ports>
      <port protocol="tcp" portid="21">
        <state state="open"/>
        <service name="ftp" version="vsftpd 1.4.1"/>
      </port>
    </ports>
  </host>
</nmaprun>
```

The `xmltodict` library converts this to a Python dictionary. The pipeline then navigates the dictionary tree, extracting one record per open port. Ports with `state = closed` or `state = filtered` are discarded — they do not represent active attack surface.

**Edge cases handled:**
- Single host returns a dict (not a list) → normalised to list
- Single port returns a dict → normalised to list
- Missing version field → defaults to "unknown"
- Multiple hosts in the same XML → all processed

---

## Stage 1 — Feature Engineering in Detail

### PRS (Port Risk Score)

PRS values are derived from analysis of NIST NVD CVE records filtered by port/service association:

```
Port 23 (Telnet) → PRS = 0.95
  Reason: Highest CVE density of any standard service port.
          CVE-2020-10188, CVE-2011-4862, and many others.
          Sends ALL data including credentials in plaintext.
          CVSS base scores frequently ≥ 9.0.

Port 21 (FTP) → PRS = 0.90
  Reason: vsftpd 2.3.4 backdoor (CVE-2011-2523), ProFTPD exploits.
          Anonymous login often misconfigured.
          No encryption = credential interception trivial.

Port 443 (HTTPS) → PRS = 0.45
  Reason: Encrypted by design. Risk is lower but not zero —
          SSL configuration issues, expired certs, weak ciphers.
          Heartbleed (CVE-2014-0160) was HTTPS but version-specific.
```

### VES (Version Entropy Score)

The formula captures information density of version strings:

```
VES(i) = 1 - len(version_string_i) / (max_j(len(version_string_j)) + ε)
```

Examples from the paper's experimental scan (max_len = 16 for "lighttpd 0.93.15"):

```
Service   Version            len   VES
telnet    unknown              0   1.000  ← can't assess, max penalty
ftp       vsftpd 1.4.1        12  0.250
ssh       OpenSSH 7.4          11  0.313
domain    dnsmasq 2.80          11  0.313
http      lighttpd 0.93.15     16  0.000  ← max info, min entropy
https     lighttpd 0.93.15     16  0.000
```

### PEF (Protocol Encryption Flag)

Decision logic (in order):
1. Check port ID against `UNENCRYPTED_PORTS` set → immediate classification
2. Check service name for encrypted keywords (https, ssl, tls, ssh, sftp, ftps)
3. Check service name for unencrypted keywords (ftp, telnet, http, smtp, ...)
4. Default: 0 (assume encrypted if unknown)

### PSC (Port Service Category)

Ordinal categories enable the Random Forest to learn service-type patterns:

```
0 = Web       : http, https
1 = Legacy    : ftp, telnet, rpc, netbios
2 = File      : smb, nfs, samba
3 = Remote    : ssh, rdp, vnc
4 = Database  : mysql, postgres, redis, mongo
5 = Other     : dns, imap, smtp, unknown
```

### VKF (Version Known Flag)

Binary flag distinct from VES:
- VES = 0.0 means the version string is maximally informative (long and specific)
- VES = 1.0 means no version information at all
- VKF = 0 means "we know nothing about the version" — useful for the RF to learn that unknown-version services need different treatment

---

## Stage 2 — Random Forest: Training Data Construction

The NVD-aligned dataset uses `numpy.random.default_rng(seed=42)` for full reproducibility. Each severity class is constructed to reflect realistic feature distributions:

### Critical Records (70)
- **Telnet records (18)**: PRS fixed at 0.95, VES sampled from [0.80, 1.00] (usually no version), PEF=1, PSC=1 (Legacy), VKF mostly 0
- **FTP records (18)**: PRS sampled from [0.85, 0.95], VES from [0.05, 0.40] (version usually known), PEF=1
- **RDP records (12)**: PRS [0.82, 0.90], PEF=0 (RDP uses its own encryption), PSC=3 (Remote)
- **SMB records (12)**: PRS [0.82, 0.92], PEF=1, PSC=2 (File sharing)
- **Redis/RPC records (10)**: PRS [0.80, 0.90], PEF=1, PSC=4 (Database)

### Why Synthetic Rather Than Real CVE Records?

Real NVD records contain vulnerability *descriptions* (text), not port-level scan metadata. Converting NVD text to (PRS, VES, PEF, PSC, VKF) tuples would require a separate NLP pipeline. The synthetic dataset instead directly encodes the *statistical properties* of CVE severity distributions for each service type — achieving the same classification signal without requiring NVD API access during deployment.

---

## Stage 3 — Isolation Forest: Mathematical Details

### Path Length and Isolation

In a random binary tree, a data point is isolated when the recursive partitioning process assigns it to a leaf. The path length `h(x)` is the number of edges from the root to the leaf where `x` is isolated.

**Anomalous points** (outliers) are isolated in fewer splits because they occupy low-density regions — there are few other points nearby, so random partitions quickly separate them.

**Normal points** are embedded in dense clusters. Random partitions rarely isolate them from their neighbours, resulting in longer path lengths.

### Normalisation

The raw `score_samples()` output from scikit-learn returns negative anomaly scores (more negative = more anomalous). We convert to the s(x,n) convention where 1.0 = most anomalous:

```python
raw    = iso.score_samples(X)    # higher = more normal (sklearn convention)
scores = (raw.max() - raw) / (raw.max() - raw.min() + ε)  # invert + normalise
```

### Contamination Parameter

`contamination=0.40` means the algorithm expects ~40% of the training data to be anomalous. This is set based on Lee & Park (2023), who found that 35–45% of services in misconfigured small-to-medium networks exhibit anomalous characteristics. In the experimental scan of 6 services, this means the algorithm expects ~2-3 anomalies — and correctly identifies Telnet as the primary one.

### Threshold τ = 0.60

Services with `anomaly_score > 0.60` are labelled ANOMALY. This threshold was determined through sensitivity analysis on the experimental dataset, balancing:
- False positive rate (incorrectly flagging normal services)
- Detection recall (correctly catching anomalous services)

---

## Stage 4 — Risk Score Formulas

### GWVS

```
GWVS = [Σ_{i=1}^{n} w_{Tier(i)} × PRS_i] / [n × w_max] × 100

Denominator uses n × w_max (not Σ w_max) to ensure proper normalisation:
if all n services were Critical with PRS=1.0, GWVS = 100%.
```

### NEF

```
NEF = (H + P + S) / (H_max + P_max + S_max)

H = number of active hosts detected
P = number of open ports detected
S = number of unique services detected
H_max, P_max, S_max = maximum values (reference upper bounds)
```

NEF ∈ (0, 1]. A network with NEF ≈ 1.0 has nearly maximum possible exposure across all three dimensions.

### ARS

```
ARS = GWVS × NEF
```

ARS is always ≤ GWVS. A network with low exposure (few hosts, few ports) gets a lower ARS than one with the same service vulnerabilities but higher exposure — correctly penalising wider attack surfaces.

---

## Remediation Logic

Remediation recommendations are generated from `REMEDIATION_MAP` — a dictionary mapping port numbers to action items. Services not in the map receive a generic recommendation.

The recommendations are ordered by **CRI** (Composite Risk Indicator), not by RF predicted tier or IF anomaly score, to ensure the most impactful services appear first regardless of which ML component flagged them.

---

## Reproducibility

All random operations use fixed seeds:
- `numpy.random.default_rng(seed=42)` — training dataset generation
- `RandomForestClassifier(random_state=42)` — RF training
- `IsolationForest(random_state=42)` — IF anomaly detection
- `StratifiedKFold(random_state=42)` — cross-validation

This ensures identical output on every run with the same input XML.
