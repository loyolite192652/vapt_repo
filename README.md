AI/ML-Based VAPT Pipeline v4.1 — Nmap Vulnerability Assessment
![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Platform: Kali Linux](https://img.shields.io/badge/platform-Kali%20Linux-purple.svg)
![Models: 6 ML](https://img.shields.io/badge/models-6%20ML-orange.svg)
100% machine learning — zero rule-based logic — zero hardcoded severity lookups.
This tool converts raw Nmap XML output into a ranked, scored vulnerability intelligence report in seconds. Six trained ML models derive every severity decision from NIST NVD CVE data. No commercial licences, no cloud dependencies, native Kali Linux deployment.
---
Publication
> **Naveen Kumar Bandla & Dr. Y. Nasir Ahmed**  
> *AI/ML-Based Intelligent Vulnerability Assessment and Penetration Testing Using Nmap in Kali Linux: A Five-Model Pipeline with NVD Integration, Real-World Validation, and Comparative Evaluation Against OpenVAS and Nessus*  
> **Cyber Security and Applications** (KeAi / Elsevier), 2025  
> GitHub: https://github.com/loyolite192652/vapt_repo
---
Quick Start
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Scan a target with Nmap (any output filename works)
nmap -sV -O -oX scan.xml 192.168.1.1

# 3. Run the ML pipeline
python3 ml_analysis.py --xml scan.xml

# 4. Offline mode (no internet required)
python3 ml_analysis.py --xml scan.xml --no-nvd

# 5. Save report to file
python3 ml_analysis.py --xml scan.xml --output report.txt

# 6. JSON output for integration
python3 ml_analysis.py --xml scan.xml --format json
```
---
ML Models
#	Model	Output	Description
1	Ridge Regression	PRS ∈ [0,1]	Port Risk Score from live NIST NVD CVE data
2	Logistic Regression + TF-IDF	PEF ∈ {0,1}	Protocol Encryption Flag (1=plaintext=dangerous)
3	Multinomial Naive Bayes	PSC ∈ {0–5}	Port Service Category (Web/Legacy/File-Share/Remote/DB/Infra)
4	Linear SVM	VKF ∈ {0,1}	Version Known Flag (0=version concealed=suspicious)
5	Random Forest	Tier ∈ {Critical/High/Medium/Low}	4-tier severity, 1,000-record NVD-aligned dataset
6	Isolation Forest	Score ∈ [0,100]	Unsupervised behavioural anomaly detection
+	TF-IDF cosine	NIST SP 800-53 mapping	NLP-based remediation recommendation
VES (Version Entropy Score) is derived via Shannon entropy: `VES = 0.70 × VES_len + 0.30 × VES_entropy`
---
Command-Line Reference
```
python3 ml_analysis.py --xml <any_filename>.xml [OPTIONS]

Required:
  --xml <file>         Nmap XML output file (any filename ending in .xml)

Options:
  --nvd-key KEY        NIST NVD API key for 50 req/30s (vs 5 req/30s default)
  --no-nvd             Offline mode — skip NVD API, use pre-trained Ridge model
  --refresh-cache      Delete NVD cache and force fresh API queries
  --hosts N            Active host count for NEF calculation
  --max-hosts N        Max hosts in scope (NEF denominator)
  --max-ports N        Max ports in scope (NEF denominator)
  --max-services N     Max service types in scope (NEF denominator)
  --output <file>.txt  Save report to text file
  --format {text,json} Output format (default: text)
  --version            Show version number and exit
```
The `--xml` argument accepts any filename: `scan.xml`, `my_results_2025.xml`, `target_network.xml`, etc. The tool is fully filename-agnostic.
---
Risk Scores
Three scalar scores characterise the assessed network:
```
GWVS = [Σ(w_tier × PRS_i) / (n × 4)] × 100  [%]
NEF  = (H + P + S) / (H_max + P_max + S_max)  [0, 1]
ARS  = GWVS × NEF  [%]

ARS ≥ 70%  : HIGH SEVERITY     — immediate action required
ARS 40–69% : MEDIUM-HIGH       — prioritised remediation needed
ARS 20–39% : MEDIUM            — planned remediation appropriate
ARS < 20%  : LOW               — routine maintenance sufficient
```
Where: `w_tier` = Critical:4 / High:3 / Medium:2 / Low:1 (CVSS-aligned); `H` = active hosts; `P` = open ports; `S` = unique service types.
---
Experimental Results
Local Virtual Testbed (6 services, 192.168.1.1)
Port	Service	Version	RF Tier	Confidence	IF Score	IF Label
23	telnet	unknown	Critical	100.00%	100.00	ANOMALY
21	ftp	vsftpd 1.4.1	Critical	99.74%	0.00	Normal
80	http	lighttpd 0.93.15	High	94.93%	16.89	Normal
22	ssh	OpenSSH 7.4	High	93.84%	2.21	Normal
53	domain	dnsmasq 2.80	High	75.07%	24.26	Normal
443	https	lighttpd 0.93.15	Medium	50.30%	39.50	Normal
GWVS=58.33% · NEF=0.9412 · ARS=54.90% (MEDIUM-HIGH)
scanme.nmap.org
Port	Service	RF Tier	IF Score	Label
22	ssh (OpenSSH 6.6.1p1)	High	4.10	Normal
80	http (Apache 2.4.7)	High	19.32	Normal
9929	nping-echo	Medium	42.15	Normal
31337	tcpwrapped	Critical	88.72	ANOMALY
ARS = 61.47% (MEDIUM-HIGH)
pentest-ground.com (8 services)
4× Critical (FTP, Telnet, MySQL, PostgreSQL) · 1× ANOMALY (Telnet)  
ARS = 72.14% (HIGH SEVERITY)
OWASP Juice Shop (Docker)
1 service: Node.js 18.17.1 / Express 4.18.2 — Medium  
ARS = 37.20% (MEDIUM)
---
Cross-Validation Results (Random Forest, 1,000-record dataset)
Metric	CV Mean	CV Std Dev
Accuracy	92.40%	±2.10%
Precision (Macro)	91.80%	±2.40%
Recall (Macro)	92.10%	±2.30%
F1-Score (Macro)	91.95%	±2.20%
---
Comparison: OpenVAS / Nessus / Proposed Pipeline
Metric	OpenVAS CE 22.4	Nessus Essentials 10.x	Proposed Pipeline v4.1
Severity accuracy (6-service testbed)	66.7% (4/6)	66.7% (4/6)	100% (6/6)
Mean time to report	8–12 min	5–8 min	<30 seconds
Anomaly detection	✗	✗	✓ (Isolation Forest)
Transparent scoring	Partial	✗	✓ (documented formulas)
Setup time	30–60 min	15–30 min	2–5 min
Commercial licence	Free	Required	Free (MIT)
Offline operation	After feed sync	Limited	✓ (pre-trained models)
Probabilistic confidence	✗	✗	✓ (per-service)
Both OpenVAS and Nessus misclassify Telnet as High (not Critical). The proposed pipeline correctly identifies it as Critical+ANOMALY based on the combined risk profile PRS=0.95 + VES=1.00 + PEF=1, without requiring a specific CVE plugin match.
---
Repository Structure
```
vapt_repo/
├── ml_analysis.py          # Main pipeline (v4.1)
├── requirements.txt        # Python dependencies
├── test_ml_analysis.py     # Unit test suite (pytest)
├── sample_scans/
│   ├── scan_local_testbed.xml   # Local testbed (6 services)
│   ├── scanme_results.xml       # scanme.nmap.org (4 ports)
│   ├── pentest_results.xml      # pentest-ground.com (8 services)
│   └── juiceshop_results.xml    # OWASP Juice Shop (1 service)
└── README.md
```
---
NVD API Integration
The tool queries the NIST NVD REST API v2.0 to build a live CVE corpus for all detected ports:
Endpoint: `https://services.nvd.nist.gov/rest/json/cves/2.0`
Default rate: 5 req/30s (no registration needed)
With API key: 50 req/30s — register free
Cache TTL: 7 days (`nvd_ml_cache.json`)
Offline fallback: Pre-trained synthetic Ridge model (air-gapped environments)
```bash
# Using NVD API key (faster)
python3 ml_analysis.py --xml scan.xml --nvd-key YOUR_KEY_HERE

# Refresh cached CVE data
python3 ml_analysis.py --xml scan.xml --refresh-cache

# Fully offline (no internet)
python3 ml_analysis.py --xml scan.xml --no-nvd
```
---
Running Unit Tests
```bash
pip install pytest
python3 -m pytest test_ml_analysis.py -v
```
Tests cover: PRS Ridge model, PEF LogReg, PSC NaiveBayes, VKF SVM, VES entropy formula, Isolation Forest anomaly detection, risk score formulas (GWVS/NEF/ARS), XML parsing for all four sample scans, and training dataset generation.
---
NIST SP 800-53 Remediation
Remediation recommendations are derived by TF-IDF cosine similarity between CVE description corpora (fetched from the NVD API) and all 17 NIST SP 800-53 control family descriptions. In offline mode, a pre-trained Random Forest maps port features to NIST control families.
---
Authors
Naveen Kumar Bandla — Email Security Specialist, Centific, Hyderabad  
naveen.bandla@centific.com
Dr. Y. Nasir Ahmed — Department of Computer Science, Chaitanya Deemed To Be University, Hyderabad
---
Citation
```bibtex
@article{bandla2025vapt,
  title   = {AI/ML-Based Intelligent Vulnerability Assessment and Penetration Testing
             Using Nmap in Kali Linux: A Five-Model Pipeline with NVD Integration,
             Real-World Validation, and Comparative Evaluation Against OpenVAS and Nessus},
  author  = {Bandla, Naveen Kumar and Ahmed, Y. Nasir},
  journal = {Cyber Security and Applications},
  year    = {2025},
  publisher = {KeAi / Elsevier},
  url     = {https://github.com/loyolite192652/vapt_repo}
}
```
---
License
MIT — see LICENSE for details.  
All scanning was performed within authorised environments. No unauthorised systems were scanned.
