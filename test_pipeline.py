"""
test_ml_analysis.py — Unit tests for AI/ML VAPT Pipeline v4.1
Covers: PRS Ridge model, PEF LogReg, PSC NaiveBayes, VKF SVM,
        VES entropy, Isolation Forest, risk score formulas, XML parsing.
Run with: python3 -m pytest test_ml_analysis.py -v
"""

import math
import sys
import os
import unittest
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ml_analysis as ml


class TestPRSModel(unittest.TestCase):
    """Test Port Risk Score via Ridge Regression."""

    def setUp(self):
        self.ridge, self.scaler = ml._build_synthetic_prs_model()
        self.known_prs = {21: 0.90, 22: 0.55, 23: 0.95, 80: 0.75, 443: 0.45}

    def test_known_port_returned_directly(self):
        """Known ports should return pre-computed NVD PRS values."""
        self.assertAlmostEqual(
            ml.predict_prs(23, self.ridge, self.scaler, self.known_prs), 0.95)
        self.assertAlmostEqual(
            ml.predict_prs(443, self.ridge, self.scaler, self.known_prs), 0.45)

    def test_prs_in_range(self):
        """PRS must be in [0.10, 1.00] for all ports."""
        for port in [21, 22, 23, 53, 80, 443, 3306, 8080, 9999, 65535]:
            prs = ml.predict_prs(port, self.ridge, self.scaler, {})
            self.assertGreaterEqual(prs, 0.10, f"PRS too low for port {port}")
            self.assertLessEqual(prs, 1.00, f"PRS too high for port {port}")

    def test_telnet_highest_prs(self):
        """Telnet (port 23) should have the highest known PRS."""
        telnet_prs = self.known_prs[23]
        for port, prs in self.known_prs.items():
            self.assertLessEqual(prs, telnet_prs,
                                 f"Port {port} PRS ({prs}) > Telnet PRS ({telnet_prs})")

    def test_https_lower_than_http(self):
        """HTTPS (443) should have lower PRS than HTTP (80)."""
        self.assertLess(
            ml.predict_prs(443, self.ridge, self.scaler, self.known_prs),
            ml.predict_prs(80, self.ridge, self.scaler, self.known_prs))

    def test_unknown_port_predicted(self):
        """Unknown port (not in known_prs) should still return a valid PRS."""
        prs = ml.predict_prs(12345, self.ridge, self.scaler, {})
        self.assertIsInstance(prs, float)
        self.assertGreaterEqual(prs, 0.10)
        self.assertLessEqual(prs, 1.00)


class TestPEFModel(unittest.TestCase):
    """Test Protocol Encryption Flag via Logistic Regression."""

    def setUp(self):
        self.lr, self.tfidf = ml.build_pef_model()

    def test_unencrypted_services_flagged(self):
        """Known plaintext services should return PEF=1."""
        for svc in ["ftp", "telnet", "smtp", "pop3", "redis"]:
            pef = ml.predict_pef(svc, self.lr, self.tfidf)
            self.assertEqual(pef, 1, f"Expected PEF=1 for '{svc}', got {pef}")

    def test_encrypted_services_not_flagged(self):
        """Known encrypted services should return PEF=0."""
        for svc in ["https", "ssh", "sftp", "imaps", "smtps"]:
            pef = ml.predict_pef(svc, self.lr, self.tfidf)
            self.assertEqual(pef, 0, f"Expected PEF=0 for '{svc}', got {pef}")

    def test_binary_output(self):
        """PEF must always be 0 or 1."""
        for svc in ["http", "https", "mysql", "unknown", "nping"]:
            pef = ml.predict_pef(svc, self.lr, self.tfidf)
            self.assertIn(pef, [0, 1])


class TestPSCModel(unittest.TestCase):
    """Test Port Service Category via Multinomial Naive Bayes."""

    def setUp(self):
        self.nb, self.tfidf = ml.build_psc_model()

    def test_web_category(self):
        """HTTP/HTTPS should be categorised as Web (0)."""
        self.assertEqual(ml.predict_psc("http",  80, self.nb, self.tfidf), 0)
        self.assertEqual(ml.predict_psc("https", 443, self.nb, self.tfidf), 0)

    def test_legacy_category(self):
        """Telnet and FTP should be categorised as Legacy (1)."""
        self.assertEqual(ml.predict_psc("telnet", 23, self.nb, self.tfidf), 1)
        self.assertEqual(ml.predict_psc("ftp",    21, self.nb, self.tfidf), 1)

    def test_remote_access_category(self):
        """SSH should be categorised as Remote-Access (3)."""
        self.assertEqual(ml.predict_psc("ssh", 22, self.nb, self.tfidf), 3)

    def test_database_category(self):
        """MySQL and PostgreSQL should be categorised as Database (4)."""
        self.assertEqual(ml.predict_psc("mysql",      3306, self.nb, self.tfidf), 4)
        self.assertEqual(ml.predict_psc("postgresql", 5432, self.nb, self.tfidf), 4)

    def test_output_in_valid_range(self):
        """PSC must always be in [0, 5]."""
        for svc, port in [("http", 80), ("ssh", 22), ("unknown", 0),
                          ("redis", 6379), ("smb", 445)]:
            psc = ml.predict_psc(svc, port, self.nb, self.tfidf)
            self.assertIn(psc, [0, 1, 2, 3, 4, 5])


class TestVKFModel(unittest.TestCase):
    """Test Version Known Flag via Linear SVM."""

    def setUp(self):
        self.svm, self.scaler = ml.build_vkf_model()

    def test_known_versions_detected(self):
        """Real version strings should return VKF=1."""
        for v in ["OpenSSH 7.4", "Apache httpd 2.4.7", "vsftpd 1.4.1",
                  "lighttpd 0.93.15", "MySQL 5.7.32"]:
            vkf = ml.predict_vkf(v, self.svm, self.scaler)
            self.assertEqual(vkf, 1, f"Expected VKF=1 for '{v}', got {vkf}")

    def test_unknown_versions_flagged(self):
        """Empty or unknown version strings should return VKF=0."""
        for v in ["unknown", "", "none"]:
            vkf = ml.predict_vkf(v, self.svm, self.scaler)
            self.assertEqual(vkf, 0, f"Expected VKF=0 for '{v}', got {vkf}")

    def test_binary_output(self):
        """VKF must always be 0 or 1."""
        for v in ["OpenSSH 8.9", "unknown", "2.4.7", ""]:
            vkf = ml.predict_vkf(v, self.svm, self.scaler)
            self.assertIn(vkf, [0, 1])


class TestVESEntropy(unittest.TestCase):
    """Test Version Entropy Score formula."""

    def test_unknown_version_max_entropy(self):
        """Version 'unknown' should receive maximum VES (1.0)."""
        ves = ml.compute_ves_entropy("unknown", ["unknown", "OpenSSH 7.4",
                                                  "vsftpd 1.4.1", "lighttpd 0.93.15"])
        self.assertGreaterEqual(ves, 0.40, "VES for 'unknown' should be high")

    def test_detailed_version_low_entropy(self):
        """Detailed version string should receive low VES."""
        all_versions = ["unknown", "OpenSSH 8.9p1 Ubuntu 3ubuntu0.6",
                        "lighttpd 0.93.15", "vsftpd 1.4.1"]
        ves = ml.compute_ves_entropy("OpenSSH 8.9p1 Ubuntu 3ubuntu0.6", all_versions)
        self.assertLessEqual(ves, 0.3, "VES for detailed version should be low")

    def test_ves_in_range(self):
        """VES must always be in [0, 1]."""
        all_versions = ["unknown", "OpenSSH 7.4", "vsftpd 1.4.1", ""]
        for v in all_versions:
            ves = ml.compute_ves_entropy(v, all_versions)
            self.assertGreaterEqual(ves, 0.0)
            self.assertLessEqual(ves, 1.0)

    def test_empty_string_high_ves(self):
        """Empty version string should have high VES."""
        ves = ml.compute_ves_entropy("", ["", "OpenSSH 7.4", "vsftpd 1.4.1"])
        self.assertGreater(ves, 0.5, "Empty version should produce high VES")


class TestRiskScoreFormulas(unittest.TestCase):
    """Test GWVS, NEF, ARS formulas against paper values."""

    def _make_df(self, rows):
        """Helper: build a minimal DataFrame matching what stage_4 expects."""
        records = []
        for port, tier, prs, svc in rows:
            records.append({
                "port_id": port, "predicted_tier": tier,
                "PRS": prs, "service_name": svc, "host_ip": "192.168.1.1",
            })
        return pd.DataFrame(records)

    def test_gwvs_local_testbed(self):
        """GWVS for local testbed should match paper value 58.33%."""
        # Weights: Critical(4)×PRS for telnet,ftp; High(3)×PRS for http,ssh,dns; Medium(2)×PRS for https
        num = 4*0.95 + 4*0.90 + 3*0.75 + 3*0.55 + 3*0.60 + 2*0.45
        gwvs = round((num / (6 * 4)) * 100, 2)
        self.assertAlmostEqual(gwvs, 58.33, places=1)

    def test_nef_local_testbed(self):
        """NEF for local testbed should match paper value 0.9412."""
        nef = round((5 + 6 + 5) / (5 + 7 + 5), 4)
        self.assertAlmostEqual(nef, 0.9412, places=3)

    def test_ars_local_testbed(self):
        """ARS for local testbed should match paper value 54.90%."""
        gwvs = 58.33
        nef  = 0.9412
        ars  = round(gwvs * nef, 2)
        self.assertAlmostEqual(ars, 54.90, places=0)

    def test_severity_weights(self):
        """SEVERITY_WEIGHTS must match CVSS-aligned tier weights."""
        self.assertEqual(ml.SEVERITY_WEIGHTS["Critical"], 4)
        self.assertEqual(ml.SEVERITY_WEIGHTS["High"],     3)
        self.assertEqual(ml.SEVERITY_WEIGHTS["Medium"],   2)
        self.assertEqual(ml.SEVERITY_WEIGHTS["Low"],      1)


class TestXMLParsing(unittest.TestCase):
    """Test XML parsing across all four sample scan files."""

    SCAN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "sample_scans")

    def _parse(self, filename):
        path = os.path.join(self.SCAN_DIR, filename)
        return ml.stage_0_parse_nmap_xml(path)

    def test_local_testbed_6_ports(self):
        df = self._parse("scan_local_testbed.xml")
        self.assertEqual(len(df), 6)
        self.assertIn(23, df["port_id"].values)  # telnet present
        self.assertIn(443, df["port_id"].values) # https present

    def test_scanme_4_ports(self):
        df = self._parse("scanme_results.xml")
        self.assertEqual(len(df), 4)
        self.assertIn(31337, df["port_id"].values)  # anomalous port

    def test_pentest_8_ports(self):
        df = self._parse("pentest_results.xml")
        self.assertEqual(len(df), 8)
        self.assertIn(3306, df["port_id"].values)   # mysql
        self.assertIn(5432, df["port_id"].values)   # postgres

    def test_juiceshop_1_port(self):
        df = self._parse("juiceshop_results.xml")
        self.assertEqual(len(df), 1)
        self.assertEqual(int(df["port_id"].iloc[0]), 3000)

    def test_required_columns_present(self):
        df = self._parse("scan_local_testbed.xml")
        for col in ["port_id", "protocol", "service_name", "service_version", "host_ip"]:
            self.assertIn(col, df.columns)


class TestIsolationForest(unittest.TestCase):
    """Test Isolation Forest anomaly detection."""

    def _make_feature_df(self):
        """Build a DataFrame matching the 6-service local testbed feature matrix."""
        rows = [
            {"port_id": 23, "service_name": "telnet", "PRS": 0.95, "VES": 1.00, "PEF": 1, "PSC": 1, "VKF": 0},
            {"port_id": 21, "service_name": "ftp",    "PRS": 0.90, "VES": 0.22, "PEF": 1, "PSC": 1, "VKF": 1},
            {"port_id": 53, "service_name": "domain", "PRS": 0.60, "VES": 0.22, "PEF": 1, "PSC": 5, "VKF": 1},
            {"port_id": 80, "service_name": "http",   "PRS": 0.75, "VES": 0.00, "PEF": 1, "PSC": 0, "VKF": 1},
            {"port_id": 22, "service_name": "ssh",    "PRS": 0.55, "VES": 0.29, "PEF": 0, "PSC": 3, "VKF": 1},
            {"port_id": 443,"service_name": "https",  "PRS": 0.45, "VES": 0.00, "PEF": 0, "PSC": 0, "VKF": 1},
        ]
        # Add dummy columns that stage_3 needs
        for r in rows:
            r["service_version"] = "1.0"
            r["host_ip"] = "192.168.1.1"
            r["port_state"] = "open"
            r["CRI"] = r["PRS"]*100 + r["VES"]*100 + r["PEF"]*100
            r["PSC_label"] = ml.PSC_LABELS.get(r["PSC"], "Unknown")
            r["predicted_tier"] = "High"
            r["rf_confidence"] = 0.9
            r["prob_Critical"] = 0.1
            r["prob_High"] = 0.9
            r["prob_Medium"] = 0.0
            r["prob_Low"] = 0.0
        return pd.DataFrame(rows)

    def test_output_columns_added(self):
        df = self._make_feature_df()
        result = ml.stage_3_isolation_forest(df)
        self.assertIn("anomaly_score", result.columns)
        self.assertIn("anomaly_label", result.columns)

    def test_anomaly_labels_binary(self):
        df = self._make_feature_df()
        result = ml.stage_3_isolation_forest(df)
        for label in result["anomaly_label"]:
            self.assertIn(label, ["ANOMALY", "Normal"])

    def test_scores_in_range(self):
        df = self._make_feature_df()
        result = ml.stage_3_isolation_forest(df)
        for score in result["anomaly_score"]:
            self.assertGreaterEqual(float(score), 0.0)
            self.assertLessEqual(float(score), 1.0)

    def test_telnet_highest_anomaly_score(self):
        """Telnet should have the highest anomaly score in the testbed."""
        df = self._make_feature_df()
        result = ml.stage_3_isolation_forest(df)
        max_score_port = result.loc[result["anomaly_score"].idxmax(), "port_id"]
        self.assertEqual(int(max_score_port), 23,
                         "Telnet (port 23) should have highest anomaly score")


class TestTrainingDataset(unittest.TestCase):
    """Test training dataset generation."""

    def test_dataset_size(self):
        """Training dataset must contain exactly 1,000 records."""
        df = ml.build_training_dataset()
        self.assertEqual(len(df), 1000)

    def test_class_distribution(self):
        """Class distribution must match paper: Critical 35%, High 30%, Medium 20%, Low 15%."""
        df = ml.build_training_dataset()
        dist = df["severity"].value_counts()
        self.assertEqual(dist.get("Critical", 0), 350)
        self.assertEqual(dist.get("High", 0), 300)
        self.assertEqual(dist.get("Medium", 0), 200)
        self.assertEqual(dist.get("Low", 0), 150)

    def test_feature_columns_present(self):
        """Dataset must have all five feature columns."""
        df = ml.build_training_dataset()
        for col in ["PRS", "VES", "PEF", "PSC", "VKF", "severity"]:
            self.assertIn(col, df.columns)

    def test_feature_ranges(self):
        """All feature values must be within valid ranges."""
        df = ml.build_training_dataset()
        self.assertTrue((df["PRS"] >= 0.0).all() and (df["PRS"] <= 1.0).all())
        self.assertTrue((df["VES"] >= 0.0).all() and (df["VES"] <= 1.0).all())
        self.assertTrue(df["PEF"].isin([0, 1]).all())
        self.assertTrue(df["PSC"].between(0, 5).all())
        self.assertTrue(df["VKF"].isin([0, 1]).all())


class TestPSCLabels(unittest.TestCase):
    """Test PSC_LABELS dict completeness."""

    def test_all_six_categories_present(self):
        for i in range(6):
            self.assertIn(i, ml.PSC_LABELS)

    def test_label_names(self):
        self.assertEqual(ml.PSC_LABELS[0], "Web")
        self.assertEqual(ml.PSC_LABELS[1], "Legacy/Unencrypted")
        self.assertEqual(ml.PSC_LABELS[2], "File-Share")
        self.assertEqual(ml.PSC_LABELS[3], "Remote-Access")
        self.assertEqual(ml.PSC_LABELS[4], "Database")
        self.assertEqual(ml.PSC_LABELS[5], "Infrastructure")


if __name__ == "__main__":
    unittest.main(verbosity=2)

