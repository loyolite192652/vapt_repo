# ==============================================================================
# FILE  : tests/test_pipeline.py
# USAGE : python3 -m pytest tests/test_pipeline.py -v
#         python3 tests/test_pipeline.py      (run without pytest)
# ==============================================================================

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from ml_analysis import (
    compute_prs, compute_ves, compute_pef, compute_psc, compute_vkf,
    build_nvd_training_dataset, stage_1_feature_engineering,
    PORT_RISK_SCORES, DEFAULT_PRS, IF_ANOMALY_THRESHOLD
)


# ==============================================================================
# Tests: Feature Engineering Functions
# ==============================================================================

class TestComputePRS:
    def test_known_port_telnet(self):
        assert compute_prs(23) == 0.95, "Telnet should have PRS = 0.95"

    def test_known_port_ftp(self):
        assert compute_prs(21) == 0.90, "FTP should have PRS = 0.90"

    def test_known_port_https(self):
        assert compute_prs(443) == 0.45, "HTTPS should have PRS = 0.45"

    def test_unknown_port_default(self):
        assert compute_prs(9999) == DEFAULT_PRS, \
            f"Unknown port should return DEFAULT_PRS = {DEFAULT_PRS}"

    def test_all_prs_in_range(self):
        for port, prs in PORT_RISK_SCORES.items():
            assert 0.0 <= prs <= 1.0, f"PRS for port {port} out of [0,1]: {prs}"


class TestComputeVES:
    def test_unknown_version_max_entropy(self):
        assert compute_ves("unknown", 16) == 1.0

    def test_empty_version_max_entropy(self):
        assert compute_ves("", 16) == 1.0

    def test_none_string_max_entropy(self):
        assert compute_ves("none", 16) == 1.0

    def test_known_version_lower_entropy(self):
        ves = compute_ves("lighttpd 0.93.15", 16)
        assert ves < 1.0, "Known version should have VES < 1.0"

    def test_ves_in_range(self):
        for v in ["unknown", "OpenSSH 7.4", "vsftpd 1.4.1", "", "1.0"]:
            result = compute_ves(v, 16)
            assert 0.0 <= result <= 1.0, f"VES out of [0,1] for version '{v}': {result}"

    def test_longer_version_lower_ves(self):
        short_ves = compute_ves("1.0", 16)
        long_ves  = compute_ves("lighttpd 0.93.15", 16)
        assert long_ves <= short_ves, \
            "Longer version string should have lower or equal VES"


class TestComputePEF:
    def test_ftp_unencrypted(self):
        assert compute_pef(21, "ftp") == 1

    def test_telnet_unencrypted(self):
        assert compute_pef(23, "telnet") == 1

    def test_http_unencrypted(self):
        assert compute_pef(80, "http") == 1

    def test_https_encrypted(self):
        assert compute_pef(443, "https") == 0

    def test_ssh_encrypted(self):
        assert compute_pef(22, "ssh") == 0

    def test_pef_binary(self):
        for port, svc in [(21,"ftp"),(22,"ssh"),(23,"telnet"),(80,"http"),(443,"https")]:
            result = compute_pef(port, svc)
            assert result in (0, 1), f"PEF must be binary, got {result}"


class TestComputePSC:
    def test_http_is_web(self):
        assert compute_psc("http") == 0

    def test_https_is_web(self):
        assert compute_psc("https") == 0

    def test_ftp_is_legacy(self):
        assert compute_psc("ftp") == 1

    def test_telnet_is_legacy(self):
        assert compute_psc("telnet") == 1

    def test_ssh_is_remote(self):
        assert compute_psc("ssh") == 3

    def test_mysql_is_database(self):
        assert compute_psc("mysql") == 4

    def test_unknown_is_other(self):
        assert compute_psc("unknown") == 5

    def test_psc_in_range(self):
        for svc in ["http","https","ftp","telnet","ssh","mysql","unknown","domain"]:
            result = compute_psc(svc)
            assert 0 <= result <= 5, f"PSC out of range for service '{svc}': {result}"


class TestComputeVKF:
    def test_unknown_version_returns_zero(self):
        assert compute_vkf("unknown") == 0

    def test_empty_version_returns_zero(self):
        assert compute_vkf("") == 0

    def test_known_version_returns_one(self):
        assert compute_vkf("OpenSSH 7.4") == 1

    def test_vkf_binary(self):
        for v in ["unknown", "", "1.0", "vsftpd 1.4.1", "none"]:
            result = compute_vkf(v)
            assert result in (0, 1), f"VKF must be binary, got {result}"


# ==============================================================================
# Tests: NVD Training Dataset
# ==============================================================================

class TestNVDDataset:
    def setup_method(self):
        self.df = build_nvd_training_dataset()

    def test_dataset_size(self):
        assert len(self.df) == 200, \
            f"Expected 200 training records, got {len(self.df)}"

    def test_severity_distribution(self):
        counts = self.df["severity"].value_counts()
        assert counts["Critical"] == 70
        assert counts["High"]     == 60
        assert counts["Medium"]   == 40
        assert counts["Low"]      == 30

    def test_prs_in_range(self):
        assert (self.df["PRS"] >= 0).all() and (self.df["PRS"] <= 1).all()

    def test_ves_in_range(self):
        assert (self.df["VES"] >= 0).all() and (self.df["VES"] <= 1).all()

    def test_pef_binary(self):
        assert self.df["PEF"].isin([0, 1]).all()

    def test_vkf_binary(self):
        assert self.df["VKF"].isin([0, 1]).all()

    def test_psc_in_range(self):
        assert self.df["PSC"].between(0, 5).all()

    def test_required_columns(self):
        required = {"PRS", "VES", "PEF", "PSC", "VKF", "severity"}
        assert required.issubset(set(self.df.columns))

    def test_reproducibility(self):
        df2 = build_nvd_training_dataset()
        pd.testing.assert_frame_equal(self.df, df2)


# ==============================================================================
# Tests: Feature Engineering Stage (Stage 1)
# ==============================================================================

class TestStage1:
    def setup_method(self):
        self.sample_df = pd.DataFrame([
            {"port_id": 21,  "protocol": "tcp", "service_name": "ftp",
             "service_version": "vsftpd 1.4.1", "host_ip": "192.168.1.1",
             "port_state": "open"},
            {"port_id": 22,  "protocol": "tcp", "service_name": "ssh",
             "service_version": "OpenSSH 7.4",  "host_ip": "192.168.1.1",
             "port_state": "open"},
            {"port_id": 23,  "protocol": "tcp", "service_name": "telnet",
             "service_version": "unknown",        "host_ip": "192.168.1.1",
             "port_state": "open"},
            {"port_id": 80,  "protocol": "tcp", "service_name": "http",
             "service_version": "lighttpd 0.93.15","host_ip": "192.168.1.1",
             "port_state": "open"},
            {"port_id": 443, "protocol": "tcp", "service_name": "https",
             "service_version": "lighttpd 0.93.15","host_ip": "192.168.1.1",
             "port_state": "open"},
        ])

    def test_output_has_all_features(self):
        result = stage_1_feature_engineering(self.sample_df)
        for col in ["PRS", "VES", "PEF", "PSC", "VKF", "CRI"]:
            assert col in result.columns, f"Missing feature column: {col}"

    def test_telnet_highest_prs(self):
        result = stage_1_feature_engineering(self.sample_df)
        telnet_prs = result[result["port_id"] == 23]["PRS"].values[0]
        assert telnet_prs == 0.95

    def test_telnet_max_ves(self):
        result = stage_1_feature_engineering(self.sample_df)
        telnet_ves = result[result["port_id"] == 23]["VES"].values[0]
        assert telnet_ves == 1.0, f"Telnet VES should be 1.0, got {telnet_ves}"

    def test_ftp_unencrypted(self):
        result = stage_1_feature_engineering(self.sample_df)
        ftp_pef = result[result["port_id"] == 21]["PEF"].values[0]
        assert ftp_pef == 1

    def test_https_encrypted(self):
        result = stage_1_feature_engineering(self.sample_df)
        https_pef = result[result["port_id"] == 443]["PEF"].values[0]
        assert https_pef == 0

    def test_cri_equals_prs_plus_ves_plus_pef(self):
        result = stage_1_feature_engineering(self.sample_df)
        for _, row in result.iterrows():
            expected_cri = round(row["PRS"] + row["VES"] + row["PEF"], 6)
            actual_cri   = round(row["CRI"], 6)
            assert abs(expected_cri - actual_cri) < 1e-5, \
                f"CRI mismatch for port {row['port_id']}: {expected_cri} vs {actual_cri}"

    def test_row_count_preserved(self):
        result = stage_1_feature_engineering(self.sample_df)
        assert len(result) == len(self.sample_df)


# ==============================================================================
# Tests: Anomaly Threshold Constant
# ==============================================================================

class TestConstants:
    def test_anomaly_threshold_range(self):
        assert 0.0 < IF_ANOMALY_THRESHOLD < 1.0, \
            f"Anomaly threshold must be in (0,1), got {IF_ANOMALY_THRESHOLD}"

    def test_contamination_range(self):
        from ml_analysis import IF_CONTAMINATION
        assert 0.01 <= IF_CONTAMINATION <= 0.49, \
            f"Contamination must be in [0.01, 0.49], got {IF_CONTAMINATION}"


# ==============================================================================
# Simple runner (no pytest required)
# ==============================================================================

def run_all_tests():
    test_classes = [
        TestComputePRS,
        TestComputeVES,
        TestComputePEF,
        TestComputePSC,
        TestComputeVKF,
        TestNVDDataset,
        TestStage1,
        TestConstants,
    ]

    total   = 0
    passed  = 0
    failed  = 0
    errors  = []

    for cls in test_classes:
        instance = cls()
        methods  = [m for m in dir(cls) if m.startswith("test_")]
        for method_name in methods:
            total += 1
            try:
                if hasattr(instance, "setup_method"):
                    instance.setup_method()
                getattr(instance, method_name)()
                print(f"  ✓  {cls.__name__}.{method_name}")
                passed += 1
            except AssertionError as e:
                print(f"  ✗  {cls.__name__}.{method_name}  →  {e}")
                failed += 1
                errors.append((f"{cls.__name__}.{method_name}", str(e)))
            except Exception as e:
                print(f"  !  {cls.__name__}.{method_name}  →  ERROR: {e}")
                failed += 1
                errors.append((f"{cls.__name__}.{method_name}", f"ERROR: {e}"))

    print()
    print("=" * 60)
    print(f"  RESULTS: {passed}/{total} tests passed")
    if errors:
        print(f"  FAILURES:")
        for name, msg in errors:
            print(f"    - {name}: {msg}")
    else:
        print("  All tests PASSED.")
    print("=" * 60)
    return failed == 0


if __name__ == "__main__":
    print("\nRunning VAPT Pipeline Unit Tests")
    print("=" * 60)
    success = run_all_tests()
    sys.exit(0 if success else 1)
