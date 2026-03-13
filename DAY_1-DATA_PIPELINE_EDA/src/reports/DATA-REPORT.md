# EDA Report - NF-UQ-NIDS-v2 Dataset
## Network Intrusion Detection System Analysis

**Generated:** 2026-02-18 11:09:32
**Data Source:** sample

---

## Dataset Overview

| Metric | Value |
|--------|-------|
| Total Rows | 100,000 |
| Total Columns | 46 |
| Numerical Columns | 42 |
| Categorical Columns | 4 |
| Memory Usage | 55.14 MB |

---

## Target Variable

- **Target Column:** Attack
- **Number of Classes:** 20
- **Imbalance Ratio:** 32986.00:1

### Class Distribution:
Attack
Benign            32986
DDoS              28742
DoS               23542
scanning           4955
Reconnaissance     3444
xss                3217
password           1519
injection           930
Bot                 197
Brute Force         190
Infilteration       161
Exploits             34
Fuzzers              25
Backdoor             23
Generic              19
mitm                  8
ransomware            3
Shellcode             2
Analysis              2
Theft                 1

---

## Data Quality

| Metric | Value |
|--------|-------|
| Missing Values | 0 |
| Missing Percentage | 0.0000% |
| Duplicate Rows | 0 |
| Duplicate Percentage | 0.00% |

---

## Highly Correlated Features

                 Feature 1                    Feature 2  Correlation
2                OUT_BYTES  NUM_PKTS_1024_TO_1514_BYTES        1.000
8                ICMP_TYPE               ICMP_IPV4_TYPE        1.000
6         LONGEST_FLOW_PKT               MAX_IP_PKT_LEN        1.000
5                  MIN_TTL                      MAX_TTL        0.999
1                OUT_BYTES                     OUT_PKTS        0.999
3                 OUT_PKTS  NUM_PKTS_1024_TO_1514_BYTES        0.999
4                TCP_FLAGS             CLIENT_TCP_FLAGS        0.995
7  RETRANSMITTED_OUT_BYTES       RETRANSMITTED_OUT_PKTS        0.991
0                 IN_BYTES                      IN_PKTS        0.965

Total pairs: 9

---

## Outlier Summary (Top 10)

                    Feature  IQR Outliers  IQR Outliers %  Z-Score Outliers  Z-Score Outliers %
34          TCP_WIN_MAX_OUT         24901           24.90              6343                6.34
9          CLIENT_TCP_FLAGS         24203           24.20              7738                7.74
20  SRC_TO_DST_SECOND_BYTES         22647           22.65                 0                0.00
7                  OUT_PKTS         21796           21.80                29                0.03
17        SHORTEST_FLOW_PKT         19724           19.72               125                0.12
1               L4_DST_PORT         19568           19.57              3981                3.98
21  DST_TO_SRC_SECOND_BYTES         18809           18.81                 0                0.00
6                 OUT_BYTES         18191           18.19                14                0.01
16         LONGEST_FLOW_PKT         17351           17.35                 2                0.00
19           MAX_IP_PKT_LEN         17351           17.35                 2                0.00

---

## Recommendations

1. Class Imbalance: Handle using SMOTE or class weights if ratio > 10
2. Missing Values: No action needed
3. Duplicates: No action needed
4. High Correlations: Acceptable

---

## Generated Visualizations

- target_distribution.png
- missing_values.png
- feature_distributions.png
- boxplots_outliers.png
- correlation_matrix_full.png
- correlation_matrix_top15.png
- feature_by_target.png
- outlier_percentage.png

---

## Day 1 EDA Complete!
