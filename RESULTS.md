# Results

- Dataset file: `WSN-DS.csv`
- Rows: 374661 (after removing duplicates/missing: 365788)
- Features: 17 (Time, Is_CH, who CH, Dist_To_CH, ADV_S, ADV_R, JOIN_S, JOIN_R, SCH_S, SCH_R, Rank, DATA_S, DATA_R, Data_Sent_To_BS, dist_CH_To_BS, send_code, Expaned Energy)
- Label column: `Attack type`
- Class counts:
  - Normal: 332040
  - Grayhole: 13909
  - Blackhole: 10049
  - TDMA: 6633
  - Flooding: 3157
- Split: 80% train (292630) / 20% test (73158), stratified, random_state=42
- Best model (weighted F1): **GradientBoosting**

## Metrics on the test set (%)

| model | accuracy | precision_weighted | recall_weighted | f1_weighted | attack_precision | attack_recall | attack_f1 |
|---|---|---|---|---|---|---|---|
| AdaBoost | 97.55 | 97.70 | 97.55 | 97.60 | 91.12 | 93.35 | 92.22 |
| GradientBoosting | 99.54 | 99.55 | 99.54 | 99.54 | 98.40 | 98.09 | 98.24 |
| LDA | 95.92 | 96.77 | 95.92 | 96.11 | 84.01 | 96.00 | 89.61 |
