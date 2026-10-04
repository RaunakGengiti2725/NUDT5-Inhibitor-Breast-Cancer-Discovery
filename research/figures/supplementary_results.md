# Automatically generated supplementary results

These are exploratory diagnostics. AP and trapezoidal PR-AUC are distinct. Threshold 0.5 is arbitrary. No measured activity probabilities or prospective error rates are claimed.

| Design | Method | n | ROC-AUC | AP | Brier | MCC |
|---|---|---:|---:|---:|---:|---:|
| full_valid_set | Constant_0_5 | 45 | 0.5000 | 0.4222 | 0.2500 | 0.0000 |
| full_valid_set | Equal_mean | 45 | 0.9312 | 0.9512 | 0.0541 | 0.9115 |
| full_valid_set | Fingerprint_LR | 45 | 0.9170 | 0.9451 | 0.0460 | 0.9115 |
| full_valid_set | GBT | 45 | 0.8431 | 0.7749 | 0.0889 | 0.8178 |
| full_valid_set | Mean_without_GBT | 45 | 0.9312 | 0.9512 | 0.0515 | 0.9115 |
| full_valid_set | Mean_without_Nearest_active | 45 | 0.9474 | 0.9589 | 0.0542 | 0.8633 |
| full_valid_set | Mean_without_RF | 45 | 0.9231 | 0.9473 | 0.0562 | 0.8633 |
| full_valid_set | Mean_without_SVM_RBF | 45 | 0.9271 | 0.9496 | 0.0585 | 0.8633 |
| full_valid_set | Nearest_active | 45 | 0.9130 | 0.9436 | 0.0678 | 0.9115 |
| full_valid_set | Property_LR | 45 | 0.9798 | 0.9776 | 0.0436 | 0.9115 |
| full_valid_set | RF | 45 | 0.9413 | 0.9548 | 0.0513 | 0.9115 |
| full_valid_set | SVM_RBF | 45 | 0.9595 | 0.9639 | 0.0452 | 0.9115 |
| full_valid_set | Tanimoto_kNN | 45 | 0.9332 | 0.9392 | 0.0658 | 0.8633 |
| full_valid_set | Train_prevalence | 45 | 0.2460 | 0.3390 | 0.2930 | 0.0000 |
| nearest_property_subset | Constant_0_5 | 38 | 0.5000 | 0.5000 | 0.2500 | 0.0000 |
| nearest_property_subset | Equal_mean | 38 | 0.9280 | 0.9566 | 0.0634 | 0.8433 |
| nearest_property_subset | Fingerprint_LR | 38 | 0.9197 | 0.9541 | 0.0569 | 0.8997 |
| nearest_property_subset | GBT | 38 | 0.8490 | 0.8179 | 0.1053 | 0.7895 |
| nearest_property_subset | Mean_without_GBT | 38 | 0.9280 | 0.9566 | 0.0597 | 0.8997 |
| nearest_property_subset | Mean_without_Nearest_active | 38 | 0.9252 | 0.9556 | 0.0646 | 0.8433 |
| nearest_property_subset | Mean_without_RF | 38 | 0.9363 | 0.9606 | 0.0656 | 0.8433 |
| nearest_property_subset | Mean_without_SVM_RBF | 38 | 0.9169 | 0.9529 | 0.0681 | 0.8433 |
| nearest_property_subset | Nearest_active | 38 | 0.9280 | 0.9582 | 0.0752 | 0.8997 |
| nearest_property_subset | Property_LR | 38 | 0.9945 | 0.9946 | 0.0425 | 0.8997 |
| nearest_property_subset | RF | 38 | 0.9183 | 0.9520 | 0.0600 | 0.8997 |
| nearest_property_subset | SVM_RBF | 38 | 0.9557 | 0.9683 | 0.0539 | 0.8997 |
| nearest_property_subset | Tanimoto_kNN | 38 | 0.9266 | 0.9092 | 0.0947 | 0.7895 |
| nearest_property_subset | Train_prevalence | 38 | 0.2548 | 0.4026 | 0.2840 | -0.3684 |
| similarity_component_split | Constant_0_5 | 45 | 0.5000 | 0.4222 | 0.2500 | 0.0000 |
| similarity_component_split | Equal_mean | 45 | 0.9818 | 0.9696 | 0.1161 | 0.6253 |
| similarity_component_split | Fingerprint_LR | 45 | 1.0000 | 1.0000 | 0.1266 | 0.5849 |
| similarity_component_split | GBT | 45 | 0.9322 | 0.8612 | 0.1958 | 0.5607 |
| similarity_component_split | Mean_without_GBT | 45 | 1.0000 | 1.0000 | 0.1073 | 0.5849 |
| similarity_component_split | Mean_without_Nearest_active | 45 | 0.9818 | 0.9696 | 0.1231 | 0.6253 |
| similarity_component_split | Mean_without_RF | 45 | 0.9818 | 0.9696 | 0.1101 | 0.6253 |
| similarity_component_split | Mean_without_SVM_RBF | 45 | 0.9818 | 0.9698 | 0.1315 | 0.6253 |
| similarity_component_split | Nearest_active | 45 | 1.0000 | 1.0000 | 0.1093 | 0.6253 |
| similarity_component_split | Property_LR | 45 | 1.0000 | 1.0000 | 0.0227 | 0.9115 |
| similarity_component_split | RF | 45 | 1.0000 | 1.0000 | 0.1517 | 0.2523 |
| similarity_component_split | SVM_RBF | 45 | 1.0000 | 1.0000 | 0.0778 | 0.7054 |
| similarity_component_split | Tanimoto_kNN | 45 | 0.7935 | 0.6903 | 0.2702 | -0.0481 |
| similarity_component_split | Train_prevalence | 45 | 0.1569 | 0.3753 | 0.3959 | 0.0000 |
| measured_source_below_1.0_uM | Equal_mean | 10 | 1.0000 | 1.0000 | 0.1138 | 0.6124 |
| measured_source_below_1.0_uM | GBT | 10 | 0.9375 | 0.6667 | 0.2091 | 0.6124 |
| measured_source_below_1.0_uM | Nearest_active | 10 | 1.0000 | 1.0000 | 0.1049 | 0.6667 |
| measured_source_below_1.0_uM | Property_LR | 10 | 0.5625 | 0.2917 | 0.5550 | 0.0000 |
| measured_source_below_1.0_uM | RF | 10 | 1.0000 | 1.0000 | 0.1111 | 0.7638 |
| measured_source_below_1.0_uM | SVM_RBF | 10 | 1.0000 | 1.0000 | 0.1171 | 0.7638 |
| measured_source_below_10.0_uM | Equal_mean | 10 | 1.0000 | 1.0000 | 0.1138 | 0.6124 |
| measured_source_below_10.0_uM | GBT | 10 | 0.9375 | 0.6667 | 0.2091 | 0.6124 |
| measured_source_below_10.0_uM | Nearest_active | 10 | 1.0000 | 1.0000 | 0.1049 | 0.6667 |
| measured_source_below_10.0_uM | Property_LR | 10 | 0.5625 | 0.2917 | 0.5550 | 0.0000 |
| measured_source_below_10.0_uM | RF | 10 | 1.0000 | 1.0000 | 0.1111 | 0.7638 |
| measured_source_below_10.0_uM | SVM_RBF | 10 | 1.0000 | 1.0000 | 0.1171 | 0.7638 |
| measured_source_below_50.0_uM | Equal_mean | 10 | 1.0000 | 1.0000 | 0.1170 | 0.8165 |
| measured_source_below_50.0_uM | GBT | 10 | 0.9400 | 0.9250 | 0.0788 | 0.8165 |
| measured_source_below_50.0_uM | Nearest_active | 10 | 1.0000 | 1.0000 | 0.1838 | 0.3333 |
| measured_source_below_50.0_uM | Property_LR | 10 | 0.8000 | 0.7100 | 0.2801 | 0.0000 |
| measured_source_below_50.0_uM | RF | 10 | 1.0000 | 1.0000 | 0.1651 | 0.6547 |
| measured_source_below_50.0_uM | SVM_RBF | 10 | 1.0000 | 1.0000 | 0.1274 | 0.6547 |
| authenticated_reference_molecule | Equal_mean | 45 | 0.9980 | 0.9974 | 0.0228 | 0.9551 |
| authenticated_reference_molecule | GBT | 45 | 0.9302 | 0.8819 | 0.0446 | 0.9089 |
| authenticated_reference_molecule | Nearest_active | 45 | 1.0000 | 1.0000 | 0.0457 | 1.0000 |
| authenticated_reference_molecule | Property_LR | 45 | 0.9939 | 0.9928 | 0.0319 | 0.9115 |
| authenticated_reference_molecule | RF | 45 | 1.0000 | 1.0000 | 0.0255 | 0.9115 |
| authenticated_reference_molecule | SVM_RBF | 45 | 1.0000 | 1.0000 | 0.0143 | 1.0000 |
| authenticated_reference_scaffold | Equal_mean | 45 | 0.9312 | 0.9512 | 0.0565 | 0.8633 |
| authenticated_reference_scaffold | GBT | 45 | 0.8462 | 0.7989 | 0.0889 | 0.8178 |
| authenticated_reference_scaffold | Nearest_active | 45 | 0.9150 | 0.9443 | 0.0749 | 0.9115 |
| authenticated_reference_scaffold | Property_LR | 45 | 0.9798 | 0.9776 | 0.0466 | 0.9115 |
| authenticated_reference_scaffold | RF | 45 | 0.9241 | 0.9457 | 0.0528 | 0.9115 |
| authenticated_reference_scaffold | SVM_RBF | 45 | 0.9696 | 0.9697 | 0.0489 | 0.9115 |
| authenticated_reference_series | Equal_mean | 45 | 0.5850 | 0.4446 | 0.3767 | 0.0000 |
| authenticated_reference_series | GBT | 45 | 0.2895 | 0.3975 | 0.4222 | 0.0000 |
| authenticated_reference_series | Nearest_active | 45 | 0.3502 | 0.3540 | 0.3388 | 0.0000 |
| authenticated_reference_series | Property_LR | 45 | 0.9960 | 0.9946 | 0.0327 | 0.9115 |
| authenticated_reference_series | RF | 45 | 0.6417 | 0.5149 | 0.3871 | 0.0000 |
| authenticated_reference_series | SVM_RBF | 45 | 0.7449 | 0.5879 | 0.3687 | 0.0000 |
| authenticated_reference_source_below_1.0_uM | Equal_mean | 10 | 1.0000 | 1.0000 | 0.1401 | 0.6124 |
| authenticated_reference_source_below_1.0_uM | GBT | 10 | 0.9375 | 0.6667 | 0.2700 | 0.6124 |
| authenticated_reference_source_below_1.0_uM | Nearest_active | 10 | 1.0000 | 1.0000 | 0.1049 | 0.6667 |
| authenticated_reference_source_below_1.0_uM | Property_LR | 10 | 0.5625 | 0.2917 | 0.5761 | 0.0000 |
| authenticated_reference_source_below_1.0_uM | RF | 10 | 0.9688 | 0.8333 | 0.1194 | 0.7638 |
| authenticated_reference_source_below_1.0_uM | SVM_RBF | 10 | 1.0000 | 1.0000 | 0.1454 | 0.7638 |
| authenticated_reference_source_below_10.0_uM | Equal_mean | 10 | 1.0000 | 1.0000 | 0.1401 | 0.6124 |
| authenticated_reference_source_below_10.0_uM | GBT | 10 | 0.9375 | 0.6667 | 0.2700 | 0.6124 |
| authenticated_reference_source_below_10.0_uM | Nearest_active | 10 | 1.0000 | 1.0000 | 0.1049 | 0.6667 |
| authenticated_reference_source_below_10.0_uM | Property_LR | 10 | 0.5625 | 0.2917 | 0.5761 | 0.0000 |
| authenticated_reference_source_below_10.0_uM | RF | 10 | 0.9688 | 0.8333 | 0.1194 | 0.7638 |
| authenticated_reference_source_below_10.0_uM | SVM_RBF | 10 | 1.0000 | 1.0000 | 0.1454 | 0.7638 |
| authenticated_reference_source_below_50.0_uM | Equal_mean | 10 | 1.0000 | 1.0000 | 0.1161 | 0.8165 |
| authenticated_reference_source_below_50.0_uM | GBT | 10 | 0.9400 | 0.9250 | 0.0866 | 0.8165 |
| authenticated_reference_source_below_50.0_uM | Nearest_active | 10 | 1.0000 | 1.0000 | 0.1838 | 0.3333 |
| authenticated_reference_source_below_50.0_uM | Property_LR | 10 | 0.8000 | 0.7100 | 0.2985 | 0.0000 |
| authenticated_reference_source_below_50.0_uM | RF | 10 | 1.0000 | 1.0000 | 0.1514 | 0.6547 |
| authenticated_reference_source_below_50.0_uM | SVM_RBF | 10 | 1.0000 | 1.0000 | 0.1218 | 0.6547 |
| ablation_full_valid_set | clogp_only_lr | 45 | 0.7895 | 0.7860 | 0.1785 | 0.3989 |
| ablation_full_valid_set | ecfp4_plus_properties_rf | 45 | 0.9281 | 0.9467 | 0.0499 | 0.9115 |
| ablation_full_valid_set | ecfp4_rf | 45 | 0.9413 | 0.9548 | 0.0513 | 0.9115 |
| ablation_full_valid_set | fsp3_only_lr | 45 | 0.7510 | 0.6342 | 0.2089 | 0.2493 |
| ablation_full_valid_set | hba_only_lr | 45 | 0.9960 | 0.9947 | 0.0425 | 0.9115 |
| ablation_full_valid_set | hbd_only_lr | 45 | 0.7176 | 0.6318 | 0.2285 | 0.0225 |
| ablation_full_valid_set | mw_only_lr | 45 | 0.9879 | 0.9863 | 0.0439 | 0.9115 |
| ablation_full_valid_set | nrb_only_lr | 45 | 0.8704 | 0.7804 | 0.1477 | 0.7267 |
| ablation_full_valid_set | properties_lr | 45 | 0.9798 | 0.9776 | 0.0436 | 0.9115 |
| ablation_full_valid_set | tpsa_only_lr | 45 | 0.9960 | 0.9946 | 0.0416 | 0.8633 |
| ablation_full_valid_set | without_clogp_lr | 45 | 0.9838 | 0.9813 | 0.0396 | 0.9115 |
| ablation_full_valid_set | without_fsp3_lr | 45 | 0.9838 | 0.9813 | 0.0401 | 0.9115 |
| ablation_full_valid_set | without_hba_lr | 45 | 0.9676 | 0.9682 | 0.0512 | 0.9115 |
| ablation_full_valid_set | without_hbd_lr | 45 | 0.9838 | 0.9813 | 0.0412 | 0.9115 |
| ablation_full_valid_set | without_mw_lr | 45 | 0.9798 | 0.9776 | 0.0491 | 0.9115 |
| ablation_full_valid_set | without_nrb_lr | 45 | 0.9818 | 0.9795 | 0.0395 | 0.9115 |
| ablation_full_valid_set | without_tpsa_lr | 45 | 0.9798 | 0.9776 | 0.0509 | 0.9115 |
| ablation_nearest_property_subset | clogp_only_lr | 38 | 0.7230 | 0.7610 | 0.2072 | 0.2108 |
| ablation_nearest_property_subset | ecfp4_plus_properties_rf | 38 | 0.9294 | 0.9563 | 0.0572 | 0.8997 |
| ablation_nearest_property_subset | ecfp4_rf | 38 | 0.9183 | 0.9520 | 0.0600 | 0.8997 |
| ablation_nearest_property_subset | fsp3_only_lr | 38 | 0.7742 | 0.7298 | 0.1866 | 0.6325 |
| ablation_nearest_property_subset | hba_only_lr | 38 | 0.9972 | 0.9947 | 0.0414 | 0.8997 |
| ablation_nearest_property_subset | hbd_only_lr | 38 | 0.6814 | 0.6783 | 0.2151 | 0.4763 |
| ablation_nearest_property_subset | mw_only_lr | 38 | 1.0000 | 1.0000 | 0.0409 | 0.8997 |
| ablation_nearest_property_subset | nrb_only_lr | 38 | 0.8740 | 0.8275 | 0.1246 | 0.7379 |
| ablation_nearest_property_subset | properties_lr | 38 | 0.9945 | 0.9946 | 0.0425 | 0.8997 |
| ablation_nearest_property_subset | tpsa_only_lr | 38 | 0.9945 | 0.9946 | 0.0382 | 0.8433 |
| ablation_nearest_property_subset | without_clogp_lr | 38 | 0.9945 | 0.9946 | 0.0379 | 0.8997 |
| ablation_nearest_property_subset | without_fsp3_lr | 38 | 0.9945 | 0.9946 | 0.0370 | 0.8997 |
| ablation_nearest_property_subset | without_hba_lr | 38 | 0.9834 | 0.9853 | 0.0509 | 0.8997 |
| ablation_nearest_property_subset | without_hbd_lr | 38 | 0.9945 | 0.9946 | 0.0413 | 0.8997 |
| ablation_nearest_property_subset | without_mw_lr | 38 | 0.9861 | 0.9876 | 0.0470 | 0.8997 |
| ablation_nearest_property_subset | without_nrb_lr | 38 | 0.9945 | 0.9946 | 0.0429 | 0.8997 |
| ablation_nearest_property_subset | without_tpsa_lr | 38 | 0.9751 | 0.9795 | 0.0470 | 0.8997 |
| ablation_similarity_component_split | clogp_only_lr | 45 | 0.0931 | 0.2752 | 0.4125 | -0.3353 |
| ablation_similarity_component_split | ecfp4_plus_properties_rf | 45 | 1.0000 | 1.0000 | 0.1237 | 0.5439 |
| ablation_similarity_component_split | ecfp4_rf | 45 | 1.0000 | 1.0000 | 0.1517 | 0.2523 |
| ablation_similarity_component_split | fsp3_only_lr | 45 | 0.0810 | 0.3036 | 0.4362 | -0.2670 |
| ablation_similarity_component_split | hba_only_lr | 45 | 0.9970 | 0.9946 | 0.0318 | 0.9115 |
| ablation_similarity_component_split | hbd_only_lr | 45 | 0.3715 | 0.4072 | 0.3582 | -0.2029 |
| ablation_similarity_component_split | mw_only_lr | 45 | 1.0000 | 1.0000 | 0.0286 | 0.9115 |
| ablation_similarity_component_split | nrb_only_lr | 45 | 0.7581 | 0.6461 | 0.2718 | -0.2285 |
| ablation_similarity_component_split | properties_lr | 45 | 1.0000 | 1.0000 | 0.0227 | 0.9115 |
| ablation_similarity_component_split | tpsa_only_lr | 45 | 0.9980 | 0.9974 | 0.0289 | 0.9089 |
| ablation_similarity_component_split | without_clogp_lr | 45 | 1.0000 | 1.0000 | 0.0230 | 0.9115 |
| ablation_similarity_component_split | without_fsp3_lr | 45 | 1.0000 | 1.0000 | 0.0220 | 0.9115 |
| ablation_similarity_component_split | without_hba_lr | 45 | 0.9960 | 0.9950 | 0.0290 | 0.9115 |
| ablation_similarity_component_split | without_hbd_lr | 45 | 1.0000 | 1.0000 | 0.0203 | 0.9115 |
| ablation_similarity_component_split | without_mw_lr | 45 | 0.9980 | 0.9974 | 0.0273 | 0.9115 |
| ablation_similarity_component_split | without_nrb_lr | 45 | 1.0000 | 1.0000 | 0.0211 | 0.9115 |
| ablation_similarity_component_split | without_tpsa_lr | 45 | 0.9960 | 0.9946 | 0.0298 | 0.9115 |

## Source cutoff feasibility

No score is imputed when a cutoff lacks both classes; undefined metrics are omitted from all_metrics.csv, not treated as zero.

| Design | n | Positive | Negative | Status |
|---|---:|---:|---:|---|
| measured_source_below_1.0_uM | 10 | 2 | 8 | available |
| measured_source_below_10.0_uM | 10 | 2 | 8 | available |
| measured_source_below_50.0_uM | 10 | 5 | 5 | available |
| authenticated_reference_source_below_1.0_uM | 10 | 2 | 8 | available |
| authenticated_reference_source_below_10.0_uM | 10 | 2 | 8 | available |
| authenticated_reference_source_below_50.0_uM | 10 | 5 | 5 | available |

## All fixed-seed runs

| Design | Seed | Property LR AUC | Equal mean AUC |
|---|---:|---:|---:|
| full_valid_set | 42 | 0.9798 | 0.9312 |
| full_valid_set | 43 | 0.9798 | 0.9332 |
| full_valid_set | 44 | 0.9858 | 0.9211 |
| full_valid_set | 45 | 0.9858 | 0.9372 |
| full_valid_set | 46 | 0.9798 | 0.9352 |
| nearest_property_subset | 42 | 0.9945 | 0.9280 |
| nearest_property_subset | 43 | 0.9945 | 0.9280 |
| nearest_property_subset | 44 | 0.9889 | 0.9197 |
| nearest_property_subset | 45 | 0.9889 | 0.9418 |
| nearest_property_subset | 46 | 0.9778 | 0.9391 |
| similarity_component_split | 42 | 1.0000 | 0.9818 |
| similarity_component_split | 43 | 1.0000 | 0.9798 |
| similarity_component_split | 44 | 1.0000 | 0.9696 |
| similarity_component_split | 45 | 1.0000 | 0.9798 |
| similarity_component_split | 46 | 1.0000 | 1.0000 |

## Conformal stress test

No coverage guarantee under scaffold shift. Classwise coverage, class counts, p-values and all prediction sets are in controls.json.

| Cohort | Nominal coverage | Empirical coverage | Mean set size | Singleton rate |
|---|---:|---:|---:|---:|
| full_valid_set | 0.90 | 0.978 | 1.778 | 0.222 |
| full_valid_set | 0.80 | 0.844 | 1.089 | 0.867 |
| nearest_property_subset | 0.90 | 1.000 | 1.868 | 0.132 |
| nearest_property_subset | 0.80 | 0.921 | 1.316 | 0.632 |
| similarity_component_split | 0.90 | 0.978 | 1.956 | 0.044 |
| similarity_component_split | 0.80 | 0.956 | 1.933 | 0.067 |
