# ABCD-MultitasksML-dynamic


**Multi-task discrete-time prediction of substance use initiation using longitudinal risk windows in the ABCD Study**  
Mengman Wei*, Hanwen Zhang, and Qian Peng*


---

## 1) Repository contents

Main scripts:
- `00_make_splits_dynamic.R`  
  Create train/val/test splits (subject-level splitting).
- `01_build_mtl_dynamic_dataset.R`  
  Build the longitudinal interval dataset with labels + observation masks.
- `02_train_mtl_dynamic.py`  
  Train and evaluate the dynamic multi-task model (shared MLP + joint output).

- Dependencies

R/4.3.0 python/3.11.4 pytorch/2.5.0py311-cuda

---

## 2) Data access and notes

This project uses **ABCD Release 5.1**.  
**No ABCD raw data are included in this repo.** You must have authorized access and follow ABCD/NIH data use rules.

---

## 3) Citation

If you use this work, please cite:

Wei, M., Zhang, H., & Peng, Q. (2026). *Dynamic and Baseline Multi-Task Learning for Predicting Substance Use Initiation in the ABCD Study*. medRxiv. https://doi.org/10.64898/2026.04.10

