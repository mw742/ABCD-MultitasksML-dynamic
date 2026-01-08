# ABCD-MultitasksML-dynamic


**Multi-task discrete-time prediction of substance use initiation using longitudinal risk windows in the ABCD Study**  
Mengman Wei\* and Qian Peng\*


---

## 1) Repository contents

Main scripts:
- `00_make_splits_dynamic.R`  
  Create train/val/test splits (subject-level splitting).
- `01_build_mtl_dynamic_dataset.R`  
  Build the longitudinal interval dataset with labels + observation masks.
- `02_train_mtl_dynamic.py`  
  Train and evaluate the dynamic multi-task model (shared MLP + joint output).

---

## 2) Data access and notes

This project uses **ABCD Release 5.1**.  
**No ABCD raw data are included in this repo.** You must have authorized access and follow ABCD/NIH data use rules.

---

## 3) Citation

If you use or adapt this project’s ideas and scripts, please cite the associated manuscript:

**Multi-task discrete-time prediction of substance use initiation using longitudinal risk windows in the ABCD Study**  
Mengman Wei\* and Qian Peng\*


