# Recitation 2 – Code

This repository contains the source code used to support **Recitation 2**.  
Please note that this code is provided **for reference only** and is **not** part of the official recitation material.  

For the session itself, we adapted the model implementations and training pipeline into a single Google Colab notebook to ensure a smoother learning experience.  

If you choose to explore this codebase, feel free to do so — but keep in mind that it is intended as supplementary material rather than a guided component of the recitation.

## General Structure

```bash
.
├── data_prep   # Folder for data retrieval from mlogit R package
│   ├── data-generation.r
│   └── data-transformation.py  # Converting CSV data into one pickle
├── models
│   ├── __init__.py
│   ├── asu_dnn_architecture_visualization.txt  # ASCII visualization of the ASU-DNN
│   ├── asu_dnn_general.py
│   ├── data_loading.py
│   ├── fc_dnn.py
│   ├── model_evaluation.py
│   ├── multinomial_logit.py
│   └── simple_logistic_baseline.py
├── tests   # Code to test the implementations
│   ├── test_diverse.py
│   ├── test_fc_specific.py
│   ├── test_phase1.py
│   ├── test_phase2.py
│   ├── test_phase5.py
│   └── test_plotting.py
├── demo_choice_plots.py
├── final_demo.py
├── plot_demo.py
├── pyproject.toml
├── README.md
└── uv.lock
```
