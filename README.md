# EEG RSA Toolkit

[中文说明](README.zh-CN.md)

A Python/Tkinter GUI for task EEG representational similarity analysis using EEGLAB `.set` files.

## Quick start

Install Python (the launcher recommends 3.10 or 3.11), then double-click `启动_RSA.bat` on Windows. The launcher installs required Python packages. Prefer preprocessed, epoched `.set` files; keep any external `.fdt` beside its `.set`.

## Scope

The basic workflow averages trials and time points within a selected window, computes condition RDMs, and compares them with a model RDM. Correlation p-values do not replace group-level inference, permutation tests, or correction for multiple comparisons.

This repository preserves the supplied tool files. Upload checks cover file integrity and Python syntax where applicable; they do not establish scientific validation or end-to-end application testing.
