from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import json
import numpy as np
import pandas as pd
import mne
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr, pearsonr

try:
    import mne_rsa
except Exception:
    mne_rsa = None


SUPPORTED_DISTANCES = {
    "Correlation (1-r)": "correlation",
    "Euclidean": "euclidean",
    "Cosine": "cosine",
    "Cityblock (Manhattan)": "cityblock",
    "Chebyshev": "chebyshev",
}

SUPPORTED_RSA_METRICS = ["Spearman", "Pearson", "Kendall tau-a"]


@dataclass
class EEGInfo:
    path: str
    kind: str  # 'epochs' or 'raw'
    sfreq: float
    n_channels: int
    channel_names: List[str]
    tmin_ms: float | None
    tmax_ms: float | None
    event_counts: Dict[str, int]
    event_id: Dict[str, int]


class EEGHandle:
    def __init__(self, info: EEGInfo, obj):
        self.info = info
        self.obj = obj


def _event_counts_from_epochs(epochs: mne.Epochs) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for name, code in epochs.event_id.items():
        out[name] = int(np.sum(epochs.events[:, 2] == code))
    return out


def load_eeglab_set(path: str) -> EEGHandle:
    """Load epoched or continuous EEGLAB .set using MNE.

    Preference is epoched data because task-RSA is simplest when each trial is already epoched.
    If epoch loading fails, continuous loading is attempted and events are read from annotations.
    """
    path = str(Path(path).resolve())

    epochs_error = None
    try:
        epochs = mne.read_epochs_eeglab(path, verbose="ERROR")
        picks = mne.pick_types(epochs.info, eeg=True, meg=False, exclude="bads")
        if len(picks) == 0:
            picks = np.arange(len(epochs.ch_names))
        ch_names = [epochs.ch_names[i] for i in picks]
        info = EEGInfo(
            path=path,
            kind="epochs",
            sfreq=float(epochs.info["sfreq"]),
            n_channels=len(ch_names),
            channel_names=ch_names,
            tmin_ms=float(epochs.times[0] * 1000),
            tmax_ms=float(epochs.times[-1] * 1000),
            event_counts=_event_counts_from_epochs(epochs),
            event_id=dict(epochs.event_id),
        )
        return EEGHandle(info, epochs)
    except Exception as exc:
        epochs_error = exc

    try:
        raw = mne.io.read_raw_eeglab(path, preload=True, verbose="ERROR")
        events, event_id = mne.events_from_annotations(raw, verbose="ERROR")
        if len(event_id) == 0:
            raise RuntimeError("连续数据中没有读取到可用的 EEGLAB event/annotation。")
        counts = {name: int(np.sum(events[:, 2] == code)) for name, code in event_id.items()}
        picks = mne.pick_types(raw.info, eeg=True, meg=False, exclude="bads")
        if len(picks) == 0:
            picks = np.arange(len(raw.ch_names))
        ch_names = [raw.ch_names[i] for i in picks]
        info = EEGInfo(
            path=path,
            kind="raw",
            sfreq=float(raw.info["sfreq"]),
            n_channels=len(ch_names),
            channel_names=ch_names,
            tmin_ms=None,
            tmax_ms=None,
            event_counts=counts,
            event_id=dict(event_id),
        )
        return EEGHandle(info, raw)
    except Exception as raw_exc:
        raise RuntimeError(
            "无法读取该 .set 文件。\n\n"
            f"按分段数据读取失败：{epochs_error}\n\n"
            f"按连续数据读取失败：{raw_exc}\n\n"
            "请确认同名 .fdt（若有）与 .set 位于同一文件夹，并且文件可在 EEGLAB 中正常打开。"
        ) from raw_exc


def _pick_eeg_names(inst) -> List[str]:
    picks = mne.pick_types(inst.info, eeg=True, meg=False, exclude="bads")
    if len(picks) == 0:
        return list(inst.ch_names)
    return [inst.ch_names[i] for i in picks]


def extract_condition_patterns(
    handle: EEGHandle,
    conditions: List[str],
    tmin_ms: float,
    tmax_ms: float,
) -> Tuple[np.ndarray, List[str], Dict[str, int]]:
    """Return condition x channel pattern matrix.

    Each pattern is the mean EEG amplitude across trials and across the selected time window.
    For raw data, epochs are created around each selected event using exactly the selected window.
    """
    if len(conditions) < 2:
        raise ValueError("至少选择 2 个条件。")
    if tmax_ms <= tmin_ms:
        raise ValueError("时间窗终点必须大于起点。")

    tmin = tmin_ms / 1000.0
    tmax = tmax_ms / 1000.0

    if handle.info.kind == "epochs":
        epochs: mne.Epochs = handle.obj
        if tmin < epochs.times[0] - 1e-9 or tmax > epochs.times[-1] + 1e-9:
            raise ValueError(
                f"所选时间窗 {tmin_ms:g}–{tmax_ms:g} ms 超出数据范围 "
                f"{epochs.times[0]*1000:.1f}–{epochs.times[-1]*1000:.1f} ms。"
            )
        ch_names = _pick_eeg_names(epochs)
        patterns = []
        counts = {}
        for cond in conditions:
            if cond not in epochs.event_id:
                raise ValueError(f"数据中找不到条件：{cond}")
            ep = epochs[cond].copy().pick(ch_names).crop(tmin=tmin, tmax=tmax, include_tmax=True)
            data = ep.get_data(copy=True)  # trial x channel x time
            counts[cond] = int(data.shape[0])
            if data.shape[0] == 0:
                raise ValueError(f"条件 {cond} 没有有效 trial。")
            patterns.append(data.mean(axis=(0, 2)))
        return np.vstack(patterns), ch_names, counts

    raw: mne.io.BaseRaw = handle.obj
    events, event_id = mne.events_from_annotations(raw, verbose="ERROR")
    ch_names = _pick_eeg_names(raw)
    patterns = []
    counts = {}

    # Continuous task data: create epochs separately per condition to make dropped-trial counts clear.
    for cond in conditions:
        if cond not in event_id:
            raise ValueError(f"连续数据中找不到事件：{cond}")
        ep = mne.Epochs(
            raw,
            events,
            event_id={cond: event_id[cond]},
            tmin=tmin,
            tmax=tmax,
            baseline=None,
            preload=True,
            picks=ch_names,
            reject_by_annotation=True,
            verbose="ERROR",
        )
        data = ep.get_data(copy=True)
        counts[cond] = int(data.shape[0])
        if data.shape[0] == 0:
            raise ValueError(f"条件 {cond} 在所选时间窗下没有可用 trial。")
        patterns.append(data.mean(axis=(0, 2)))

    return np.vstack(patterns), ch_names, counts


def compute_rdm(patterns: np.ndarray, distance_label: str) -> np.ndarray:
    if distance_label not in SUPPORTED_DISTANCES:
        raise ValueError(f"不支持的距离：{distance_label}")
    if patterns.ndim != 2 or patterns.shape[0] < 2:
        raise ValueError("patterns 必须是 condition × feature 的二维矩阵。")
    metric = SUPPORTED_DISTANCES[distance_label]
    # Primary engine: MNE-RSA. It returns a condensed RDM, matching scipy.pdist.
    # The SciPy fallback keeps the teaching package usable if mne-rsa is not yet installed.
    if mne_rsa is not None:
        d = mne_rsa.compute_rdm(patterns, metric=metric)
    else:
        d = pdist(patterns, metric=metric)
    rdm = squareform(np.asarray(d, dtype=float))
    if not np.all(np.isfinite(rdm)):
        raise ValueError(
            "RDM 中出现 NaN/Inf。常见原因是某个条件在所选时间窗内所有通道完全相同，"
            "使 correlation/cosine distance 无法计算。"
        )
    return rdm


def validate_model_rdm(model: np.ndarray, n_conditions: int) -> np.ndarray:
    model = np.asarray(model, dtype=float)
    if model.shape != (n_conditions, n_conditions):
        raise ValueError(f"Model RDM 必须是 {n_conditions}×{n_conditions} 方阵。")
    if not np.all(np.isfinite(model)):
        raise ValueError("Model RDM 不能包含 NaN/Inf。")
    if not np.allclose(model, model.T, atol=1e-8):
        raise ValueError("Model RDM 必须对称。")
    if not np.allclose(np.diag(model), 0, atol=1e-8):
        raise ValueError("Model RDM 对角线必须为 0。")
    return model


def compare_rdms(neural_rdm: np.ndarray, model_rdm: np.ndarray, metric: str):
    if neural_rdm.shape != model_rdm.shape:
        raise ValueError("Neural RDM 与 Model RDM 尺寸不一致。")
    tri = np.triu_indices(neural_rdm.shape[0], 1)
    x = neural_rdm[tri]
    y = model_rdm[tri]
    if len(x) < 2:
        raise ValueError("至少需要 3 个条件才能稳定计算 RDM 相关。")

    # Prefer MNE-RSA for the RSA statistic itself.
    if mne_rsa is not None:
        metric_map = {
            "Spearman": "spearman",
            "Pearson": "pearson",
            "Kendall tau-a": "kendall-tau-a",
        }
        if metric not in metric_map:
            raise ValueError(f"不支持的 RDM 比较方法：{metric}")
        stat = float(mne_rsa.rsa(neural_rdm, model_rdm, metric=metric_map[metric], verbose=False))
    else:
        # Fallback: Spearman/Pearson are identical; Kendall tau-a requires MNE-RSA.
        if metric == "Spearman":
            stat = float(spearmanr(x, y).statistic)
        elif metric == "Pearson":
            stat = float(pearsonr(x, y).statistic)
        else:
            raise RuntimeError("Kendall tau-a 需要安装 mne-rsa。请运行 requirements.txt 依赖安装。")

    # Parametric p-values are provided only as descriptive convenience.
    # MNE-RSA's Kendall tau-a statistic does not map to scipy's tau-b p-value, so p=NaN there.
    if metric == "Spearman":
        p = float(spearmanr(x, y).pvalue)
    elif metric == "Pearson":
        p = float(pearsonr(x, y).pvalue)
    else:
        p = float("nan")

    return stat, p, x, y


def read_model_rdm_csv(path: str, conditions: List[str]) -> np.ndarray:
    """Read either a plain numeric CSV or a labeled square CSV."""
    p = Path(path)
    # First try labeled table: first column as index and matching condition names.
    try:
        df = pd.read_csv(p, index_col=0)
        if set(conditions).issubset(df.index.astype(str)) and set(conditions).issubset(df.columns.astype(str)):
            arr = df.loc[conditions, conditions].to_numpy(dtype=float)
            return validate_model_rdm(arr, len(conditions))
    except Exception:
        pass

    # Then plain numeric matrix.
    arr = pd.read_csv(p, header=None).to_numpy(dtype=float)
    return validate_model_rdm(arr, len(conditions))


def save_model_rdm_csv(path: str, model: np.ndarray, conditions: List[str]):
    df = pd.DataFrame(model, index=conditions, columns=conditions)
    df.to_csv(path, encoding="utf-8-sig")


def export_results(
    output_dir: str,
    source_path: str,
    conditions: List[str],
    trial_counts: Dict[str, int],
    tmin_ms: float,
    tmax_ms: float,
    distance_label: str,
    rsa_metric: str,
    rsa_stat: float,
    rsa_p: float,
    patterns: np.ndarray,
    ch_names: List[str],
    neural_rdm: np.ndarray,
    model_rdm: np.ndarray,
):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(patterns, index=conditions, columns=ch_names).to_csv(
        out / "01_条件空间模式.csv", encoding="utf-8-sig"
    )
    pd.DataFrame(neural_rdm, index=conditions, columns=conditions).to_csv(
        out / "02_Neural_RDM.csv", encoding="utf-8-sig"
    )
    pd.DataFrame(model_rdm, index=conditions, columns=conditions).to_csv(
        out / "03_Model_RDM.csv", encoding="utf-8-sig"
    )

    result_df = pd.DataFrame(
        [{
            "source_file": source_path,
            "n_conditions": len(conditions),
            "time_window_ms": f"{tmin_ms:g} to {tmax_ms:g}",
            "distance": distance_label,
            "rsa_metric": rsa_metric,
            "rsa_value": rsa_stat,
            "p_value_descriptive_only": rsa_p,
        }]
    )
    result_df.to_csv(out / "04_RSA结果.csv", index=False, encoding="utf-8-sig")

    params = {
        "source_file": source_path,
        "conditions": conditions,
        "trial_counts": trial_counts,
        "time_window_ms": [tmin_ms, tmax_ms],
        "distance": distance_label,
        "rsa_metric": rsa_metric,
        "rsa_value": rsa_stat,
        "p_value_descriptive_only": rsa_p,
        "note": "当前版本的 p 值仅来自单个 RDM 向量相关，不替代被试组水平统计或置换检验。",
    }
    (out / "05_分析参数.json").write_text(json.dumps(params, ensure_ascii=False, indent=2), encoding="utf-8")

    return out
