from __future__ import annotations

import os
import traceback
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rsa_core import (
    SUPPORTED_DISTANCES,
    SUPPORTED_RSA_METRICS,
    load_eeglab_set,
    extract_condition_patterns,
    compute_rdm,
    compare_rdms,
    validate_model_rdm,
    read_model_rdm_csv,
    save_model_rdm_csv,
    export_results,
)


APP_TITLE = "EEG-RSA 一键分析工具 · Python版"


class RSAApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1040x760")
        self.minsize(980, 700)

        self.eeg_handle = None
        self.model_rdm = None
        self.model_conditions = None
        self.last_output_dir = None

        self._configure_style()
        self._build_ui()

    def _configure_style(self):
        style = ttk.Style(self)
        if "vista" in style.theme_names():
            style.theme_use("vista")
        style.configure("Title.TLabel", font=("Microsoft YaHei UI", 18, "bold"))
        style.configure("Subtitle.TLabel", font=("Microsoft YaHei UI", 10))
        style.configure("Step.TLabelframe.Label", font=("Microsoft YaHei UI", 10, "bold"))
        style.configure("Primary.TButton", font=("Microsoft YaHei UI", 10, "bold"), padding=8)
        style.configure("TButton", padding=6)
        style.configure("TLabel", font=("Microsoft YaHei UI", 9))
        style.configure("TCombobox", padding=3)

    def _build_ui(self):
        root = ttk.Frame(self, padding=18)
        root.pack(fill="both", expand=True)

        ttk.Label(root, text="EEG-RSA 一键分析工具", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            root,
            text="任务态 .set → 选择条件 → 选择时间窗 → Neural RDM → Model RDM → RSA → 导出",
            style="Subtitle.TLabel",
        ).pack(anchor="w", pady=(2, 14))

        body = ttk.Frame(root)
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        left = ttk.Frame(body)
        right = ttk.Frame(body)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        right.grid(row=0, column=1, sticky="nsew", padx=(8, 0))

        # Step 1
        f1 = ttk.LabelFrame(left, text="1  导入任务态 EEGLAB .set", style="Step.TLabelframe", padding=10)
        f1.pack(fill="x", pady=(0, 10))
        row = ttk.Frame(f1)
        row.pack(fill="x")
        self.path_var = tk.StringVar()
        ttk.Entry(row, textvariable=self.path_var).pack(side="left", fill="x", expand=True, padx=(0, 8))
        ttk.Button(row, text="选择 .set", command=self.choose_set).pack(side="left")
        self.info_text = tk.Text(f1, height=5, wrap="word", relief="flat", background=self.cget("bg"))
        self.info_text.pack(fill="x", pady=(8, 0))
        self.info_text.configure(state="disabled")

        # Step 2
        f2 = ttk.LabelFrame(left, text="2  选择条件 / 事件（至少 3 个更适合 RSA）", style="Step.TLabelframe", padding=10)
        f2.pack(fill="both", expand=True, pady=(0, 10))
        listrow = ttk.Frame(f2)
        listrow.pack(fill="both", expand=True)
        self.condition_list = tk.Listbox(listrow, selectmode="extended", exportselection=False, height=12)
        self.condition_list.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(listrow, orient="vertical", command=self.condition_list.yview)
        sb.pack(side="right", fill="y")
        self.condition_list.configure(yscrollcommand=sb.set)
        ttk.Button(f2, text="全选", command=lambda: self.condition_list.select_set(0, tk.END)).pack(anchor="e", pady=(6, 0))

        # Step 3
        f3 = ttk.LabelFrame(right, text="3  分析参数", style="Step.TLabelframe", padding=10)
        f3.pack(fill="x", pady=(0, 10))
        grid = ttk.Frame(f3)
        grid.pack(fill="x")
        ttk.Label(grid, text="时间窗起点 (ms)").grid(row=0, column=0, sticky="w", pady=4)
        ttk.Label(grid, text="时间窗终点 (ms)").grid(row=1, column=0, sticky="w", pady=4)
        ttk.Label(grid, text="Neural RDM 距离").grid(row=2, column=0, sticky="w", pady=4)
        ttk.Label(grid, text="RDM 比较方法").grid(row=3, column=0, sticky="w", pady=4)

        self.tmin_var = tk.StringVar(value="140")
        self.tmax_var = tk.StringVar(value="200")
        self.distance_var = tk.StringVar(value="Correlation (1-r)")
        self.rsa_metric_var = tk.StringVar(value="Spearman")
        ttk.Entry(grid, textvariable=self.tmin_var, width=18).grid(row=0, column=1, sticky="ew", padx=(10, 0))
        ttk.Entry(grid, textvariable=self.tmax_var, width=18).grid(row=1, column=1, sticky="ew", padx=(10, 0))
        ttk.Combobox(grid, textvariable=self.distance_var, values=list(SUPPORTED_DISTANCES.keys()), state="readonly").grid(row=2, column=1, sticky="ew", padx=(10, 0))
        ttk.Combobox(grid, textvariable=self.rsa_metric_var, values=SUPPORTED_RSA_METRICS, state="readonly").grid(row=3, column=1, sticky="ew", padx=(10, 0))
        grid.columnconfigure(1, weight=1)

        ttk.Label(
            f3,
            text="当前基础版：每个条件在所选时间窗内，对 trial 与时间点取平均，得到 condition × channel 空间模式。",
            wraplength=430,
        ).pack(anchor="w", pady=(8, 0))

        # Step 4
        f4 = ttk.LabelFrame(right, text="4  Model RDM", style="Step.TLabelframe", padding=10)
        f4.pack(fill="x", pady=(0, 10))
        btnrow = ttk.Frame(f4)
        btnrow.pack(fill="x")
        ttk.Button(btnrow, text="建立 / 编辑 Model RDM", command=self.open_model_editor).pack(side="left", padx=(0, 6))
        ttk.Button(btnrow, text="导入 Model RDM CSV", command=self.import_model).pack(side="left", padx=6)
        ttk.Button(btnrow, text="导出模板", command=self.export_model_template).pack(side="left", padx=6)
        self.model_status_var = tk.StringVar(value="尚未建立 Model RDM")
        ttk.Label(f4, textvariable=self.model_status_var).pack(anchor="w", pady=(8, 0))

        # Step 5
        f5 = ttk.LabelFrame(right, text="5  运行与导出", style="Step.TLabelframe", padding=10)
        f5.pack(fill="both", expand=True)
        ttk.Button(f5, text="开始 RSA 分析", style="Primary.TButton", command=self.run_analysis).pack(fill="x")
        ttk.Button(f5, text="打开最近结果文件夹", command=self.open_output_folder).pack(fill="x", pady=(8, 0))
        self.status_var = tk.StringVar(value="等待导入数据")
        ttk.Label(f5, textvariable=self.status_var, wraplength=430).pack(anchor="w", pady=(10, 4))
        self.result_text = tk.Text(f5, height=9, wrap="word")
        self.result_text.pack(fill="both", expand=True, pady=(4, 0))
        self.result_text.insert("1.0", "结果将在这里显示。")
        self.result_text.configure(state="disabled")

        footer = ttk.Label(
            root,
            text="说明：本工具用于教学与科研分析辅助。单被试 RDM 相关的 p 值不能替代被试组水平统计或正式置换检验。",
            style="Subtitle.TLabel",
        )
        footer.pack(anchor="w", pady=(12, 0))

    def _set_text(self, widget: tk.Text, text: str):
        widget.configure(state="normal")
        widget.delete("1.0", tk.END)
        widget.insert("1.0", text)
        widget.configure(state="disabled")

    def choose_set(self):
        path = filedialog.askopenfilename(title="选择 EEGLAB .set 文件", filetypes=[("EEGLAB SET", "*.set"), ("All files", "*.*")])
        if not path:
            return
        self.path_var.set(path)
        self.status_var.set("正在读取 .set ...")
        self.update_idletasks()
        try:
            handle = load_eeglab_set(path)
            self.eeg_handle = handle
            self.condition_list.delete(0, tk.END)
            for name, count in handle.info.event_counts.items():
                self.condition_list.insert(tk.END, f"{name}    (n={count})")
            self.condition_list.itemconfig(0, selectforeground="white")
            kind_cn = "已分段任务态数据" if handle.info.kind == "epochs" else "连续任务态数据"
            time_text = ""
            if handle.info.kind == "epochs":
                time_text = f"\nEpoch 时间范围：{handle.info.tmin_ms:.1f}–{handle.info.tmax_ms:.1f} ms"
            info = (
                f"读取成功：{kind_cn}\n"
                f"采样率：{handle.info.sfreq:g} Hz    EEG通道：{handle.info.n_channels}    事件类型：{len(handle.info.event_counts)}"
                f"{time_text}\n"
                "左下方已列出事件类型和数量，请选择需要进入 RDM 的条件。"
            )
            self._set_text(self.info_text, info)
            self.status_var.set("数据读取完成")
            self.model_rdm = None
            self.model_conditions = None
            self.model_status_var.set("条件变化后请重新建立 Model RDM")
        except Exception as exc:
            self.eeg_handle = None
            self.status_var.set("读取失败")
            messagebox.showerror("读取失败", str(exc))

    def get_selected_conditions(self):
        if self.eeg_handle is None:
            return []
        names = list(self.eeg_handle.info.event_counts.keys())
        return [names[i] for i in self.condition_list.curselection()]

    def open_model_editor(self):
        conditions = self.get_selected_conditions()
        if len(conditions) < 3:
            messagebox.showwarning("条件不足", "请先选择至少 3 个条件，再建立 Model RDM。")
            return
        ModelEditor(self, conditions, self._receive_model, self.model_rdm if self.model_conditions == conditions else None)

    def _receive_model(self, model, conditions):
        self.model_rdm = np.asarray(model, dtype=float)
        self.model_conditions = list(conditions)
        self.model_status_var.set(f"Model RDM 已建立：{len(conditions)}×{len(conditions)}")

    def import_model(self):
        conditions = self.get_selected_conditions()
        if len(conditions) < 3:
            messagebox.showwarning("条件不足", "请先选择至少 3 个条件。")
            return
        path = filedialog.askopenfilename(title="选择 Model RDM CSV", filetypes=[("CSV", "*.csv"), ("All files", "*.*")])
        if not path:
            return
        try:
            model = read_model_rdm_csv(path, conditions)
            self._receive_model(model, conditions)
        except Exception as exc:
            messagebox.showerror("Model RDM 导入失败", str(exc))

    def export_model_template(self):
        conditions = self.get_selected_conditions()
        if len(conditions) < 3:
            messagebox.showwarning("条件不足", "请先选择至少 3 个条件。")
            return
        path = filedialog.asksaveasfilename(title="保存 Model RDM 模板", defaultextension=".csv", filetypes=[("CSV", "*.csv")], initialfile="Model_RDM_template.csv")
        if not path:
            return
        model = np.zeros((len(conditions), len(conditions)), dtype=float)
        save_model_rdm_csv(path, model, conditions)
        messagebox.showinfo("已保存", "模板已保存。请在表格中填写对称的条件间不相似度，对角线保持 0。")

    def run_analysis(self):
        if self.eeg_handle is None:
            messagebox.showwarning("缺少数据", "请先导入任务态 .set。")
            return
        conditions = self.get_selected_conditions()
        if len(conditions) < 3:
            messagebox.showwarning("条件不足", "RSA 建议至少选择 3 个条件。")
            return
        if self.model_rdm is None or self.model_conditions != conditions:
            messagebox.showwarning("缺少 Model RDM", "请按当前所选条件建立或导入 Model RDM。")
            return
        try:
            tmin_ms = float(self.tmin_var.get())
            tmax_ms = float(self.tmax_var.get())
        except ValueError:
            messagebox.showerror("参数错误", "时间窗必须填写数字。")
            return

        out_dir = filedialog.askdirectory(title="选择结果保存文件夹")
        if not out_dir:
            return
        stamp = pd.Timestamp.now().strftime("RSA_%Y%m%d_%H%M%S")
        out_dir = str(Path(out_dir) / stamp)
        self.status_var.set("正在计算条件空间模式和 RDM ...")
        self.update_idletasks()

        try:
            patterns, ch_names, counts = extract_condition_patterns(
                self.eeg_handle, conditions, tmin_ms, tmax_ms
            )
            neural = compute_rdm(patterns, self.distance_var.get())
            model = validate_model_rdm(self.model_rdm, len(conditions))
            stat, p, _, _ = compare_rdms(neural, model, self.rsa_metric_var.get())

            export_results(
                out_dir,
                self.eeg_handle.info.path,
                conditions,
                counts,
                tmin_ms,
                tmax_ms,
                self.distance_var.get(),
                self.rsa_metric_var.get(),
                stat,
                p,
                patterns,
                ch_names,
                neural,
                model,
            )
            self._save_figures(out_dir, conditions, neural, model, stat, p)
            self.last_output_dir = out_dir
            count_text = ", ".join([f"{k}: {v}" for k, v in counts.items()])
            result = (
                f"分析完成\n\n"
                f"条件：{', '.join(conditions)}\n"
                f"有效 trial：{count_text}\n"
                f"时间窗：{tmin_ms:g}–{tmax_ms:g} ms\n"
                f"Neural RDM 距离：{self.distance_var.get()}\n"
                f"RDM 比较：{self.rsa_metric_var.get()}\n\n"
                f"RSA = {stat:.4f}\n"
                f"p = {p:.6g}  （Kendall tau-a 时不提供该描述性 p 值；均不替代组水平统计）\n\n"
                f"结果文件夹：\n{out_dir}"
            )
            self._set_text(self.result_text, result)
            self.status_var.set("分析完成")
            messagebox.showinfo("完成", "RSA 分析完成，结果已导出。")
        except Exception as exc:
            self.status_var.set("分析失败")
            err_path = Path(out_dir)
            err_path.mkdir(parents=True, exist_ok=True)
            (err_path / "error_log.txt").write_text(traceback.format_exc(), encoding="utf-8")
            messagebox.showerror("分析失败", f"{exc}\n\n完整错误日志已保存到：\n{err_path / 'error_log.txt'}")

    def _save_figures(self, out_dir, conditions, neural, model, stat, p):
        out = Path(out_dir)
        for matrix, name, title in [
            (neural, "06_Neural_RDM.png", "Neural RDM"),
            (model, "07_Model_RDM.png", "Model RDM"),
        ]:
            fig, ax = plt.subplots(figsize=(6.4, 5.4))
            im = ax.imshow(matrix, aspect="equal", interpolation="nearest")
            ax.set_xticks(range(len(conditions)), conditions, rotation=45, ha="right")
            ax.set_yticks(range(len(conditions)), conditions)
            ax.set_title(title)
            fig.colorbar(im, ax=ax, shrink=0.85)
            fig.tight_layout()
            fig.savefig(out / name, dpi=180, bbox_inches="tight")
            plt.close(fig)

        tri = np.triu_indices(len(conditions), 1)
        x = model[tri]
        y = neural[tri]
        fig, ax = plt.subplots(figsize=(6.0, 5.0))
        ax.scatter(x, y)
        ax.set_xlabel("Model dissimilarity")
        ax.set_ylabel("Neural dissimilarity")
        ax.set_title(f"RSA ({self.rsa_metric_var.get()}): {stat:.3f}" + (f", p={p:.3g}" if np.isfinite(p) else ""))
        fig.tight_layout()
        fig.savefig(out / "08_RSA散点图.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    def open_output_folder(self):
        if not self.last_output_dir or not Path(self.last_output_dir).exists():
            messagebox.showinfo("暂无结果", "当前还没有可打开的结果文件夹。")
            return
        path = str(Path(self.last_output_dir).resolve())
        try:
            os.startfile(path)  # Windows
        except AttributeError:
            import subprocess
            subprocess.Popen(["xdg-open", path])


class ModelEditor(tk.Toplevel):
    def __init__(self, master, conditions, callback, initial=None):
        super().__init__(master)
        self.title("Model RDM 编辑器")
        self.conditions = list(conditions)
        self.callback = callback
        self.n = len(conditions)
        self.entries = {}
        self.geometry(f"{min(980, 220 + self.n*85)}x{min(760, 210 + self.n*48)}")

        ttk.Label(
            self,
            text="只需填写上三角：0 = 更相似，数值越大 = 越不相似。保存时自动镜像到下三角。",
            wraplength=900,
        ).pack(anchor="w", padx=12, pady=(12, 8))

        canvas = tk.Canvas(self, highlightthickness=0)
        frame = ttk.Frame(canvas)
        vs = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        hs = ttk.Scrollbar(self, orient="horizontal", command=canvas.xview)
        canvas.configure(yscrollcommand=vs.set, xscrollcommand=hs.set)
        canvas.pack(side="left", fill="both", expand=True, padx=(12, 0), pady=(0, 48))
        vs.pack(side="right", fill="y", pady=(0, 48))
        hs.pack(side="bottom", fill="x", padx=(12, 20), pady=(0, 40))
        canvas.create_window((0, 0), window=frame, anchor="nw")
        frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        ttk.Label(frame, text="").grid(row=0, column=0, padx=3, pady=3)
        for j, name in enumerate(self.conditions):
            ttk.Label(frame, text=name, width=10, anchor="center").grid(row=0, column=j+1, padx=3, pady=3)
        for i, rname in enumerate(self.conditions):
            ttk.Label(frame, text=rname, width=14, anchor="e").grid(row=i+1, column=0, padx=3, pady=3)
            for j in range(self.n):
                if i == j:
                    ttk.Label(frame, text="0", width=9, anchor="center").grid(row=i+1, column=j+1, padx=3, pady=3)
                elif i < j:
                    val = ""
                    if initial is not None:
                        val = str(float(initial[i, j]))
                    ent = ttk.Entry(frame, width=9)
                    ent.insert(0, val)
                    ent.grid(row=i+1, column=j+1, padx=3, pady=3)
                    self.entries[(i, j)] = ent
                else:
                    ttk.Label(frame, text="—", width=9, anchor="center").grid(row=i+1, column=j+1, padx=3, pady=3)

        btnbar = ttk.Frame(self)
        btnbar.place(relx=0.5, rely=1.0, anchor="s", y=-8)
        ttk.Button(btnbar, text="示例：类别模型", command=self.fill_binary_example).pack(side="left", padx=6)
        ttk.Button(btnbar, text="保存 Model RDM", command=self.save).pack(side="left", padx=6)
        ttk.Button(btnbar, text="取消", command=self.destroy).pack(side="left", padx=6)

    def fill_binary_example(self):
        # Convenience only: first half similar, second half similar, between halves dissimilar.
        split = max(1, self.n // 2)
        for (i, j), ent in self.entries.items():
            v = 0 if ((i < split and j < split) or (i >= split and j >= split)) else 1
            ent.delete(0, tk.END)
            ent.insert(0, str(v))

    def save(self):
        model = np.zeros((self.n, self.n), dtype=float)
        try:
            for (i, j), ent in self.entries.items():
                txt = ent.get().strip()
                if txt == "":
                    raise ValueError(f"请填写 {self.conditions[i]} vs {self.conditions[j]} 的模型距离。")
                v = float(txt)
                model[i, j] = v
                model[j, i] = v
            validate_model_rdm(model, self.n)
        except Exception as exc:
            messagebox.showerror("Model RDM 错误", str(exc), parent=self)
            return
        self.callback(model, self.conditions)
        self.destroy()


def main():
    app = RSAApp()
    app.mainloop()


if __name__ == "__main__":
    main()
