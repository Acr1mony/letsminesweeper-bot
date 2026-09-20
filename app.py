from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk

from PIL import Image, ImageTk

from minesweeper.automation import ActionKind, LocalAutomation, StepOutcome
from minesweeper.input_control import emergency_pressed
from minesweeper.observer import Observation, WindowObserver, observe_window
from minesweeper.recognition import CellKind
from minesweeper.statistics import StatisticsSnapshot, StatisticsStore


COLORS = {
    "snow": "#EEF3F1",
    "paper": "#F9FBFA",
    "ink": "#263238",
    "muted": "#687773",
    "pine": "#285A4A",
    "safe": "#2E86AB",
    "mine": "#D9574E",
    "warning": "#CD8925",
    "line": "#CAD5D1",
}


def observation_report(observation: Observation) -> dict[str, object]:
    counts: dict[str, int] = {}
    for row in observation.grid:
        for cell in row:
            counts[cell.kind.value] = counts.get(cell.kind.value, 0) + 1
    geometry = observation.geometry
    return {
        "window_rect": observation.frame.window_rect,
        "visible_screen_fallback": observation.frame.used_visible_screen_fallback,
        "grid": {
            "origin": [geometry.origin_x, geometry.origin_y],
            "pitch": geometry.pitch,
            "rows": geometry.full_rows,
            "columns": geometry.full_columns,
        },
        "counts": counts,
        "safe": sorted([list(position) for position in observation.result.safe]),
        "mines": sorted([list(position) for position in observation.result.mines]),
        "contradictions": list(observation.result.contradictions),
    }


class ObserverApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Let's Minesweeper · 局部自动驾驶")
        self.root.geometry("900x560+10+30")
        self.root.minsize(700, 430)
        self.root.configure(bg=COLORS["snow"])
        self.photo: ImageTk.PhotoImage | None = None
        self.canvas_image: int | None = None
        self.render_job: str | None = None
        self.render_key: tuple[int, int, int] | None = None
        self.refresh_job: str | None = None
        self.refresh_thread: threading.Thread | None = None
        self.countdown_job: str | None = None
        self.stop_event = threading.Event()
        self.worker: threading.Thread | None = None
        self.events: queue.SimpleQueue[tuple[str, object]] = queue.SimpleQueue()
        self.statistics = StatisticsStore()
        self.automation = LocalAutomation(statistics=self.statistics)
        self.last_observation: Observation | None = None
        self.auto_refresh = tk.BooleanVar(value=False)
        self.status = tk.StringVar(value="准备读取当前视口")
        self.advice = tk.StringVar(value="")
        self.mode_text = tk.StringVar(value="观察")
        self.safe_text = tk.StringVar(value="—")
        self.mine_text = tk.StringVar(value="—")
        self.unknown_text = tk.StringVar(value="—")
        self.geometry_text = tk.StringVar(value="—")
        snapshot = self.statistics.snapshot()
        self.total_opened_text = tk.StringVar(value=str(snapshot.opened))
        self.total_flagged_text = tk.StringVar(value=str(snapshot.flagged))
        self._build()
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        self.root.after(80, self._poll_events)
        self.root.after(100, self._poll_emergency)
        self.root.after(150, self.refresh)

    def _build(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background=COLORS["snow"])
        style.configure("Sidebar.TFrame", background=COLORS["paper"])
        style.configure("TLabel", background=COLORS["snow"], foreground=COLORS["ink"], font=("Microsoft YaHei UI", 9))
        style.configure("Title.TLabel", font=("Microsoft YaHei UI", 15, "bold"), foreground=COLORS["pine"])
        style.configure("Mode.TLabel", font=("Microsoft YaHei UI", 9, "bold"), foreground=COLORS["mine"])
        style.configure("Metric.TLabel", background=COLORS["paper"], font=("Bahnschrift", 18, "bold"), foreground=COLORS["ink"])
        style.configure("Caption.TLabel", background=COLORS["paper"], font=("Microsoft YaHei UI", 8), foreground=COLORS["muted"])
        style.configure("TButton", font=("Microsoft YaHei UI", 9, "bold"), padding=(10, 6), background=COLORS["pine"], foreground="white")
        style.map("TButton", background=[("active", "#34715E")])
        style.configure("TCheckbutton", background=COLORS["paper"], font=("Microsoft YaHei UI", 9), foreground=COLORS["ink"])

        shell = ttk.Frame(self.root, padding=10)
        shell.pack(fill="both", expand=True)
        header = ttk.Frame(shell)
        header.pack(fill="x", pady=(0, 7))
        ttk.Label(header, text="局部自动驾驶", style="Title.TLabel").pack(side="left")
        ttk.Label(header, text="当前视口 · 本地离线", foreground=COLORS["muted"]).pack(side="left", padx=10, pady=(3, 0))
        ttk.Label(header, textvariable=self.mode_text, style="Mode.TLabel").pack(side="right", pady=(3, 0))

        content = ttk.Frame(shell)
        content.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(content, bg="#DDE5E2", highlightthickness=1, highlightbackground=COLORS["line"])
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas.bind("<Configure>", self._queue_render)

        sidebar_shell = ttk.Frame(content, style="Sidebar.TFrame", width=218)
        sidebar_shell.pack(side="right", fill="y", padx=(8, 0))
        sidebar_shell.pack_propagate(False)

        info_shell = ttk.Frame(sidebar_shell, style="Sidebar.TFrame")
        info_shell.pack(fill="both", expand=True)
        self.sidebar_canvas = tk.Canvas(
            info_shell,
            bg=COLORS["paper"],
            highlightthickness=0,
            width=198,
        )
        sidebar_scroll = ttk.Scrollbar(info_shell, orient="vertical", command=self.sidebar_canvas.yview)
        self.sidebar_canvas.configure(yscrollcommand=sidebar_scroll.set)
        sidebar_scroll.pack(side="right", fill="y")
        self.sidebar_canvas.pack(side="left", fill="both", expand=True)
        sidebar = ttk.Frame(self.sidebar_canvas, style="Sidebar.TFrame", padding=(10, 8))
        self.sidebar_window = self.sidebar_canvas.create_window((0, 0), window=sidebar, anchor="nw")
        sidebar.bind("<Configure>", self._sync_sidebar_scrollregion)
        self.sidebar_canvas.bind("<Configure>", self._fit_sidebar_width)
        self.sidebar_canvas.bind("<MouseWheel>", self._scroll_sidebar)
        sidebar.bind("<MouseWheel>", self._scroll_sidebar)

        current = ttk.Frame(sidebar, style="Sidebar.TFrame")
        current.pack(fill="x", pady=(0, 7))
        self._compact_metric(current, "安全", self.safe_text, COLORS["safe"])
        self._compact_metric(current, "标雷", self.mine_text, COLORS["mine"])
        self._compact_metric(current, "未开", self.unknown_text, COLORS["muted"])
        ttk.Separator(sidebar).pack(fill="x", pady=6)
        ttk.Label(sidebar, text="累计操作", style="Caption.TLabel").pack(anchor="w")
        totals = ttk.Frame(sidebar, style="Sidebar.TFrame")
        totals.pack(fill="x", pady=(4, 5))
        self._compact_metric(totals, "开格", self.total_opened_text, COLORS["safe"])
        self._compact_metric(totals, "标旗", self.total_flagged_text, COLORS["mine"])
        ttk.Button(sidebar, text="清除统计", command=self.clear_statistics).pack(fill="x", pady=(0, 7))
        ttk.Separator(sidebar).pack(fill="x", pady=(0, 7))
        ttk.Label(sidebar, text="当前网格", style="Caption.TLabel").pack(anchor="w")
        ttk.Label(sidebar, textvariable=self.geometry_text, style="Caption.TLabel", wraplength=175).pack(anchor="w", pady=(3, 7))
        ttk.Label(sidebar, text="蓝框安全 · 红框待标雷 · 绿框确认雷", style="Caption.TLabel", wraplength=175).pack(anchor="w")

        controls = ttk.Frame(sidebar_shell, style="Sidebar.TFrame", padding=(10, 7))
        controls.pack(fill="x", side="bottom")
        self.advice_label = tk.Label(
            sidebar,
            textvariable=self.advice,
            bg="#FFF0CF",
            fg="#9C2E25",
            font=("Microsoft YaHei UI", 8, "bold"),
            justify="left",
            wraplength=185,
            padx=6,
            pady=5,
        )
        self.primary_row = ttk.Frame(controls, style="Sidebar.TFrame")
        self.primary_row.pack(fill="x", pady=(0, 4))
        self.start_button = tk.Button(
            self.primary_row,
            text="开始自动运行",
            command=self.start_automatic,
            bg=COLORS["pine"],
            fg="white",
            activebackground="#34715E",
            activeforeground="white",
            relief="flat",
            font=("Microsoft YaHei UI", 8, "bold"),
            pady=3,
            cursor="hand2",
        )
        self.start_button.pack(side="left", fill="x", expand=True, padx=(0, 3))
        self.pause_button = ttk.Button(self.primary_row, text="暂停", command=self.pause_automatic)
        self.pause_button.pack(side="left", fill="x", expand=True, padx=(3, 0))
        action_row = ttk.Frame(controls, style="Sidebar.TFrame")
        action_row.pack(fill="x", pady=(0, 4))
        self.step_button = ttk.Button(action_row, text="执行一批", command=self.run_single_batch)
        self.step_button.pack(side="left", fill="x", expand=True, padx=(0, 3))
        self.refresh_button = ttk.Button(action_row, text="重新读取", command=self.refresh)
        self.refresh_button.pack(side="left", fill="x", expand=True, padx=(3, 0))
        utility_row = ttk.Frame(controls, style="Sidebar.TFrame")
        utility_row.pack(fill="x")
        self.auto_refresh_check = ttk.Checkbutton(
            utility_row,
            text="每秒自动刷新",
            variable=self.auto_refresh,
            command=self._schedule,
        )
        self.auto_refresh_check.pack(side="left")
        ttk.Label(utility_row, text="F8 急停", style="Caption.TLabel").pack(side="right")
        # Pack the fixed controls before the expanding information pane so the
        # packer never clips buttons when the window is short.
        controls.pack_forget()
        info_shell.pack_forget()
        controls.pack(fill="x", side="bottom")
        info_shell.pack(fill="both", expand=True)

        footer = ttk.Frame(shell)
        footer.pack(fill="x", pady=(6, 0))
        ttk.Label(footer, textvariable=self.status, foreground=COLORS["muted"]).pack(side="left")

    def _metric(self, parent: ttk.Frame, label: str, variable: tk.StringVar, color: str) -> None:
        block = ttk.Frame(parent, style="Sidebar.TFrame")
        block.pack(fill="x", pady=(0, 13))
        marker = tk.Frame(block, bg=color, width=5, height=45)
        marker.pack(side="left", fill="y", padx=(0, 12))
        text = ttk.Frame(block, style="Sidebar.TFrame")
        text.pack(side="left", fill="x")
        ttk.Label(text, text=label, style="Caption.TLabel").pack(anchor="w")
        ttk.Label(text, textvariable=variable, style="Metric.TLabel").pack(anchor="w")

    def _compact_metric(
        self,
        parent: ttk.Frame,
        label: str,
        variable: tk.StringVar,
        color: str,
    ) -> None:
        block = ttk.Frame(parent, style="Sidebar.TFrame")
        block.pack(side="left", fill="x", expand=True)
        tk.Label(
            block,
            textvariable=variable,
            bg=COLORS["paper"],
            fg=color,
            font=("Bahnschrift", 14, "bold"),
        ).pack(anchor="w")
        ttk.Label(block, text=label, style="Caption.TLabel").pack(anchor="w")

    def _sync_sidebar_scrollregion(self, _event: tk.Event | None = None) -> None:
        self.sidebar_canvas.configure(scrollregion=self.sidebar_canvas.bbox("all"))

    def _fit_sidebar_width(self, event: tk.Event) -> None:
        self.sidebar_canvas.itemconfigure(self.sidebar_window, width=max(1, event.width))

    def _scroll_sidebar(self, event: tk.Event) -> str:
        delta = -1 if event.delta > 0 else 1
        self.sidebar_canvas.yview_scroll(delta, "units")
        return "break"

    def _show_statistics(self, snapshot: StatisticsSnapshot | None = None) -> None:
        current = snapshot or self.statistics.snapshot()
        self.total_opened_text.set(str(current.opened))
        self.total_flagged_text.set(str(current.flagged))

    def clear_statistics(self) -> None:
        self._show_statistics(self.statistics.clear())
        self.status.set("累计开格和标旗统计已清零")

    def refresh(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            return
        if self.refresh_thread is not None and self.refresh_thread.is_alive():
            return
        if self.refresh_job is not None:
            self.root.after_cancel(self.refresh_job)
            self.refresh_job = None
        self.status.set("正在后台读取画面…")
        self.refresh_thread = threading.Thread(target=self._refresh_worker, daemon=True)
        self.refresh_thread.start()

    def _refresh_worker(self) -> None:
        try:
            self.events.put(("refresh", observe_window()))
        except Exception as error:
            self.events.put(("refresh_error", error))

    def _apply_observation(self, observation: Observation) -> None:
        self.last_observation = observation
        self._show(observation)
        report = observation_report(observation)
        counts = report["counts"]
        self.safe_text.set(str(len(observation.result.safe)))
        self.mine_text.set(str(len(observation.result.mines)))
        self.unknown_text.set(str(counts.get(CellKind.CLOSED.value, 0)))
        geometry = observation.geometry
        treasure_count = counts.get(CellKind.TREASURE.value, 0)
        self.geometry_text.set(
            f"{geometry.full_columns} 列 × {geometry.full_rows} 行\n"
            f"格距 {geometry.pitch:.2f} px\n宝箱 {treasure_count}"
        )

    def _show(self, observation: Observation) -> None:
        self.last_observation = observation
        self._queue_render()

    def _queue_render(self, _event: tk.Event | None = None) -> None:
        if self.render_job is not None:
            self.root.after_cancel(self.render_job)
        self.render_job = self.root.after(35, self._render_latest)

    def _render_latest(self) -> None:
        self.render_job = None
        if self.last_observation is None:
            return
        canvas_width = max(180, self.canvas.winfo_width() - 10)
        canvas_height = max(140, self.canvas.winfo_height() - 10)
        source = self.last_observation.annotated
        scale = min(canvas_width / source.width, canvas_height / source.height, 1.0)
        target_width = max(1, round(source.width * scale))
        target_height = max(1, round(source.height * scale))
        key = (id(self.last_observation), target_width, target_height)
        if key == self.render_key:
            return
        image = source if (target_width, target_height) == source.size else source.resize(
            (target_width, target_height),
            Image.Resampling.BILINEAR,
        )
        self.photo = ImageTk.PhotoImage(image)
        center_x = max(1, self.canvas.winfo_width()) // 2
        center_y = max(1, self.canvas.winfo_height()) // 2
        if self.canvas_image is None:
            self.canvas_image = self.canvas.create_image(center_x, center_y, image=self.photo, anchor="center")
        else:
            self.canvas.itemconfigure(self.canvas_image, image=self.photo)
            self.canvas.coords(self.canvas_image, center_x, center_y)
        self.render_key = key

    def _schedule(self) -> None:
        if self.refresh_job is not None:
            self.root.after_cancel(self.refresh_job)
            self.refresh_job = None
        if (
            self.auto_refresh.get()
            and not (self.worker is not None and self.worker.is_alive())
            and not (self.refresh_thread is not None and self.refresh_thread.is_alive())
        ):
            self.refresh_job = self.root.after(1000, self.refresh)

    def start_automatic(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            return
        self.pause_automatic()
        self._cancel_refresh()
        self.advice.set("")
        self.stop_event.clear()
        self.automation = LocalAutomation(observer=WindowObserver(), statistics=self.statistics)
        self.start_button.configure(state="disabled")
        self.step_button.configure(state="disabled")
        self._countdown(1)

    def _countdown(self, remaining: int) -> None:
        if self.stop_event.is_set():
            self._set_idle("已取消启动")
            return
        if remaining > 0:
            self.mode_text.set(f"{remaining} 秒后启动")
            self.status.set("请将鼠标移开游戏窗口；按 F8 可取消")
            self.countdown_job = self.root.after(1000, lambda: self._countdown(remaining - 1))
            return
        self.countdown_job = None
        self.mode_text.set("自动运行")
        self._launch_when_refresh_idle(True)

    def run_single_batch(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            return
        self._cancel_refresh()
        self.stop_event.clear()
        self.advice.set("")
        self.automation = LocalAutomation(observer=WindowObserver(), statistics=self.statistics)
        self.mode_text.set("执行一批")
        self.start_button.configure(state="disabled")
        self.step_button.configure(state="disabled")
        self.root.after(50, lambda: self._launch_when_refresh_idle(False))

    def _launch_when_refresh_idle(self, continuous: bool) -> None:
        if self.stop_event.is_set():
            self._set_idle("已取消启动")
            return
        if self.refresh_thread is not None and self.refresh_thread.is_alive():
            self.status.set("等待后台读取完成后启动…")
            self.root.after(50, lambda: self._launch_when_refresh_idle(continuous))
            return
        self.status.set("正在批量处理当前视口" if continuous else "正在执行当前批次")
        self.root.after(30, lambda: self._launch_worker(continuous))

    def _launch_worker(self, continuous: bool) -> None:
        if self.stop_event.is_set():
            self._set_idle("已取消启动")
            return
        self.worker = threading.Thread(target=self._automation_worker, args=(continuous,), daemon=True)
        self.worker.start()

    def _automation_worker(self, continuous: bool) -> None:
        try:
            while not self.stop_event.is_set():
                outcome = self.automation.perform_cycle(self.stop_event)
                self.events.put(("outcome", outcome))
                if not continuous or outcome.action.kind == ActionKind.COMPLETE:
                    break
        except Exception as error:
            self.events.put(("error", error))
        finally:
            self.events.put(("stopped", None))

    def pause_automatic(self) -> None:
        self.stop_event.set()
        if self.countdown_job is not None:
            self.root.after_cancel(self.countdown_job)
            self.countdown_job = None
        if self.worker is not None and self.worker.is_alive():
            self.mode_text.set("正在暂停")
            self.status.set("等待当前鼠标按键释放")
        else:
            self._set_idle("已暂停")

    def _poll_events(self) -> None:
        pending_outcome: StepOutcome | None = None
        pending_refresh: Observation | None = None
        pending_error: object | None = None
        stopped = False
        while True:
            try:
                kind, payload = self.events.get_nowait()
            except queue.Empty:
                break
            if kind == "outcome":
                assert isinstance(payload, StepOutcome)
                pending_outcome = payload
            elif kind == "refresh":
                assert isinstance(payload, Observation)
                pending_refresh = payload
                self.refresh_thread = None
            elif kind == "refresh_error":
                self.refresh_thread = None
                pending_error = payload
            elif kind == "error":
                self.stop_event.set()
                pending_error = payload
            elif kind == "stopped":
                stopped = True
        if pending_outcome is not None:
            self._apply_observation(pending_outcome.observation)
            self._show_statistics()
            self.status.set(pending_outcome.message)
            self.advice.set(pending_outcome.advice or "")
            if pending_outcome.advice:
                if not self.advice_label.winfo_manager():
                    self.advice_label.pack(fill="x", pady=(7, 0))
            elif self.advice_label.winfo_manager():
                self.advice_label.pack_forget()
        elif pending_refresh is not None:
            self._apply_observation(pending_refresh)
            if pending_refresh.result.contradictions:
                self.status.set("识别存在矛盾，已停止给出建议")
            elif pending_refresh.frame.used_visible_screen_fallback:
                self.status.set("读取完成；游戏必须保持可见")
            else:
                self.status.set("读取完成")
        if pending_error is not None:
            self.status.set(str(pending_error))
            if self.worker is not None:
                self.mode_text.set("安全停机")
        if stopped:
            self.worker = None
            self._set_idle(self.status.get())
        elif pending_refresh is not None or (pending_error is not None and self.worker is None):
            self._schedule()
        self.root.after(50, self._poll_events)

    def _poll_emergency(self) -> None:
        if emergency_pressed() and self.mode_text.get() != "观察":
            self.pause_automatic()
            self.status.set("F8 紧急停止已触发")
        self.root.after(100, self._poll_emergency)

    def _set_idle(self, message: str) -> None:
        self.mode_text.set("观察")
        self.status.set(message)
        self.start_button.configure(state="normal")
        self.step_button.configure(state="normal")
        self._schedule()

    def _cancel_refresh(self) -> None:
        if self.refresh_job is not None:
            self.root.after_cancel(self.refresh_job)
            self.refresh_job = None

    def _close(self) -> None:
        self.stop_event.set()
        if self.render_job is not None:
            self.root.after_cancel(self.render_job)
        self.root.destroy()


def run_once(output: Path, report_path: Path) -> None:
    observation = observe_window()
    output.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    observation.annotated.save(output)
    report_path.write_text(json.dumps(observation_report(observation), ensure_ascii=False, indent=2), encoding="utf-8")
    print(report_path.resolve())


def main() -> None:
    parser = argparse.ArgumentParser(description="Let's Minesweeper 离线局部观察器")
    parser.add_argument("--once", action="store_true", help="只读取一次并输出文件")
    parser.add_argument("--output", type=Path, default=Path("samples/live-observation.png"))
    parser.add_argument("--report", type=Path, default=Path("samples/live-observation.json"))
    args = parser.parse_args()
    if args.once:
        run_once(args.output, args.report)
        return
    root = tk.Tk()
    ObserverApp(root)
    root.mainloop()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(error, file=sys.stderr)
        raise
