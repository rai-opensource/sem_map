# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
"""Textual TUI for end-to-end segmentation layer pipeline.

Wizard-style flow: MCAP → Data Selection → LLM Filter → SAM3 → Train → ONNX → TRT

Run:
    python scripts/sem_map/cli/pipeline_train_tui.py
"""

import sys
import time
from pathlib import Path
from dataclasses import dataclass, field

from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, ScrollableContainer
from textual.widgets import (
    Header, Footer, Static, Button, Input, Label,
    ProgressBar, RichLog, Switch,
)
from textual.screen import Screen

# ── Path setup ───────────────────────────────────────────────────────────────
_this = Path(__file__).resolve()


def _repo_root() -> Path:
    """The sem_map repo root (directory containing package.xml)."""
    for _p in _this.parents:
        if (_p / "package.xml").exists():
            return _p
    return Path.cwd()


_scripts = str(_repo_root() / "scripts")
if _scripts not in sys.path:
    sys.path.insert(0, _scripts)


def _import_steps():
    """Lazy import via direct file load (avoids sem_map.__init__)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "pipeline_steps", Path(__file__).resolve().parent / "pipeline_steps.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def list_seg_layer_yamls():
    d = _repo_root() / "config" / "seg_layers"
    return sorted(f.name for f in d.glob("*.yaml")) if d.exists() else []


def load_seg_layer_classes(yaml_name):
    import yaml
    yp = _repo_root() / "config" / "seg_layers" / yaml_name
    if yp.exists():
        with open(yp) as f:
            return sorted(yaml.safe_load(f).get("categories", []), key=lambda c: c["id"])
    return []


# ── Constants ────────────────────────────────────────────────────────────────

STEPS = [
    ("MCAP/Video Extract", "cyan"),
    ("Data Selection", "green"),
    ("LLM Class Filter", "blue"),
    ("SAM3 Label", "yellow"),
    ("Train", "magenta"),
    ("ONNX Export", "blue"),
    ("TRT Build", "red"),
]


@dataclass
class State:
    step: int = 0
    status: list = field(default_factory=lambda: ["pending"] * 7)
    times: list = field(default_factory=lambda: [0.0] * 7)
    t0: float = 0.0
    # Carried data
    mcap_path: str = ""
    mcap_topic: str = ""
    images_dir: str = ""
    dataset_dir: str = ""
    k_value: int = 100
    train_split: str = ""
    val_split: str = ""
    yaml_path: str = ""
    seg_tag: str = ""
    checkpoint_path: str = ""
    onnx_path: str = ""
    engine_path: str = ""
    # LLM class-presence pre-filter
    present_classes_dir: str = ""
    llm_base_url: str = ""
    llm_model: str = ""


# ── Base Step Screen ─────────────────────────────────────────────────────────

class StepScreen(Screen):

    def __init__(self, idx: int, state: State):
        super().__init__()
        self.idx = idx
        self.state = state
        self._cancelled = False

    def compose(self) -> ComposeResult:
        name, color = STEPS[self.idx]
        # Step bar
        dots = []
        for i, (n, c) in enumerate(STEPS):
            if i == self.idx:
                dots.append(f"[bold {c}]● {n}[/bold {c}]")
            elif self.state.status[i] == "done":
                dots.append(f"[{c}]✓[/{c}]")
            elif self.state.status[i] == "skipped":
                dots.append("[dim]⏭[/dim]")
            else:
                dots.append("[dim]○[/dim]")
        yield Header()
        yield Static("  ".join(dots), id="bar")
        # Config fields (scrollable)
        with ScrollableContainer(id="fields"):
            yield from self.compose_fields()
        # Progress + log (fixed, always visible)
        yield ProgressBar(id="prog", total=100, show_eta=True)
        yield RichLog(id="log", max_lines=200, wrap=True, markup=True)
        # Buttons (docked bottom)
        with Horizontal(id="btn"):
            yield Button("◀ Back", id="back", disabled=self.idx == 0)
            yield Button("▶ Run", id="run", variant="primary")
            yield Button("✕ Cancel", id="cancel", variant="error", disabled=True)
            yield Button("Skip ▶", id="skip")
        yield Footer()

    def compose_fields(self) -> ComposeResult:
        yield Static("[dim]No config[/dim]")

    # ── Button handlers ──────────────────────────────────────────────────

    @on(Button.Pressed, "#run")
    def _on_run(self):
        self._cancelled = False
        self._log(f"[bold]Running {STEPS[self.idx][0]}...[/bold]")
        self.run_step()

    @on(Button.Pressed, "#cancel")
    def _on_cancel(self):
        self._cancelled = True
        self._buttons_idle()
        self._log("[yellow]Cancelling...[/yellow]")

    @on(Button.Pressed, "#back")
    def _on_back(self):
        if self.state.step > 0:
            self.state.step -= 1
            self.app.switch_screen(SCREEN_MAP[self.state.step](self.state.step, self.state))

    @on(Button.Pressed, "#skip")
    def _on_skip(self):
        self.state.status[self.idx] = "skipped"
        self._advance()

    # ── Helpers ───────────────────────────────────────────────────────────

    def _log(self, msg):
        self.query_one("#log", RichLog).write(msg)

    def _set_input_value(self, selector: str, value: str):
        self.query_one(selector, Input).value = value

    def _buttons_busy(self):
        self.query_one("#run", Button).disabled = True
        self.query_one("#cancel", Button).disabled = False
        self.query_one("#skip", Button).disabled = True
        self.query_one("#back", Button).disabled = True

    def _buttons_idle(self):
        self.query_one("#run", Button).disabled = False
        self.query_one("#cancel", Button).disabled = True
        self.query_one("#skip", Button).disabled = False
        self.query_one("#back", Button).disabled = (self.idx == 0)

    def _progress(self, cur, total):
        if total > 0:
            self.query_one("#prog", ProgressBar).update(progress=int(cur / total * 100))

    def _advance(self):
        if self.state.step < 6:
            self.state.step += 1
            self.app.switch_screen(SCREEN_MAP[self.state.step](self.state.step, self.state))
        else:
            self.app.switch_screen(DoneScreen(self.state))

    def _run_loop(self, gen, step_idx, on_done=None):
        """Common loop for running a step generator.

        Args:
            gen: Step generator yielding StepProgress.
            step_idx: Index of this step.
            on_done: Optional callback(result_dict) called BEFORE advancing,
                     use to store results in state.
        """
        self.state.status[step_idx] = "running"
        self.app.call_from_thread(self._buttons_busy)
        t0 = time.time()
        result = {}
        try:
            for p in gen:
                if self._cancelled:
                    self.app.call_from_thread(self._log, "[yellow]Cancelled[/yellow]")
                    self.app.call_from_thread(self._buttons_idle)
                    return {}
                self.app.call_from_thread(self._log, p.message)
                self.app.call_from_thread(self._progress, p.current, p.total)
                if p.done:
                    result = p.result
            self.state.times[step_idx] = time.time() - t0
            self.state.status[step_idx] = "done"
            if on_done and result:
                on_done(result)
            self.app.call_from_thread(self._log,
                f"[green]Done in {self.state.times[step_idx]:.1f}s[/green]")
            self.app.call_from_thread(self._buttons_idle)
            self.app.call_from_thread(self._advance)
        except Exception as e:
            self.app.call_from_thread(self._log, f"[red]Error: {e}[/red]")
            self.app.call_from_thread(self._buttons_idle)
        return result

    def run_step(self):
        pass


# ── Step 1: MCAP ────────────────────────────────────────────────────────────

class S1_MCAP(StepScreen):
    def compose_fields(self):
        with Horizontal():
            with Vertical(classes="field"):
                yield Label("MCAP or video file:")
                yield Input(id="mcap", value=self.state.mcap_path,
                            placeholder="/path/to/file.mcap | .mp4 | .mov")
            with Vertical(classes="field"):
                yield Label("Topic [dim](mcap only; empty=auto)[/dim]:")
                yield Input(id="topic", value=self.state.mcap_topic, placeholder="/camera/image_raw")
        with Horizontal():
            with Vertical(classes="field"):
                yield Label("Every Nth frame:")
                yield Input(id="nth", value="1")
            with Vertical(classes="field"):
                yield Label("Output dir [dim](empty=sibling of source)[/dim]:")
                yield Input(id="outdir", value=self.state.images_dir, placeholder="<parent>/images")
        with Horizontal():
            with Vertical(classes="field"):
                yield Label("Rotate 180° [dim](upside-down camera)[/dim]:")
                yield Switch(id="rot180", value=False)
            with Vertical(classes="field"):
                yield Static("")

    @work(thread=True)
    def run_step(self):
        src = self.query_one("#mcap", Input).value.strip()
        topic = self.query_one("#topic", Input).value.strip()
        nth = max(1, int(self.query_one("#nth", Input).value.strip() or "1"))
        out = self.query_one("#outdir", Input).value.strip()
        rot180 = self.query_one("#rot180", Switch).value

        if not src:
            self.app.call_from_thread(self._log, "[red]Source path required[/red]")
            return

        steps = _import_steps()
        is_video = steps.is_video_file(src)

        if not is_video and not topic:
            self.app.call_from_thread(self._log, "Scanning for image topics...")
            try:
                infos = steps.list_mcap_topics(src)
                if infos:
                    for t in infos:
                        self.app.call_from_thread(self._log,
                            f"  {t['topic']} [dim]({t['message_count']} msgs, {t['fps']} fps)[/dim]")
                    picked = infos[0]["topic"]
                    self.app.call_from_thread(self._set_input_value, "#topic", picked)
                    self.app.call_from_thread(self._log,
                        f"[green]Selected: {picked}. Click Run again to extract.[/green]")
                else:
                    self.app.call_from_thread(self._log, "[yellow]No image topics found[/yellow]")
            except Exception as e:
                self.app.call_from_thread(self._log, f"[red]{e}[/red]")
            self.app.call_from_thread(self._buttons_idle)
            return

        if not out:
            src_dir = Path(src).resolve().parent
            # images/ sits directly next to the source file
            out = str(src_dir / "images")

        self.state.mcap_path = src
        self.state.mcap_topic = "" if is_video else topic

        def _save(r):
            self.state.images_dir = r.get("output_dir", out)
            self.state.dataset_dir = str(Path(out).parent)

        if is_video:
            self.app.call_from_thread(self._log,
                f"[cyan]Detected video file ({Path(src).suffix}), using OpenCV decoder[/cyan]")
            gen = steps.step_video_extract(src, out, skip=nth - 1, rotate_180=rot180)
        else:
            gen = steps.step_mcap_extract(src, topic, out, skip=nth - 1, rotate_180=rot180)
        self._run_loop(gen, 0, on_done=_save)


# ── Step 2: Data Selection ──────────────────────────────────────────────────

class S2_Select(StepScreen):
    def compose_fields(self):
        default_dir = self.state.images_dir or (
            f"{self.state.dataset_dir}/images" if self.state.dataset_dir else "")
        with Horizontal():
            with Vertical(classes="field"):
                yield Label("Images dir:")
                yield Input(id="imgdir", value=default_dir)
            with Vertical(classes="field"):
                yield Label("K (representative images):")
                yield Input(id="k", value=str(self.state.k_value))
        with Horizontal():
            with Vertical(classes="field"):
                yield Label("Split ratio (train):")
                yield Input(id="ratio", value="0.8")
            with Vertical(classes="field"):
                yield Static("")

    @work(thread=True)
    def run_step(self):
        d = self.query_one("#imgdir", Input).value.strip()
        k = int(self.query_one("#k", Input).value.strip() or "100")
        r = float(self.query_one("#ratio", Input).value.strip() or "0.8")
        if not d:
            self.app.call_from_thread(self._log, "[red]Images dir required[/red]"); return
        self.state.images_dir = d
        self.state.dataset_dir = str(Path(d).parent)
        self.state.k_value = k
        steps = _import_steps()
        def _save(res):
            self.state.dataset_dir = res.get("dataset_dir", self.state.dataset_dir)
            self.state.train_split = res.get("train_split", "")
            self.state.val_split = res.get("val_split", "")
        self._run_loop(steps.step_faiss_select(d, k=k, split_ratio=r), 1, on_done=_save)


# ── Step 3: LLM Class Filter ────────────────────────────────────────────────

class S3_LLMFilter(StepScreen):
    def compose_fields(self):
        with Horizontal():
            with Vertical(classes="field"):
                yield Label("Dataset dir:")
                yield Input(id="ddir", value=self.state.dataset_dir)
            with Vertical(classes="field"):
                yield Label("Seg YAML:")
                yield Input(id="yaml", value=self.state.yaml_path or "coarse_3.yaml")
        with Horizontal():
            with Vertical(classes="field"):
                yield Label("LLM model [dim](or env SEM_MAP_LLM_MODEL)[/dim]:")
                yield Input(id="model", value=self.state.llm_model,
                            placeholder="gemini-2.5-flash")
            with Vertical(classes="field"):
                yield Label("Base URL [dim](or env SEM_MAP_LLM_BASE_URL)[/dim]:")
                yield Input(id="base", value=self.state.llm_base_url,
                            placeholder="https://generativelanguage.googleapis.com/v1beta/openai/")
        with Horizontal():
            with Vertical(classes="field"):
                yield Label("Overwrite cached sidecars:")
                yield Switch(id="ow", value=False)
            with Vertical(classes="field"):
                yield Static("[dim]API key from SEM_MAP_LLM_API_KEY / OPENAI_API_KEY[/dim]")
        default_splits = ",".join(s for s in (self.state.train_split, self.state.val_split) if s)
        with Horizontal():
            with Vertical(classes="field"):
                yield Label("Splits [dim](comma-sep, empty=all *_images.txt or images/)[/dim]:")
                yield Input(id="splits", value=default_splits,
                            placeholder="train_kmeans_k400,val_kmeans_k400")
            with Vertical(classes="field"):
                yield Static("")

    @work(thread=True)
    def run_step(self):
        d = self.query_one("#ddir", Input).value.strip()
        y = self.query_one("#yaml", Input).value.strip()
        model = self.query_one("#model", Input).value.strip() or None
        base = self.query_one("#base", Input).value.strip() or None
        ow = self.query_one("#ow", Switch).value
        splits_raw = self.query_one("#splits", Input).value.strip()
        splits = [s.strip() for s in splits_raw.split(",") if s.strip()] or None
        if not d or not y:
            self.app.call_from_thread(self._log, "[red]Dataset dir and YAML required[/red]"); return
        self.state.dataset_dir = d
        self.state.yaml_path = y
        self.state.llm_model = model or ""
        self.state.llm_base_url = base or ""

        steps = _import_steps()

        def _save(r):
            self.state.present_classes_dir = r.get("present_classes_dir", "")

        self._run_loop(
            steps.step_llm_class_filter(
                d, y, llm_base_url=base, llm_model=model, overwrite=ow,
                splits=splits),
            2, on_done=_save)


# ── Step 4: SAM3 Label ──────────────────────────────────────────────────────

class S4_SAM3(StepScreen):
    def compose_fields(self):
        with Horizontal():
            with Vertical(classes="field"):
                yield Label("Dataset dir:")
                yield Input(id="ddir", value=self.state.dataset_dir)
            with Vertical(classes="field"):
                yield Label("Seg YAML:")
                yield Input(id="yaml", value=self.state.yaml_path or "coarse_3.yaml")
        with Horizontal():
            with Vertical(classes="field"):
                yield Label("Threshold:")
                yield Input(id="thr", value="0.5")
            with Vertical(classes="field"):
                yield Label("Presence sidecars dir [dim](optional)[/dim]:")
                yield Input(id="pcd", value=self.state.present_classes_dir,
                            placeholder="<dataset>/llm_present_classes")
        default_splits = ",".join(s for s in (self.state.train_split, self.state.val_split) if s)
        with Horizontal():
            with Vertical(classes="field"):
                yield Label("Splits [dim](comma-sep, empty=all *_images.txt or images/)[/dim]:")
                yield Input(id="splits", value=default_splits,
                            placeholder="train_kmeans_k400,val_kmeans_k400")
            with Vertical(classes="field"):
                yield Static("")

    @work(thread=True)
    def run_step(self):
        d = self.query_one("#ddir", Input).value.strip()
        y = self.query_one("#yaml", Input).value.strip()
        t = float(self.query_one("#thr", Input).value.strip() or "0.5")
        pcd = self.query_one("#pcd", Input).value.strip() or None
        splits_raw = self.query_one("#splits", Input).value.strip()
        splits = [s.strip() for s in splits_raw.split(",") if s.strip()] or None
        if not d or not y:
            self.app.call_from_thread(self._log, "[red]Dataset dir and YAML required[/red]"); return
        self.state.dataset_dir = d
        self.state.yaml_path = y
        if pcd:
            self.state.present_classes_dir = pcd

        classes = load_seg_layer_classes(y)
        non_bg = [c for c in classes if c["id"] != 0]
        prompts = {c["id"]: c["concept"] for c in non_bg if c.get("concept", "").strip()}
        self.app.call_from_thread(self._log, f"Prompts: {prompts}")
        if pcd:
            self.app.call_from_thread(self._log, f"[dim]Filtering with sidecars at {pcd}[/dim]")

        steps = _import_steps()
        self._run_loop(
            steps.step_sam3_label(
                d, y, prompts=prompts, threshold=t, present_classes_dir=pcd,
                splits=splits),
            3)


# ── Step 5: Train ───────────────────────────────────────────────────────────

class S5_Train(StepScreen):
    def compose_fields(self):
        with Horizontal():
            with Vertical(classes="field"):
                yield Label("Dataset dir:")
                yield Input(id="ddir", value=self.state.dataset_dir)
            with Vertical(classes="field"):
                yield Label("Seg config:")
                yield Input(id="cfg", value=self.state.yaml_path)
        with Horizontal():
            with Vertical(classes="field"):
                yield Label("Seg tag:")
                yield Input(id="tag", value=self.state.seg_tag, placeholder="factory_v1")
            with Vertical(classes="field"):
                yield Label("Epochs:")
                yield Input(id="ep", value="50")
        with Horizontal():
            with Vertical(classes="field"):
                yield Label("LR:")
                yield Input(id="lr", value="5e-4")
            with Vertical(classes="field"):
                yield Label("Batch size:")
                yield Input(id="bs", value="4")
        with Horizontal():
            with Vertical(classes="field"):
                yield Label("Early stop patience:")
                yield Input(id="pat", value="5")
            with Vertical(classes="field"):
                yield Static("")
        default_train_split = self.state.train_split or (
            f"train_kmeans_k{self.state.k_value}" if self.state.k_value else "train")
        default_val_split = self.state.val_split or (
            f"val_kmeans_k{self.state.k_value}" if self.state.k_value else "val")
        with Horizontal():
            with Vertical(classes="field"):
                yield Label("Train split [dim](without _images.txt)[/dim]:")
                yield Input(id="train_split", value=default_train_split)
            with Vertical(classes="field"):
                yield Label("Val split [dim](without _images.txt)[/dim]:")
                yield Input(id="val_split", value=default_val_split)
        with Horizontal():
            with Vertical(classes="field"):
                yield Label("Upsampler [dim](jafar | bilinear)[/dim]:")
                yield Input(id="upsampler", value="jafar")
            with Vertical(classes="field"):
                yield Static("")
        with Horizontal(id="wb_row"):
            yield Label("  Wandb logging: ")
            yield Switch(id="wb", value=False)
        with Horizontal():
            with Vertical(classes="field"):
                yield Label("Wandb entity:")
                yield Input(id="wb_e", value="", placeholder="default account", disabled=True)
            with Vertical(classes="field"):
                yield Label("Wandb project:")
                yield Input(id="wb_p", value="sem_map_seg", disabled=True)
        with Horizontal():
            with Vertical(classes="field"):
                yield Label("Wandb group:")
                yield Input(id="wb_g", value="", placeholder="optional", disabled=True)
            with Vertical(classes="field"):
                yield Static("")

    @on(Switch.Changed, "#wb")
    def _wb_toggle(self, e: Switch.Changed):
        for wid in ["#wb_e", "#wb_p", "#wb_g"]:
            self.query_one(wid, Input).disabled = not e.value

    @work(thread=True)
    def run_step(self):
        d = self.query_one("#ddir", Input).value.strip()
        tag = self.query_one("#tag", Input).value.strip()
        cfg = self.query_one("#cfg", Input).value.strip()
        ep = int(self.query_one("#ep", Input).value.strip() or "50")
        lr = float(self.query_one("#lr", Input).value.strip() or "5e-4")
        bs = int(self.query_one("#bs", Input).value.strip() or "4")
        pat = int(self.query_one("#pat", Input).value.strip() or "5")
        train_split = self.query_one("#train_split", Input).value.strip() or "train"
        val_split = self.query_one("#val_split", Input).value.strip() or "val"
        upsampler = self.query_one("#upsampler", Input).value.strip() or "jafar"
        if upsampler not in ("jafar", "bilinear"):
            self.app.call_from_thread(self._log,
                f"[red]Invalid upsampler '{upsampler}' (use jafar or bilinear)[/red]"); return
        wb = self.query_one("#wb", Switch).value
        wb_e = self.query_one("#wb_e", Input).value.strip() or None
        wb_p = self.query_one("#wb_p", Input).value.strip()
        wb_g = self.query_one("#wb_g", Input).value.strip() or None

        if not d or not tag or not cfg:
            self.app.call_from_thread(self._log, "[red]All fields required[/red]"); return
        self.state.dataset_dir = d
        self.state.seg_tag = tag
        self.state.yaml_path = cfg
        self.state.train_split = train_split
        self.state.val_split = val_split

        steps = _import_steps()
        def _save(r):
            self.state.checkpoint_path = r.get("checkpoint_path", "")
        self._run_loop(
            steps.step_train(d, tag, cfg, num_epochs=ep, lr=lr, batch_size=bs,
                             early_stopping_patience=pat, use_wandb=wb,
                             wandb_entity=wb_e, wandb_project=wb_p, wandb_group=wb_g,
                             train_split=train_split, val_split=val_split,
                             upsampler_type=upsampler), 4, on_done=_save)


# ── Step 6: ONNX ────────────────────────────────────────────────────────────

class S6_ONNX(StepScreen):
    def compose_fields(self):
        with Horizontal():
            with Vertical(classes="field"):
                yield Label("Checkpoint:")
                yield Input(id="ckpt", value=self.state.checkpoint_path)
            with Vertical(classes="field"):
                yield Label("Seg config:")
                yield Input(id="cfg", value=self.state.yaml_path)
        with Horizontal():
            with Vertical(classes="field"):
                yield Label("Seg tag:")
                yield Input(id="tag", value=self.state.seg_tag)
            with Vertical(classes="field"):
                yield Static("")

    @work(thread=True)
    def run_step(self):
        ckpt = self.query_one("#ckpt", Input).value.strip()
        cfg = self.query_one("#cfg", Input).value.strip()
        tag = self.query_one("#tag", Input).value.strip()
        if not ckpt or not cfg:
            self.app.call_from_thread(self._log, "[red]Checkpoint and config required[/red]"); return
        self.state.checkpoint_path = ckpt
        self.state.seg_tag = tag

        steps = _import_steps()
        def _save(r):
            self.state.onnx_path = r.get("onnx_path", "")
        self._run_loop(steps.step_onnx_export(ckpt, cfg, seg_tag=tag), 5, on_done=_save)


# ── Step 7: TRT ─────────────────────────────────────────────────────────────

class S7_TRT(StepScreen):
    def compose_fields(self):
        with Horizontal():
            with Vertical(classes="field"):
                yield Label("ONNX:")
                yield Input(id="onnx", value=self.state.onnx_path)
            with Vertical(classes="field"):
                yield Label("Tag:")
                yield Input(id="tag", value=self.state.seg_tag)
        with Horizontal():
            with Vertical(classes="field"):
                yield Label("Workspace GB:")
                yield Input(id="ws", value="4.0")
            with Vertical(classes="field"):
                yield Static("")

    @work(thread=True)
    def run_step(self):
        onnx = self.query_one("#onnx", Input).value.strip()
        tag = self.query_one("#tag", Input).value.strip()
        ws = float(self.query_one("#ws", Input).value.strip() or "4.0")
        if not onnx:
            self.app.call_from_thread(self._log, "[red]ONNX path required[/red]"); return
        self.state.onnx_path = onnx
        self.state.seg_tag = tag

        steps = _import_steps()
        def _save(r):
            self.state.engine_path = r.get("engine_path", "")
        self._run_loop(steps.step_trt_build(onnx, seg_tag=tag, fp16=True, workspace_gb=ws), 6, on_done=_save)


# ── Summary Screen ──────────────────────────────────────────────────────────

class DoneScreen(Screen):
    def __init__(self, state: State):
        super().__init__()
        self.state = state

    def compose(self):
        yield Header()
        with ScrollableContainer(id="main"):
            yield Static("[bold green]Pipeline Complete[/bold green]", id="bar")
            total = sum(self.state.times)
            rows = []
            for i, (name, _) in enumerate(STEPS):
                s = self.state.status[i]
                t = self.state.times[i]
                icon = {"done": "✅", "skipped": "⏭", "pending": "○"}.get(s, "?")
                rows.append(f" {icon} {name:<16} {t:>6.1f}s  {s}")
            rows.append(f"{'─' * 36}")
            rows.append(f"   Total           {total:>6.1f}s")
            yield Static("\n".join(rows))

            arts = []
            if self.state.checkpoint_path:
                arts.append(f"  torch: {self.state.checkpoint_path}")
            if self.state.onnx_path:
                arts.append(f"  onnx:  {self.state.onnx_path}")
            if self.state.engine_path:
                arts.append(f"  trt:   {self.state.engine_path}")
            if arts:
                yield Static("[bold]Artifacts:[/bold]\n" + "\n".join(arts))
            yield Button("Quit", id="quit", variant="primary")
        yield Footer()

    @on(Button.Pressed, "#quit")
    def _quit(self):
        self.app.exit()


# ── Registry ─────────────────────────────────────────────────────────────────

SCREEN_MAP = [S1_MCAP, S2_Select, S3_LLMFilter, S4_SAM3, S5_Train, S6_ONNX, S7_TRT]


# ── App ──────────────────────────────────────────────────────────────────────

class PipelineApp(App):
    TITLE = "SemMap: Train Segmentation Layer"
    CSS = """
    #bar { text-align: center; padding: 1 2; background: $primary-background; }
    #fields { height: auto; max-height: 50%; padding: 0 1; }
    #prog { height: 1; margin: 0; dock: bottom; }
    #log { height: 1fr; min-height: 4; border: solid $accent; margin: 0; }
    #btn { dock: bottom; height: auto; padding: 1 2 1 2; background: $surface-darken-1; }
    #btn Button { margin: 0 2 0 0; min-width: 12; }
    .field { width: 1fr; height: auto; padding: 0 1 0 0; }
    .field-sm { width: 1fr; height: auto; padding: 0 1 0 0; }
    Horizontal { height: auto; }
    Vertical { height: auto; }
    Input { height: 3; margin: 0; }
    Label { height: 1; margin: 0; padding: 0; color: $text-muted; }
    Switch { height: auto; margin: 0 1; }
    #wb_row { height: auto; padding: 1 0; }
    Static { height: auto; margin: 0; padding: 0; }
    ProgressBar { height: 1; margin: 0; }
    """
    BINDINGS = [("q", "quit", "Quit")]

    def __init__(self):
        super().__init__()
        self.state = State(t0=time.time())

    def on_mount(self):
        self.push_screen(SCREEN_MAP[0](0, self.state))


if __name__ == "__main__":
    PipelineApp().run()
