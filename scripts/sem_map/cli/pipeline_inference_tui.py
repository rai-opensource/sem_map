# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
"""Textual TUI for running segmentation inference.

Single-screen wizard:
    pick input source (MCAP / video / image dir / glob / single image) →
    pick model stack (backbone + upsampler + seg head, all backends configurable) →
    pick output dir and FPS → Run.

Writes three videos into the output dir:
    rgb.mp4              — input frames (resized to network input)
    segmentation.mp4     — colorized predicted class IDs
    traversability.mp4   — per-pixel traversability (viridis colormap)

Run:
    python scripts/sem_map/cli/pipeline_inference_tui.py
"""

import sys
import time
from pathlib import Path
from dataclasses import dataclass

from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, ScrollableContainer
from textual.widgets import (
    Header, Footer, Static, Button, Input, Label,
    ProgressBar, RichLog, Switch, Select,
)
from textual.screen import Screen

# ── Path setup ───────────────────────────────────────────────────────────────
_this = Path(__file__).resolve()
for _p in _this.parents:
    if (_p / "package.xml").exists():
        _scripts = str(_p / "scripts")
        if _scripts not in sys.path:
            sys.path.insert(0, _scripts)
        break


def _import_steps():
    """Lazy import via direct file load (avoids sem_map.__init__)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "pipeline_steps", Path(__file__).resolve().parent / "pipeline_steps.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── State ────────────────────────────────────────────────────────────────────

@dataclass
class State:
    source: str = ""
    topic: str = ""
    every_nth: int = 1
    rotate_180: bool = False
    family: str = "dinov3s"
    backbone_backend: str = "trt"
    upsampler_mode: str = "jafar_full"
    upsampler_backend: str = "trt"
    upsample_partial_factor: int = 16
    seg_layer: str = ""
    seg_backend: str = "trt"
    temperature: float = 0.5
    latent_w: int = 32
    latent_h: int = 20
    output_dir: str = ""
    fps: float = 10.0
    overlay_alpha: float = 0.5


# ── Screen ───────────────────────────────────────────────────────────────────

class InferenceScreen(Screen):

    def __init__(self, state: State):
        super().__init__()
        self.state = state
        self._cancelled = False
        self._families: list[str] = []
        self._seg_layers: list[str] = []

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("[bold]SemMap · Inference[/bold]  "
                     "[dim]MCAP / video / images → rgb + segmentation + traversability[/dim]",
                     id="bar")
        with ScrollableContainer(id="fields"):
            # ── Input source ─────────────────────────────────────────────
            yield Static("[bold cyan]Input source[/bold cyan]")
            with Horizontal():
                with Vertical(classes="field"):
                    yield Label("Source [dim](mcap / video / image dir / glob / single image)[/dim]:")
                    yield Input(id="source", value=self.state.source,
                                placeholder="/path/to/file.mcap | .mp4 | .mov | dir | *.png")
                with Vertical(classes="field"):
                    yield Label("Topic [dim](mcap only, empty=auto-detect)[/dim]:")
                    yield Input(id="topic", value=self.state.topic,
                                placeholder="/camera/image_raw")
            with Horizontal():
                with Vertical(classes="field"):
                    yield Label("Every Nth frame:")
                    yield Input(id="nth", value=str(self.state.every_nth))
                with Vertical(classes="field"):
                    yield Label("Rotate 180° [dim](upside-down camera)[/dim]:")
                    yield Switch(id="rot180", value=self.state.rotate_180)

            # ── Model stack ──────────────────────────────────────────────
            yield Static("\n[bold magenta]Model stack[/bold magenta]")
            with Horizontal():
                with Vertical(classes="field"):
                    yield Label("Family:")
                    yield Select([(self.state.family, self.state.family)],
                                 value=self.state.family,
                                 allow_blank=False, id="family")
                with Vertical(classes="field"):
                    yield Label("Backbone backend:")
                    yield Select([("torch", "torch"), ("trt", "trt")],
                                 value=self.state.backbone_backend,
                                 allow_blank=False, id="bb_backend")
            with Horizontal():
                with Vertical(classes="field"):
                    yield Label("Upsampler mode:")
                    yield Select([("bilinear", "bilinear"),
                                  ("jafar_full", "jafar_full"),
                                  ("jafar_partial", "jafar_partial")],
                                 value=self.state.upsampler_mode,
                                 allow_blank=False, id="up_mode")
                with Vertical(classes="field"):
                    yield Label("Upsampler backend:")
                    yield Select([("torch", "torch"), ("trt", "trt")],
                                 value=self.state.upsampler_backend,
                                 allow_blank=False, id="up_backend")
            with Horizontal():
                with Vertical(classes="field"):
                    yield Label("Upsample partial factor [dim](bilinear & jafar_partial)[/dim]:")
                    yield Select([("2", "2"), ("4", "4"), ("8", "8"), ("16", "16")],
                                 value=str(self.state.upsample_partial_factor),
                                 allow_blank=False, id="up_partial_f")
                with Vertical(classes="field"):
                    yield Label("Temperature:")
                    yield Input(id="temp", value=str(self.state.temperature))
            with Horizontal():
                with Vertical(classes="field"):
                    yield Label("Seg layer:")
                    yield Select([("(pick family first)", "__none__")],
                                 value="__none__", allow_blank=False, id="seg_layer")
                with Vertical(classes="field"):
                    yield Label("Seg backend:")
                    yield Select([("torch", "torch"), ("trt", "trt")],
                                 value=self.state.seg_backend,
                                 allow_blank=False, id="seg_backend")
            with Horizontal():
                with Vertical(classes="field"):
                    yield Label("Latent W:")
                    yield Input(id="lw", value=str(self.state.latent_w))
                with Vertical(classes="field"):
                    yield Label("Latent H:")
                    yield Input(id="lh", value=str(self.state.latent_h))

            # ── Output ───────────────────────────────────────────────────
            yield Static("\n[bold green]Output[/bold green]")
            with Horizontal():
                with Vertical(classes="field"):
                    yield Label("Output dir [dim](empty=sibling of source)[/dim]:")
                    yield Input(id="outdir", value=self.state.output_dir,
                                placeholder="<source-parent>/inference_<seg_layer>")
                with Vertical(classes="field"):
                    yield Label("Output FPS:")
                    yield Input(id="fps", value=str(self.state.fps))
            with Horizontal():
                with Vertical(classes="field"):
                    yield Label("Overlay alpha [dim](0=off, 0<a≤1 writes seg/trav overlays on RGB)[/dim]:")
                    yield Input(id="alpha", value=str(self.state.overlay_alpha))

        yield ProgressBar(id="prog", total=100, show_eta=True)
        yield RichLog(id="log", max_lines=300, wrap=True, markup=True)
        with Horizontal(id="btn"):
            yield Button("▶ Run", id="run", variant="primary")
            yield Button("✕ Cancel", id="cancel", variant="error", disabled=True)
            yield Button("Scan topics", id="scan")
            yield Button("Quit", id="quit")
        yield Footer()

    def on_mount(self):
        self._populate_families()

    # ── Population / refresh helpers ─────────────────────────────────────

    def _populate_families(self):
        steps = _import_steps()
        fams = steps.list_model_families()
        if not fams:
            self._log("[yellow]No families found in models.yaml — using default 'dinov3s'[/yellow]")
            fams = ["dinov3s"]
        self._families = fams
        sel = self.query_one("#family", Select)
        sel.set_options([(f, f) for f in fams])
        chosen = self.state.family if self.state.family in fams else fams[0]
        sel.value = chosen
        self.state.family = chosen
        self._refresh_seg_layers(chosen)

    def _refresh_seg_layers(self, family: str):
        steps = _import_steps()
        layers = steps.list_seg_layers_for_family(family)
        self._seg_layers = layers
        sel = self.query_one("#seg_layer", Select)
        if not layers:
            sel.set_options([("(none configured)", "__none__")])
            sel.value = "__none__"
            self._log(f"[yellow]No seg_layers configured for family '{family}'[/yellow]")
            return
        sel.set_options([(l, l) for l in layers])
        chosen = self.state.seg_layer if self.state.seg_layer in layers else layers[0]
        sel.value = chosen
        self.state.seg_layer = chosen

    # ── Reactive handlers ────────────────────────────────────────────────

    @on(Select.Changed, "#family")
    def _on_family_change(self, e: Select.Changed):
        if e.value and e.value != "__none__":
            self.state.family = str(e.value)
            self._refresh_seg_layers(self.state.family)

    # ── Button handlers ──────────────────────────────────────────────────

    @on(Button.Pressed, "#run")
    def _on_run(self):
        self._cancelled = False
        self.run_inference()

    @on(Button.Pressed, "#cancel")
    def _on_cancel(self):
        self._cancelled = True
        self._buttons_idle()
        self._log("[yellow]Cancelling...[/yellow]")

    @on(Button.Pressed, "#scan")
    def _on_scan(self):
        self.scan_topics()

    @on(Button.Pressed, "#quit")
    def _on_quit(self):
        self.app.exit()

    # ── Helpers ──────────────────────────────────────────────────────────

    def _log(self, msg):
        self.query_one("#log", RichLog).write(msg)

    def _set_input_value(self, selector: str, value: str):
        self.query_one(selector, Input).value = value

    def _buttons_busy(self):
        self.query_one("#run", Button).disabled = True
        self.query_one("#cancel", Button).disabled = False
        self.query_one("#scan", Button).disabled = True
        self.query_one("#quit", Button).disabled = True

    def _buttons_idle(self):
        self.query_one("#run", Button).disabled = False
        self.query_one("#cancel", Button).disabled = True
        self.query_one("#scan", Button).disabled = False
        self.query_one("#quit", Button).disabled = False

    def _progress(self, cur, total):
        bar = self.query_one("#prog", ProgressBar)
        if total > 0:
            bar.update(progress=int(cur / total * 100))
        else:
            # indeterminate-ish: keep bumping
            bar.update(progress=(cur % 100))

    def _read_form(self) -> dict | None:
        """Validate and pack form state. Returns None on error."""
        try:
            src = self.query_one("#source", Input).value.strip()
            topic = self.query_one("#topic", Input).value.strip()
            nth = max(1, int(self.query_one("#nth", Input).value.strip() or "1"))
            rot = self.query_one("#rot180", Switch).value
            family = str(self.query_one("#family", Select).value or "dinov3s")
            bb_be = str(self.query_one("#bb_backend", Select).value or "trt")
            up_mode = str(self.query_one("#up_mode", Select).value or "jafar_full")
            up_be = str(self.query_one("#up_backend", Select).value or "trt")
            upf = int(self.query_one("#up_partial_f", Select).value or "16")
            temp = float(self.query_one("#temp", Input).value.strip() or "0.5")
            seg = str(self.query_one("#seg_layer", Select).value or "")
            seg_be = str(self.query_one("#seg_backend", Select).value or "trt")
            lw = int(self.query_one("#lw", Input).value.strip() or "32")
            lh = int(self.query_one("#lh", Input).value.strip() or "20")
            outdir = self.query_one("#outdir", Input).value.strip()
            fps = float(self.query_one("#fps", Input).value.strip() or "10.0")
            alpha = float(self.query_one("#alpha", Input).value.strip() or "0.0")
        except ValueError as e:
            self._log(f"[red]Form error: {e}[/red]")
            return None

        if not src:
            self._log("[red]Source path is required[/red]"); return None
        if seg in ("", "__none__"):
            self._log("[red]Seg layer is required[/red]"); return None

        if not outdir:
            src_path = Path(src)
            parent = src_path.parent if src_path.is_file() else (
                src_path if src_path.is_dir() else Path.cwd())
            # Prefix the folder with the source file's name (spaces → _) so
            # outputs from different videos don't collide.
            prefix = src_path.stem.replace(" ", "_") + "_" if src_path.is_file() else ""
            outdir = str(parent / f"{prefix}inference_{seg}")

        return {
            "source": src, "topic": topic, "every_nth": nth, "rotate_180": rot,
            "family": family, "backbone_backend": bb_be,
            "upsampler_mode": up_mode, "upsampler_backend": up_be,
            "upsample_partial_factor": upf,
            "seg_layer": seg, "seg_backend": seg_be, "temperature": temp,
            "latent_w": lw, "latent_h": lh,
            "output_dir": outdir, "fps": fps, "overlay_alpha": alpha,
        }

    # ── Actions ──────────────────────────────────────────────────────────

    @work(thread=True)
    def scan_topics(self):
        """Auto-detect MCAP image topics and prefill the topic field."""
        src = self.query_one("#source", Input).value.strip()
        if not src:
            self.app.call_from_thread(self._log, "[red]Enter a source path first[/red]"); return
        if not src.lower().endswith(".mcap"):
            self.app.call_from_thread(self._log,
                "[yellow]Topic scan only applies to .mcap files[/yellow]"); return
        steps = _import_steps()
        try:
            infos = steps.list_mcap_topics(src)
        except Exception as e:
            self.app.call_from_thread(self._log, f"[red]Scan failed: {e}[/red]"); return
        if not infos:
            self.app.call_from_thread(self._log,
                "[yellow]No image topics found in MCAP[/yellow]"); return
        for t in infos:
            self.app.call_from_thread(self._log,
                f"  {t['topic']} [dim]({t['message_count']} msgs, {t['fps']} fps)[/dim]")
        picked = infos[0]["topic"]
        self.app.call_from_thread(self._set_input_value, "#topic", picked)
        self.app.call_from_thread(self._log, f"[green]Selected: {picked}[/green]")

    @work(thread=True)
    def run_inference(self):
        cfg = self.app.call_from_thread(self._read_form)
        if cfg is None:
            return
        # Persist to state for re-runs
        for k, v in cfg.items():
            setattr(self.state, k, v)

        self.app.call_from_thread(self._buttons_busy)
        self.app.call_from_thread(self._log,
            f"[bold]Running inference → {cfg['output_dir']}[/bold]")

        steps = _import_steps()
        t0 = time.time()
        try:
            gen = steps.step_inference(
                cfg["source"], cfg["output_dir"],
                family=cfg["family"],
                backbone_backend=cfg["backbone_backend"],
                upsampler_mode=cfg["upsampler_mode"],
                upsampler_backend=cfg["upsampler_backend"],
                upsample_partial_factor=cfg["upsample_partial_factor"],
                seg_layer=cfg["seg_layer"],
                seg_backend=cfg["seg_backend"],
                temperature=cfg["temperature"],
                latent_w=cfg["latent_w"],
                latent_h=cfg["latent_h"],
                topic=cfg["topic"],
                every_nth=cfg["every_nth"],
                rotate_180=cfg["rotate_180"],
                fps=cfg["fps"],
                overlay_alpha=cfg["overlay_alpha"],
            )
            result = {}
            for p in gen:
                if self._cancelled:
                    self.app.call_from_thread(self._log, "[yellow]Cancelled[/yellow]")
                    break
                self.app.call_from_thread(self._log, p.message)
                self.app.call_from_thread(self._progress, p.current, p.total)
                if p.done:
                    result = p.result
            if result:
                dt = time.time() - t0
                self.app.call_from_thread(self._log,
                    f"[green]Done in {dt:.1f}s · {result.get('num_frames', 0)} frames[/green]")
                self.app.call_from_thread(self._log,
                    f"  rgb:  {result.get('rgb_path', '')}")
                self.app.call_from_thread(self._log,
                    f"  seg:  {result.get('seg_path', '')}")
                self.app.call_from_thread(self._log,
                    f"  trav: {result.get('trav_path', '')}")
                if result.get("seg_overlay_path"):
                    self.app.call_from_thread(self._log,
                        f"  seg overlay:  {result['seg_overlay_path']}")
                if result.get("trav_overlay_path"):
                    self.app.call_from_thread(self._log,
                        f"  trav overlay: {result['trav_overlay_path']}")
        except Exception as e:
            self.app.call_from_thread(self._log, f"[red]Error: {e}[/red]")
        finally:
            self.app.call_from_thread(self._buttons_idle)


# ── App ──────────────────────────────────────────────────────────────────────

class InferenceApp(App):
    TITLE = "SemMap: Run Segmentation Inference"
    CSS = """
    #bar { text-align: center; padding: 1 2; background: $primary-background; }
    #fields { height: auto; max-height: 60%; padding: 0 1; }
    #prog { height: 1; margin: 0; dock: bottom; }
    #log { height: 1fr; min-height: 4; border: solid $accent; margin: 0; }
    #btn { dock: bottom; height: auto; padding: 1 2 1 2; background: $surface-darken-1; }
    #btn Button { margin: 0 2 0 0; min-width: 12; }
    .field { width: 1fr; height: auto; padding: 0 1 0 0; }
    Horizontal { height: auto; }
    Vertical { height: auto; }
    Input { height: 3; margin: 0; }
    Select { height: 3; margin: 0; }
    Label { height: 1; margin: 0; padding: 0; color: $text-muted; }
    Switch { height: auto; margin: 0 1; }
    Static { height: auto; margin: 0; padding: 0; }
    ProgressBar { height: 1; margin: 0; }
    """
    BINDINGS = [("q", "quit", "Quit")]

    def __init__(self):
        super().__init__()
        self.state = State()

    def on_mount(self):
        self.push_screen(InferenceScreen(self.state))


if __name__ == "__main__":
    InferenceApp().run()
