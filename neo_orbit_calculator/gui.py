"""Small desktop interface for the NEO orbit calculator."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk


class OrbitCalculator(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("3.5ST NEO Orbit Calculator")
        self.geometry("900x650")
        self.configure(bg="#0e1b18")
        self._build()

    def _build(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TFrame", background="#0e1b18")
        style.configure("TLabel", background="#0e1b18", foreground="#f4f1e8", font=("DejaVu Sans", 11))
        style.configure("TButton", font=("DejaVu Sans", 11, "bold"))
        style.configure("TCheckbutton", background="#0e1b18", foreground="#f4f1e8")

        frame = ttk.Frame(self, padding=24)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="NASA/JPL-backed NEO orbit calculator", font=("DejaVu Sans", 18, "bold"), foreground="#ffd166").grid(column=0, row=0, columnspan=4, sticky="w", pady=(0, 18))

        self.values: dict[str, tk.StringVar] = {
            "designation": tk.StringVar(value="99942"),
            "start": tk.StringVar(value="2026-01-01"),
            "stop": tk.StringVar(value="2126-01-01"),
            "samples": tk.StringVar(value="401"),
            "area_mass": tk.StringVar(value="0"),
            "a1": tk.StringVar(value="0"),
            "a2": tk.StringVar(value="0"),
            "a3": tk.StringVar(value="0"),
            "output": tk.StringVar(value=str(Path("neo_orbit_calculator/output").resolve())),
        }
        fields = [
            ("Designation", "designation"),
            ("Start, TDB", "start"),
            ("Stop, TDB", "stop"),
            ("Samples", "samples"),
            ("Area / mass [m^2 kg^-1]", "area_mass"),
            ("A1 [au d^-2]", "a1"),
            ("A2 [au d^-2]", "a2"),
            ("A3 [au d^-2]", "a3"),
        ]
        for row, (label, key) in enumerate(fields, start=1):
            ttk.Label(frame, text=label).grid(column=0, row=row, sticky="w", pady=5)
            ttk.Entry(frame, textvariable=self.values[key], width=32).grid(column=1, row=row, columnspan=2, sticky="ew", pady=5)

        ttk.Label(frame, text="Output directory").grid(column=0, row=9, sticky="w", pady=5)
        ttk.Entry(frame, textvariable=self.values["output"]).grid(column=1, row=9, sticky="ew", pady=5)
        ttk.Button(frame, text="Choose", command=self._choose_output).grid(column=2, row=9, padx=(8, 0))

        self.relativity = tk.BooleanVar(value=True)
        self.pr_drag = tk.BooleanVar(value=True)
        self.solar_wind = tk.BooleanVar(value=True)
        ttk.Checkbutton(frame, text="Solar 1PN relativity", variable=self.relativity).grid(column=0, row=10, sticky="w", pady=8)
        ttk.Checkbutton(frame, text="Poynting-Robertson drag", variable=self.pr_drag).grid(column=1, row=10, sticky="w", pady=8)
        ttk.Checkbutton(frame, text="Solar-wind drag", variable=self.solar_wind).grid(column=2, row=10, sticky="w", pady=8)

        ttk.Button(frame, text="Download authoritative SPK", command=lambda: self._run("spk")).grid(column=0, row=11, sticky="ew", pady=(12, 8), padx=(0, 8))
        ttk.Button(frame, text="Fetch Horizons vectors", command=lambda: self._run("vectors")).grid(column=1, row=11, sticky="ew", pady=(12, 8), padx=8)
        ttk.Button(frame, text="Run custom propagation", command=lambda: self._run("propagate")).grid(column=2, row=11, sticky="ew", pady=(12, 8), padx=(8, 0))

        self.output = tk.Text(frame, height=12, bg="#172923", fg="#f4f1e8", insertbackground="#f4f1e8", relief="flat", font=("DejaVu Sans Mono", 10), wrap="word")
        self.output.grid(column=0, row=12, columnspan=4, sticky="nsew", pady=(12, 0))
        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(12, weight=1)

    def _choose_output(self) -> None:
        chosen = filedialog.askdirectory(initialdir=self.values["output"].get())
        if chosen:
            self.values["output"].set(chosen)

    def _command(self, mode: str) -> list[str]:
        command = [
            sys.executable,
            "-m",
            "neo_orbit_calculator.cli",
            mode,
            self.values["designation"].get(),
            "--start",
            self.values["start"].get(),
            "--stop",
            self.values["stop"].get(),
        ]
        if mode == "spk":
            return command + ["--output-dir", self.values["output"].get()]
        command += ["--samples", self.values["samples"].get()]
        if mode == "vectors":
            return command + ["--output", str(Path(self.values["output"].get()) / "horizons_vectors.csv")]
        command += [
            "--output-dir",
            self.values["output"].get(),
            "--kernel-dir",
            str(Path(__file__).resolve().parent / "kernels"),
            "--area-mass",
            self.values["area_mass"].get(),
            "--a1",
            self.values["a1"].get(),
            "--a2",
            self.values["a2"].get(),
            "--a3",
            self.values["a3"].get(),
            "--backend",
            "fortran",
        ]
        if not self.relativity.get():
            command.append("--no-relativity")
        if not self.pr_drag.get():
            command.append("--no-pr")
        if not self.solar_wind.get():
            command.append("--no-solar-wind")
        return command

    def _run(self, mode: str) -> None:
        command = self._command(mode)
        self.output.delete("1.0", "end")
        self.output.insert("end", "Running:\n" + " ".join(command) + "\n\n")

        def worker() -> None:
            result = subprocess.run(
                command,
                cwd=Path(__file__).resolve().parent.parent,
                text=True,
                capture_output=True,
            )
            self.after(0, lambda: self._finish(result))

        threading.Thread(target=worker, daemon=True).start()

    def _finish(self, result: subprocess.CompletedProcess[str]) -> None:
        text = result.stdout if result.returncode == 0 else result.stderr
        try:
            text = json.dumps(json.loads(text), indent=2)
        except json.JSONDecodeError:
            pass
        self.output.insert("end", text)
        if result.returncode != 0:
            messagebox.showerror("Orbit calculation failed", text[-1200:])


if __name__ == "__main__":
    OrbitCalculator().mainloop()
