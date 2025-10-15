#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
experimento1.py — FFT sobre audio libre (WAV): magnitud y fase
- Lee WAV (mono/estéreo -> mono opcional)
- Ventanea (Hann por defecto)
- Calcula FFT (opción de zero-padding para densificar la grilla)
- Muestra y guarda figuras: waveform, magnitude(dB), phase(unwrapped)
Requisitos: numpy, scipy, matplotlib
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import get_window, find_peaks

FIGS_DIR = Path("figs")
AUDIO_DIR = Path("audio")

def analyze_audio_fft(
    file: Path,
    t0: float | None = None,
    t1: float | None = None,
    window: str = "hann",
    nfft: int | None = None,
    zp: int = 1,
    scale_db: bool = True,
    mono: bool = True,
    normalize: bool = True,
    show: bool = True,
    save_dir: Path = FIGS_DIR,
    tag: str | None = None,
    annotate_peaks: int = 5,
) -> dict:
    """
    Procesa un WAV y calcula su espectro (magnitud y fase).
    Retorna metadatos y ejes útiles para el informe.

    Parámetros clave:
    - file: ruta del WAV (se recomienda PCM 16-bit).
    - t0, t1: ventana temporal en segundos para analizar (opcional).
    - window: tipo de ventana (hann, hamming, blackman, rect).
    - nfft: tamaño base de FFT; si None usa len(segmento).
    - zp: factor de zero-padding (>=1). No aumenta resolución física; densifica rejilla.
    - scale_db: True -> magnitud en dB (20*log10).
    - mono: True -> si estéreo, promedia canales a mono.
    - normalize: True -> normaliza a [-1, 1] si el WAV viene en enteros.
    - show: True -> abre ventanas matplotlib; False -> solo guarda PNGs.
    - save_dir: directorio para PNGs.
    - tag: sufijo opcional para nombres de archivos (p.ej., nombre corto del audio).
    - annotate_peaks: anotar los N picos más altos en la magnitud (0 para desactivar).
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1) Cargar audio ----
    if not file.exists():
        raise FileNotFoundError(f"No existe el archivo: {file}")
    fs, x = wavfile.read(file)

    # Convertir a float [-1,1] si es entero
    if normalize:
        if np.issubdtype(x.dtype, np.integer):
            max_abs = np.iinfo(x.dtype).max
            x = x.astype(np.float64) / max_abs
        else:
            x = x.astype(np.float64)
    else:
        x = x.astype(np.float64)

    # Estéreo -> mono
    if x.ndim == 2:
        if mono:
            x = x.mean(axis=1)
        else:
            # Si no se fuerza mono, tomar canal 0 por simplicidad
            x = x[:, 0]

    # ---- 2) Selección temporal ----
    total_duration = len(x) / fs
    if t0 is None:
        t0 = 0.0
    if t1 is None or t1 > total_duration:
        t1 = total_duration
    if t1 <= t0:
        raise ValueError("Ventana temporal inválida: asegúrate de que t1 > t0 y dentro de la duración del audio.")

    n0 = int(np.floor(t0 * fs))
    n1 = int(np.floor(t1 * fs))
    x_seg = x[n0:n1]
    L = len(x_seg)
    if L < 2:
        raise ValueError("El segmento seleccionado es demasiado corto para calcular una FFT.")

    # ---- 3) Ventaneado ----
    win = None
    if window.lower() in ("rect", "rectangular", "boxcar", "none", "sin", "ninguna"):
        win = np.ones(L, dtype=np.float64)
        window_used = "rect"
    else:
        try:
            win = get_window(window, L, fftbins=True).astype(np.float64)
            window_used = window
        except Exception:
            # Fallback a Hann
            win = get_window("hann", L, fftbins=True).astype(np.float64)
            window_used = "hann (fallback)"

    xw = x_seg * win

    # ---- 4) Configuración NFFT y zero-padding ----
    # N base: o el L del segmento, o nfft si se indicó y es >= 2
    if nfft is None:
        n_base = L
    else:
        n_base = int(nfft)
        if n_base < 2:
            n_base = L

    # Zero-padding factor
    if zp < 1:
        zp = 1
    n_eff = int(n_base) * int(zp)

    # Si el segmento es más largo que n_base -> recorte conservador
    if L > n_base:
        xw_used = xw[:n_base]
    else:
        # si es más corto, pad a derecha hasta n_base
        pad = n_base - L
        xw_used = np.pad(xw, (0, pad), mode="constant")

    # Zero-padding extra hasta n_eff
    if n_eff > n_base:
        xw_used = np.pad(xw_used, (0, n_eff - n_base), mode="constant")

    # ---- 5) FFT ----
    X = np.fft.rfft(xw_used, n=n_eff)  # rfft -> solo mitad positiva (incluye DC y Nyquist)
    f = np.fft.rfftfreq(n_eff, d=1.0 / fs)  # 0 ... fs/2

    # Nota: resolución física está determinada por n_base (antes del zp):
    delta_f_physical = fs / float(n_base)
    delta_f_grid = fs / float(n_eff)  # más denso si zp>1

    # ---- 6) Magnitud y Fase ----
    mag = np.abs(X)
    eps = 1e-12
    if scale_db:
        mag_plot = 20.0 * np.log10(mag + eps)
        mag_label = "Magnitud (dB)"
    else:
        mag_plot = mag
        mag_label = "Magnitud (lineal)"

    phase = np.unwrap(np.angle(X))

    # ---- 7) Detección de picos (opcional) ----
    peaks_info = []
    if annotate_peaks and annotate_peaks > 0:
        # Ignora el primer bin (DC) para picos (si puede contaminar)
        start_idx = 1 if len(mag_plot) > 1 else 0
        # Usa prominencia para evitar ruido (ajustable)
        pk_idx, _ = find_peaks(mag_plot[start_idx:], prominence=(3.0 if scale_db else None))
        pk_idx = pk_idx + start_idx
        # Ordenar por altura y tomar top-N
        top = sorted(pk_idx, key=lambda i: mag_plot[i], reverse=True)[:annotate_peaks]
        for i in top:
            peaks_info.append((float(f[i]), float(mag_plot[i])))

    # ---- 8) Gráficas ----
    # 8.1 Forma de onda
    t_axis = np.arange(L) / fs + t0
    plt.figure(figsize=(10, 3.2), dpi=120)
    plt.plot(t_axis, x_seg, linewidth=1.0)
    plt.title(f"Forma de onda — {file.name} (fs={fs} Hz, dur={t1-t0:.2f} s)")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Amplitud")
    plt.grid(True, alpha=0.3)
    out_wave = save_dir / (f"waveform_{tag}.png" if tag else "waveform.png")
    plt.tight_layout()
    plt.savefig(out_wave, dpi=300)
    if show:
        plt.show()
    else:
        plt.close()

    # 8.2 Magnitud
    plt.figure(figsize=(10, 3.6), dpi=120)
    plt.plot(f, mag_plot, linewidth=1.0)
    plt.title(
        f"Espectro de magnitud — ventana={window_used}, "
        f"Nbase={n_base}, Δf≈{delta_f_physical:.4f} Hz, zp×{zp}"
    )
    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel(mag_label)
    plt.xlim(0, fs / 2.0)
    plt.grid(True, alpha=0.3)
    # Anotar picos
    if peaks_info:
        for fr, mv in peaks_info:
            plt.plot(fr, mv, "o")
            plt.annotate(f"{fr:.1f} Hz", (fr, mv), textcoords="offset points", xytext=(5, 5), fontsize=8)
    out_mag = save_dir / (f"magnitude_{'db' if scale_db else 'linear'}_{tag}.png" if tag else f"magnitude_{'db' if scale_db else 'linear'}.png")
    plt.tight_layout()
    plt.savefig(out_mag, dpi=300)
    if show:
        plt.show()
    else:
        plt.close()

    # 8.3 Fase
    plt.figure(figsize=(10, 3.6), dpi=120)
    plt.plot(f, phase, linewidth=1.0)
    plt.title("Espectro de fase (desenrollada)")
    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("Fase (rad)")
    plt.xlim(0, fs / 2.0)
    plt.grid(True, alpha=0.3)
    out_phase = save_dir / (f"phase_unwrapped_{tag}.png" if tag else "phase_unwrapped.png")
    plt.tight_layout()
    plt.savefig(out_phase, dpi=300)
    if show:
        plt.show()
    else:
        plt.close()

    # ---- 9) Resumen para consola/retorno ----
    meta = {
        "file": str(file),
        "fs": int(fs),
        "duration_total_s": float(total_duration),
        "segment_s": (float(t0), float(t1)),
        "window": window_used,
        "n_base": int(n_base),
        "n_eff": int(n_eff),
        "delta_f_physical_hz": float(delta_f_physical),
        "delta_f_grid_hz": float(delta_f_grid),
        "scale_db": bool(scale_db),
        "mono": bool(mono),
        "normalize": bool(normalize),
        "saved_waveform": str(out_wave),
        "saved_magnitude": str(out_mag),
        "saved_phase": str(out_phase),
        "top_peaks": [{"f_hz": fr, "mag": mv} for fr, mv in peaks_info],
    }
    # Imprime un resumen “amable”
    print("\n--- RESUMEN FFT AUDIO ---")
    print(f"Archivo      : {file.name}")
    print(f"fs           : {fs} Hz")
    print(f"Duración seg.: {t1 - t0:.3f} s  (total: {total_duration:.3f} s)")
    print(f"Ventana      : {window_used}")
    print(f"N base       : {n_base}  (Δf física ≈ {delta_f_physical:.6f} Hz)")
    print(f"Zero-padding : x{zp}  (grilla Δf ≈ {delta_f_grid:.6f} Hz)")
    print(f"Escala       : {'dB' if scale_db else 'lineal'}")
    if peaks_info:
        print("Picos (top):")
        for fr, mv in peaks_info:
            print(f"  - f={fr:8.2f} Hz, mag={'%8.2f dB'%mv if scale_db else f'{mv:.4g}'}")
    print("Figuras guardadas en:", save_dir.resolve())
    print("-------------------------\n")

    return {
        "meta": meta,
        "f_hz": f,
        "mag": mag,              # magnitud en lineal (útil si luego quieres exportar)
        "mag_plot": mag_plot,    # lo que se graficó (dB o lineal)
        "phase_unwrapped": phase
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Experimento 1: FFT de pista de audio (WAV).")
    p.add_argument("--file", type=str, default=str(AUDIO_DIR / "entrada.wav"),
                   help="Ruta al WAV de entrada (por defecto audio/entrada.wav).")
    p.add_argument("--t0", type=float, default=None, help="Inicio (s) del segmento a analizar.")
    p.add_argument("--t1", type=float, default=None, help="Fin (s) del segmento a analizar.")
    p.add_argument("--window", type=str, default="hann", help="Ventana: hann|hamming|blackman|rect ...")
    p.add_argument("--nfft", type=int, default=None, help="Tamaño base de la FFT. Por defecto: longitud del segmento.")
    p.add_argument("--zp", type=int, default=1, help="Factor de zero-padding (>=1).")
    p.add_argument("--linear", action="store_true", help="Usar escala lineal (por defecto dB).")
    p.add_argument("--stereo-keep", action="store_true", help="Mantener canal 0 si el WAV es estéreo (por defecto: mono).")
    p.add_argument("--no-norm", action="store_true", help="No normalizar la señal a [-1,1].")
    p.add_argument("--save-only", action="store_true", help="Guardar PNGs sin abrir ventanas.")
    p.add_argument("--tag", type=str, default=None, help="Sufijo para nombres de archivo en figs/.")
    p.add_argument("--no-peaks", action="store_true", help="No anotar picos principales.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    FILE = Path(args.file)
    analyze_audio_fft(
        file=FILE,
        t0=args.t0,
        t1=args.t1,
        window=args.window,
        nfft=args.nfft,
        zp=max(1, int(args.zp)),
        scale_db=not args.linear,
        mono=not args.stereo_keep,
        normalize=not args.no_norm,
        show=not args.save_only,
        save_dir=FIGS_DIR,
        tag=args.tag,
        annotate_peaks=0 if args.no_peaks else 5,
    )
