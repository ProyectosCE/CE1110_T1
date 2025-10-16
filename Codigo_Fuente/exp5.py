#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
exp5.py — Ejemplo de modulación y demodulación FSK usando FFT del punto 2

Este script implementa los módulos principales del sistema FSK:
- Tx: generador de bits, mapeo de frecuencia (f0/f1), oscilador (señal FSK)
- Rx: receptor (WAV), analizador espectral (FFT via analyze_audio_fft),
      detector por pico de frecuencia para decidir entre f0 y f1

Reusa la función analyze_audio_fft() de experimento1.py para estimar la
frecuencia dominante por símbolo.

Salida:
- Guarda el WAV modulado en audio/fsk_modulado.wav
- Guarda figuras en figs/ (forma de onda global y por símbolo opcional)
- Imprime en consola el mapeo símbolo->pico->bit detectado y el BER

Requisitos: numpy, scipy, matplotlib (ya presentes en requirements.txt)
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# Reusar utilidades del punto 2
from experimento1 import analyze_audio_fft, FIGS_DIR, AUDIO_DIR


# -------------------- Utilidades Tx --------------------
def generate_bits(n: int, pattern: str | None = None, seed: int | None = None) -> np.ndarray:
    """Genera una secuencia de bits.
    - pattern: string con '0' y '1' (p.ej. "1011010"). Si se provee, tiene prioridad.
    - n: longitud si se generan aleatorios.
    """
    if pattern is not None:
        bits = np.array([1 if ch == '1' else 0 for ch in pattern.strip()], dtype=int)
        return bits
    rng = np.random.default_rng(seed)
    return rng.integers(low=0, high=2, size=n, dtype=int)


def fsk_modulate(bits: np.ndarray, fs: int, Tb: float, f0: float, f1: float, A: float = 0.9,
                 phase0: float = 0.0, phase1: float = 0.0) -> np.ndarray:
    """Genera señal FSK binaria por concatenación de tonos f0/f1 por símbolo.
    - bits: vector de 0/1
    - fs: Hz, frecuencia de muestreo
    - Tb: s, duración de símbolo
    - f0, f1: Hz, frecuencias para 0 y 1
    - A: amplitud (<=1.0 recomendado)
    """
    Ns = int(round(fs * Tb))
    if Ns <= 1:
        raise ValueError("Tb demasiado pequeño para fs: menos de 2 muestras por símbolo")
    t = np.arange(Ns) / float(fs)
    s_list = []
    for b in bits:
        f = f1 if b == 1 else f0
        ph = phase1 if b == 1 else phase0
        s_list.append(A * np.sin(2 * np.pi * f * t + ph))
    return np.concatenate(s_list, axis=0)


def save_wav(path: Path, fs: int, x: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    x_clip = np.clip(x, -1.0, 1.0)
    wavfile.write(path, fs, (x_clip * 32767).astype(np.int16))


def plot_waveform(signal: np.ndarray, fs: int, out_png: Path, title: str = "Señal FSK (onda)",
                  show: bool = False) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(len(signal)) / float(fs)
    plt.figure(figsize=(10, 3.2), dpi=120)
    plt.plot(t, signal, linewidth=0.9)
    plt.title(title)
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Amplitud")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    if show:
        plt.show()
    else:
        plt.close()


# -------------------- Demodulación Rx (vía analyze_audio_fft) --------------------
def demodulate_fsk_fft(
    wav_path: Path,
    fs: int,
    Tb: float,
    n_symbols: int,
    f0: float,
    f1: float,
    window: str = "hann",
    zp: int = 2,
    save_symbol_figs: bool = True,
    scale_db: bool = True,
) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
    """Demodula leyendo el WAV y usando analyze_audio_fft por símbolo.
    Retorna (bits_detectados, lista de (f_pico, margen_decisión))
    margen_decisión = |pico-f1| - |pico-f0| (negativo -> más cerca de f0, positivo -> más cerca de f1)
    """
    detected = []
    diagnostics = []
    for i in range(n_symbols):
        t0 = i * Tb
        t1 = (i + 1) * Tb
        res = analyze_audio_fft(
            file=wav_path,
            t0=t0,
            t1=t1,
            window=window,
            nfft=None,
            zp=max(1, int(zp)),
            scale_db=scale_db,
            mono=True,
            normalize=True,
            show=False,
            save_dir=FIGS_DIR if save_symbol_figs else FIGS_DIR,  # mismo dir
            tag=f"fsk_bit{i}",
            annotate_peaks=1,
        )

        peaks = res["meta"].get("top_peaks", [])
        if len(peaks) >= 1:
            f_peak = float(peaks[0]["f_hz"]) if isinstance(peaks[0], dict) else float(peaks[0][0])
        else:
            # Fallback: tomar máximo absoluto de la magnitud graficada (ignorando DC)
            f_hz = res["f_hz"]
            mag_plot = res["mag_plot"]
            start = 1 if len(mag_plot) > 1 else 0
            idx = start + int(np.argmax(mag_plot[start:]))
            f_peak = float(f_hz[idx])

        d0 = abs(f_peak - f0)
        d1 = abs(f_peak - f1)
        bit_hat = 1 if d1 < d0 else 0
        detected.append(bit_hat)
        diagnostics.append((f_peak, d1 - d0))

    return np.array(detected, dtype=int), diagnostics


# -------------------- CLI / Main --------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Exp5: FSK mod/demod usando analyze_audio_fft")
    # Tx params
    p.add_argument("--fs", type=int, default=44100, help="Frecuencia de muestreo (Hz)")
    p.add_argument("--Tb", type=float, default=0.1, help="Duración de símbolo (s)")
    p.add_argument("--f0", type=float, default=1000.0, help="Frecuencia para bit 0 (Hz)")
    p.add_argument("--f1", type=float, default=2000.0, help="Frecuencia para bit 1 (Hz)")
    p.add_argument("--A", type=float, default=0.8, help="Amplitud de la señal (<=1)")
    p.add_argument("--bits", type=str, default="1011010",
                   help="Patrón binario explícito (ej '1011010'). Si se pasa, ignora --n-bits.")
    p.add_argument("--n-bits", type=int, default=0, help="Generar N bits aleatorios si --bits='' (vacío)")
    p.add_argument("--seed", type=int, default=None, help="Semilla RNG para bits aleatorios")

    # FFT/demod params
    p.add_argument("--window", type=str, default="hann", help="Ventana para FFT por símbolo")
    p.add_argument("--zp", type=int, default=2, help="Zero-padding factor para FFT (>=1)")
    p.add_argument("--no-db", action="store_true", help="No usar dB (usar magnitud lineal)")
    p.add_argument("--no-symbol-figs", action="store_true", help="No guardar figuras por símbolo")
    p.add_argument("--show", action="store_true", help="Mostrar figuras en pantalla además de guardar")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    # 1) Bits
    pattern = args.bits.strip()
    if pattern == "":
        bits = generate_bits(args.n_bits, pattern=None, seed=args.seed)
    else:
        bits = generate_bits(n=len(pattern), pattern=pattern)
    n_symbols = len(bits)

    # 2) Modulación FSK
    fs = int(args.fs)
    Tb = float(args.Tb)
    f0 = float(args.f0)
    f1 = float(args.f1)
    A = float(args.A)

    signal = fsk_modulate(bits, fs=fs, Tb=Tb, f0=f0, f1=f1, A=A)

    # 3) Guardar WAV
    wav_out = AUDIO_DIR / "fsk_modulado.wav"
    save_wav(wav_out, fs, signal)

    # 4) Graficar señal completa (opcional)
    plot_waveform(signal, fs, FIGS_DIR / "fsk_waveform.png", show=bool(args.show))

    # 5) Demodulación por símbolo usando analyze_audio_fft (punto 2)
    detected, diags = demodulate_fsk_fft(
        wav_path=wav_out,
        fs=fs,
        Tb=Tb,
        n_symbols=n_symbols,
        f0=f0,
        f1=f1,
        window=args.window,
        zp=max(1, int(args.zp)),
        save_symbol_figs=not args.no_symbol_figs,
        scale_db=not args.no_db,
    )

    # 6) Resultados e interpretación
    n_err = int(np.sum(bits != detected))
    ber = n_err / float(n_symbols) if n_symbols > 0 else 0.0

    print("\n--- RESUMEN FSK MOD/DEMOD ---")
    print(f"fs={fs} Hz, Tb={Tb:.4f} s -> Ns/símbolo={int(round(fs*Tb))} muestras, Δf≈{fs/int(round(fs*Tb)):.2f} Hz")
    print(f"f0={f0:.1f} Hz, f1={f1:.1f} Hz, separación={abs(f1-f0):.1f} Hz")
    print("Bits TX     :", ''.join(map(str, bits.tolist())))
    print("Bits detect.:", ''.join(map(str, detected.tolist())))
    print(f"Errores={n_err}/{n_symbols}  (BER={ber:.4f})")
    print("Detalle por símbolo (i: f_pico[Hz] -> decisión):")
    for i, ((f_peak, margin), b_hat) in enumerate(zip(diags, detected.tolist())):
        closer = "f1" if margin > 0 else "f0"
        print(f"  {i:02d}: {f_peak:8.1f} Hz  (más cerca de {closer}) -> {b_hat}")
    print("WAV generado  :", wav_out.resolve())
    print("Figuras en    :", FIGS_DIR.resolve())
    print("-------------------------------\n")


if __name__ == "__main__":
    main()

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import spectrogram

# Parámetros de la señal FSK
f0 = 1000  # Frecuencia para el bit '0'
f1 = 2000  # Frecuencia para el bit '1'
Fs = 10000  # Frecuencia de muestreo
T_symbol = 1  # Duración de un símbolo
T_signal = 7 * T_symbol  # Duración total de la señal (para 7 bits)
t = np.linspace(0, T_signal, int(Fs * T_signal), endpoint=False)

# Datos binarios a transmitir (ejemplo: 1011010)
data = [1, 0, 1, 1, 0, 1, 0]

# Generación de la señal FSK para la secuencia de datos
signal = np.array([])

for bit in data:
    if bit == 1:
        signal = np.append(signal, np.sin(2 * np.pi * f1 * t[:len(t)//len(data)]))  # Frecuencia para 1
    else:
        signal = np.append(signal, np.sin(2 * np.pi * f0 * t[:len(t)//len(data)]))  # Frecuencia para 0

##############################################
# Visualizaciones mejoradas para FSK
##############################################

# 1) Figura 1: Datos, onda completa con límites de símbolo, y espectro global
plt.figure(figsize=(12, 7))

# 1.a) Secuencia de datos binarios como pasos en el tiempo
plt.subplot(3, 1, 1)
x_bits = np.arange(len(data)) * T_symbol
plt.step(x_bits, data, where='post', color='blue')
for k in range(len(data) + 1):
    plt.axvline(k * T_symbol, color='k', alpha=0.2, linewidth=0.8)
plt.title("Secuencia de Datos (pasos)")
plt.xlabel("Tiempo (s)")
plt.ylabel("Bit")
plt.xlim(0, T_signal)
plt.ylim(-0.2, 1.2)
plt.grid(alpha=0.3)

# 1.b) Señal FSK — vista completa con decimación y marcas por símbolo
plt.subplot(3, 1, 2)
ts = np.arange(len(signal)) / Fs
decim = max(1, int(Fs // 2000))  # ~≤2000 pts/seg para visualizar sin saturar
plt.plot(ts[::decim], signal[::decim], color='red', linewidth=0.8)
for k in range(len(data) + 1):
    plt.axvline(k * T_symbol, color='k', alpha=0.2, linewidth=0.8)
plt.title("Señal FSK — vista completa con límites de símbolo")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud")
plt.xlim(0, T_signal)
plt.grid(alpha=0.3)

# 1.c) Espectro de la señal completa (magnitud)
n = len(signal)
signal_fft = fft(signal)
frequencies = fftfreq(n, 1 / Fs)
half = n // 2
plt.subplot(3, 1, 3)
plt.plot(frequencies[:half], np.abs(signal_fft)[:half], linewidth=0.9)
plt.title("Espectro de Frecuencia (señal completa)")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Magnitud")
plt.grid(alpha=0.3)
plt.tight_layout()
# Guardar en carpeta figs sin mostrar en pantalla
from pathlib import Path as _Path  # alias local para evitar conflictos
(_dir := FIGS_DIR).mkdir(parents=True, exist_ok=True)
plt.savefig(_dir / "fsk_demo_overview.png", dpi=300)
plt.close()

# 2) Figura 2: Zoom temporal y espectrograma
K = 2  # símbolos a mostrar en el zoom
plt.figure(figsize=(12, 4.5))

# 2.a) Zoom en los primeros K símbolos
plt.subplot(1, 2, 1)
n_zoom = min(int(K * T_symbol * Fs), len(signal))
tz = ts[:n_zoom]
decim_z = max(1, int(Fs // 4000))
plt.plot(tz[::decim_z], signal[:n_zoom:decim_z], color='#1f77b4', linewidth=1.0)
for k in range(K + 1):
    plt.axvline(k * T_symbol, color='k', alpha=0.3, linewidth=0.8)
plt.title(f"Zoom temporal — primeros {K} símbolos")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud")
plt.xlim(0, K * T_symbol)
plt.grid(alpha=0.3)

# 2.b) Espectrograma para ver el cambio de frecuencia por símbolo
plt.subplot(1, 2, 2)
nperseg = max(256, int(Fs * 0.1))  # ventana ~0.1 s o mayor
noverlap = nperseg // 2
f_spec, t_spec, Sxx = spectrogram(signal, fs=Fs, window='hann', nperseg=nperseg, noverlap=noverlap, mode='magnitude')
Sxx_db = 20.0 * np.log10(Sxx + 1e-12)
plt.pcolormesh(t_spec, f_spec, Sxx_db, shading='gouraud', cmap='magma')
plt.title("Espectrograma (dB)")
plt.xlabel("Tiempo (s)")
plt.ylabel("Frecuencia (Hz)")
plt.ylim(0, min(Fs / 2, max(f0, f1) * 2.5))
cbar = plt.colorbar()
cbar.set_label("Magnitud (dB)")
plt.tight_layout()
(_dir := FIGS_DIR).mkdir(parents=True, exist_ok=True)
plt.savefig(_dir / "fsk_demo_zoom_spectrogram.png", dpi=300)
plt.close()

# Demodulación: Decisión basada en la FFT (detectar las frecuencias f0 y f1)
# Buscar los picos principales en el espectro y decidir entre f0 y f1
peaks_freq = frequencies[np.argmax(np.abs(signal_fft))]

# Decidir si la frecuencia detectada corresponde a f0 o f1
if peaks_freq < (f0 + f1) / 2:
    print("La señal corresponde al bit '0' (frecuencia f0).")
else:
    print("La señal corresponde al bit '1' (frecuencia f1).")
