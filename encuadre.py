#!/usr/bin/env python3
"""
encuadre.py (v2) — Piloto de tesis (Protocolo Experimental v2.1)
Estabilidad del encuadre de la cámara respecto a una referencia global.

Qué hace
  Compara la franja superior de cada fotograma (edificios y cielo, que no se mueven) contra UN fotograma de
  referencia, elegido automáticamente como el más representativo del video. Para cada fotograma mide:
    · desplazamiento (dx, dy) respecto a la referencia  -> vibración y deriva de la cámara
    · similitud con la referencia                       -> detecta otro encuadre (zoom/inclinación) o eventos
    · cambio en la franja superior                      -> detecta obstrucciones del lente
  y clasifica cada fotograma:  REF · ENCUADRE_DISTINTO · EVENTO · OBSTRUCCION

Comandos
  estabilidad  Analiza la sesión y genera tablas, gráfico y hoja de revisión.
  hoja         Hoja de contacto de un rango de segundos, para revisar a ojo.

Uso (en la carpeta de pipeline.py):
  python encuadre.py estabilidad sesion_BYPASS_20260928_V01.yaml
  python encuadre.py estabilidad sesion_BYPASS_20260928_V01.yaml --rango-conteo 1830 2420
  python encuadre.py estabilidad sesion_BYPASS_20260928_V01.yaml --marcar
  python encuadre.py hoja        sesion_BYPASS_20260928_V01.yaml --desde 825 --hasta 855 --n 15
"""
import argparse
import json
import sys
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipeline import a_json, barra, cargar_config, guardar_json, guardar_jpg, hms, leer_imagen, log, salir

COLOR_TIPO = {"ENCUADRE_DISTINTO": "#7f7f7f", "EVENTO": "#C0392B", "OBSTRUCCION": "#7B3F98"}


def preparar(gris, ancho, frac_sup):
    h, w = gris.shape
    g = cv2.resize(gris, (ancho, max(1, round(h * ancho / w))), interpolation=cv2.INTER_AREA)
    return g[: max(8, int(g.shape[0] * frac_sup)), :]


def desplazamiento(a, b):
    """Desplazamiento del contenido de b respecto a a (px), y respuesta de la correlación (0-1)."""
    if a.shape != b.shape:
        return 0.0, 0.0, 0.0
    af, bf = a.astype(np.float32), b.astype(np.float32)
    ventana = cv2.createHanningWindow((af.shape[1], af.shape[0]), cv2.CV_32F)
    (dx, dy), resp = cv2.phaseCorrelate(af, bf, ventana)
    return float(dx), float(dy), float(resp)


def tramos(mascara, segundos, hueco=3):
    """Tramos consecutivos de True como (inicio_s, fin_s); une los separados por <= hueco filas."""
    idx = np.where(mascara)[0]
    if len(idx) == 0:
        return []
    grupos, ini, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - prev > hueco + 1:
            grupos.append((ini, prev))
            ini = i
        prev = i
    grupos.append((ini, prev))
    return [(float(segundos[a]), float(segundos[b])) for a, b in grupos]


def miniatura(ruta, ancho, texto):
    img = leer_imagen(ruta, cv2.IMREAD_COLOR)
    if img is None:
        img = np.full((ancho * 2, ancho, 3), 128, np.uint8)
    h, w = img.shape[:2]
    img = cv2.resize(img, (ancho, round(h * ancho / w)), interpolation=cv2.INTER_AREA)
    cv2.rectangle(img, (0, 0), (ancho, 22), (0, 0, 0), -1)
    cv2.putText(img, texto, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def hoja(rutas_textos, ruta_salida, ancho=300, columnas=5):
    minis = [miniatura(r, ancho, t) for r, t in rutas_textos]
    alto = max(m.shape[0] for m in minis)
    filas = int(np.ceil(len(minis) / columnas))
    lienzo = np.full((filas * alto, columnas * ancho, 3), 255, np.uint8)
    for i, m in enumerate(minis):
        y, x = (i // columnas) * alto, (i % columnas) * ancho
        lienzo[y:y + m.shape[0], x:x + m.shape[1]] = m
    guardar_jpg(ruta_salida, lienzo, 85)


# ------------------------------------------------------------------ estabilidad
def cmd_estabilidad(cfg, ancho, frac_sup, resp_min, min_distinto_s, umbral_dif, cambio_max, referencia, rango, ventana_s, marcar):
    ruta_meta = cfg["_sdir"] / "dataset_metadata.csv"
    if not ruta_meta.is_file():
        salir("Falta dataset_metadata.csv. Ejecuta pipeline.py (extraer) primero.")
    meta = pd.read_csv(ruta_meta).sort_values("segundo").reset_index(drop=True)
    meta = meta[meta["segundo"] >= cfg["auditoria"]["segundos_descarte_inicial"]].reset_index(drop=True)
    seg = meta["segundo"].to_numpy(dtype=float)
    ancho_orig = int(meta["ancho"].iloc[0])
    escala = ancho_orig / ancho

    print("📐 Leyendo la franja superior de cada fotograma...")
    tiras = []
    for fn in barra(meta["filename"], desc="Leyendo", unit="img"):
        g = leer_imagen(cfg["_frames"] / fn)
        tiras.append(None if g is None else preparar(g, ancho, frac_sup))
    if any(t is None for t in tiras):
        salir("Hay fotogramas ilegibles; ejecuta 'auditar' y revisa dataset_audit.csv.")

    # referencia: el fotograma más parecido, en promedio, a una muestra de ~40 fotogramas repartidos en el video
    if referencia is not None:
        ref = int(np.argmin(np.abs(seg - referencia)))
    else:
        cand = list(range(0, len(tiras), max(1, len(tiras) // 40)))
        m = np.zeros((len(cand), len(cand)))
        for i in range(len(cand)):
            for j in range(i + 1, len(cand)):
                m[i, j] = m[j, i] = desplazamiento(tiras[cand[i]], tiras[cand[j]])[2]
        ref = cand[int(np.argmax(m.sum(axis=1)))]
    print(f"🎯 Fotograma de referencia: {meta['filename'][ref]} (segundo {seg[ref]:.0f})")

    d = np.array([desplazamiento(tiras[ref], t) for t in tiras])
    dx, dy, resp = d[:, 0], d[:, 1], d[:, 2]

    # clasificación: otro encuadre (similitud baja sostenida) y eventos (caídas breves)
    suav = pd.Series(resp).rolling(15, center=True, min_periods=1).median().to_numpy()
    estado = np.array(["REF"] * len(meta), dtype=object)
    for ini, fin in tramos(suav < resp_min, seg, hueco=5):
        marca = "ENCUADRE_DISTINTO" if (fin - ini + 1) >= min_distinto_s else "EVENTO"
        estado[(seg >= ini) & (seg <= fin)] = marca
    estado[(resp < resp_min) & (estado == "REF")] = "EVENTO"

    # obstrucción: diferencia con la mediana móvil de fotogramas vecinos ya alineados con la referencia
    print("🔎 Buscando obstrucciones del lente...")
    reg = [None] * len(meta)
    for i in np.where(estado == "REF")[0]:
        M = np.float32([[1, 0, -dx[i]], [0, 1, -dy[i]]])
        reg[i] = cv2.warpAffine(tiras[i], M, (tiras[i].shape[1], tiras[i].shape[0]), borderMode=cv2.BORDER_REPLICATE)
    cambio = np.zeros(len(meta))
    for i in barra(np.where(estado == "REF")[0], desc="Obstrucción", unit="img"):
        vecinos = [j for j in np.where((seg >= seg[i] - 90) & (seg <= seg[i] + 90))[0][::10] if reg[j] is not None and j != i]
        if len(vecinos) < 5:
            continue
        fondo = np.median(np.stack([reg[j] for j in vecinos]), axis=0)
        desfase = float(np.clip(np.median(reg[i].astype(np.float32)) - np.median(fondo), -15, 15))
        cambio[i] = float((np.abs(reg[i].astype(np.float32) - fondo - desfase) > umbral_dif).mean())
    estado[(cambio > cambio_max) & (estado == "REF")] = "OBSTRUCCION"

    out = meta[["frame_id", "filename", "segundo"]].copy()
    out["dx_px"], out["dy_px"] = dx * escala, dy * escala  # en píxeles del fotograma original
    out["similitud_ref"], out["cambio_sup"], out["estado"] = resp, cambio, estado
    out.to_csv(cfg["_sdir"] / "encuadre.csv", index=False, encoding="utf-8-sig")

    # intervalos
    filas = []
    for tipo in ("ENCUADRE_DISTINTO", "EVENTO", "OBSTRUCCION"):
        for ini, fin in tramos(estado == tipo, seg, hueco=3):
            fila = meta.loc[meta["segundo"] == ini].iloc[0]
            filas.append({"tipo": tipo, "inicio_s": ini, "fin_s": fin, "duracion_s": fin - ini + 1,
                          "inicio_hms": hms(ini), "hora_reloj": fila.get("hora_reloj", "")})
    inter = pd.DataFrame(filas, columns=["tipo", "inicio_s", "fin_s", "duracion_s", "inicio_hms", "hora_reloj"])
    inter = inter.sort_values("inicio_s").reset_index(drop=True)
    inter.to_csv(cfg["_evid"] / "encuadre_intervalos.csv", index=False, encoding="utf-8-sig")

    # vibración medida en los fotogramas con el mismo encuadre que la referencia
    ok = estado == "REF"
    vib = {}
    for nombre, v in (("dx", dx[ok] * escala), ("dy", dy[ok] * escala)):
        a = np.abs(v - np.median(v))
        vib[nombre] = {"mediana_px": round(float(np.median(v)), 1), "p50_px": round(float(np.percentile(a, 50)), 1),
                       "p90_px": round(float(np.percentile(a, 90)), 1), "p99_px": round(float(np.percentile(a, 99)), 1)}

    # ventanas con mejor encuadre (para el conteo manual y la validación)
    lo, hi = (rango if rango else (seg.min(), seg.max()))
    marcado = (estado != "REF").astype(int)
    acum = np.concatenate([[0], np.cumsum(marcado)])
    ventanas = []
    for ini in np.arange(lo, hi - ventana_s + 1, 10.0):
        i0, i1 = np.searchsorted(seg, ini), np.searchsorted(seg, ini + ventana_s)
        if i1 - i0 >= int(0.93 * ventana_s):
            ventanas.append((int(acum[i1] - acum[i0]), float(ini)))
    elegidas = []
    for malos, ini in sorted(ventanas):
        if all(abs(ini - e[1]) >= ventana_s for e in elegidas):
            elegidas.append((malos, ini))
        if len(elegidas) == 3:
            break
    elegidas = [{"inicio_s": e[1], "fin_s": e[1] + ventana_s, "inicio_hms": hms(e[1]), "fin_hms": hms(e[1] + ventana_s),
                 "fotogramas_no_ref": int(e[0])} for e in sorted(elegidas, key=lambda x: x[1])]

    # gráfico
    fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    t = seg / 60
    ax[0].plot(t, resp, lw=0.7, color="#E69F00")
    ax[0].axhline(resp_min, color="black", ls="--", lw=0.8, label=f"umbral {resp_min:g}")
    ax[0].set_ylabel("Similitud con la referencia")
    ax[0].legend(loc="upper right")
    ax[1].plot(t[ok], dx[ok] * escala, lw=0.7, label="dx", color="#2E74B5")
    ax[1].plot(t[ok], dy[ok] * escala, lw=0.7, label="dy", color="#2CA02C")
    ax[1].set_ylabel("Desplazamiento (px originales)")
    ax[1].legend(loc="upper right")
    ax[2].plot(t, cambio, lw=0.7, color="#7B3F98")
    ax[2].axhline(cambio_max, color="black", ls="--", lw=0.8, label=f"umbral {cambio_max:g}")
    ax[2].set_ylabel("Cambio en franja superior")
    ax[2].set_xlabel("Minutos de video")
    ax[2].legend(loc="upper right")
    for _, r in inter.iterrows():
        for a_ in ax:
            a_.axvspan(r["inicio_s"] / 60, (r["fin_s"] + 1) / 60, color=COLOR_TIPO[r["tipo"]], alpha=0.25)
    for e in elegidas:
        ax[0].axvspan(e["inicio_s"] / 60, e["fin_s"] / 60, ymin=0.0, ymax=0.06, color="#2ca02c", alpha=0.9)
    ax[0].set_title(f"Estabilidad del encuadre — {cfg['video_id']}\n"
                    "gris = otro encuadre · rojo = evento · morado = obstrucción · verde = ventana sugerida", fontsize=10)
    fig.tight_layout()
    fig.savefig(cfg["_evid"] / "encuadre_estabilidad.png", dpi=150)
    plt.close(fig)

    # hoja: referencia + un fotograma de cada intervalo (hasta 14)
    piezas = [(cfg["_frames"] / meta["filename"][ref], f"REF {hms(seg[ref])[3:]}")]
    for _, r in inter.head(14).iterrows():
        medio = int(np.argmin(np.abs(seg - (r["inicio_s"] + r["fin_s"]) / 2)))
        piezas.append((cfg["_frames"] / meta["filename"][medio],
                       f"{r['tipo'][:4]} {hms(r['inicio_s'])[3:]}-{hms(r['fin_s'])[3:]}"))
    hoja(piezas, cfg["_evid"] / "encuadre_hoja_intervalos.jpg")

    n_marc = 0
    if marcar:
        ruta_man = cfg["_sdir"] / "revision_manual.csv"
        previo = pd.read_csv(ruta_man) if ruta_man.is_file() else pd.DataFrame(columns=["filename", "motivo"])
        nuevo = pd.concat([
            pd.DataFrame({"filename": out.loc[out["estado"] == "EVENTO", "filename"], "motivo": "CAMARA_EVENTO"}),
            pd.DataFrame({"filename": out.loc[out["estado"] == "OBSTRUCCION", "filename"], "motivo": "OBSTRUCCION_CAMARA"}),
        ])
        comb = pd.concat([previo, nuevo]).drop_duplicates(subset=["filename", "motivo"])
        n_marc = len(comb) - len(previo)
        comb.to_csv(ruta_man, index=False, encoding="utf-8-sig")

    cuenta = out["estado"].value_counts().to_dict()
    res = {
        "parametros": {"ancho_analisis": ancho, "franja_superior": frac_sup, "similitud_min": resp_min,
                       "min_encuadre_distinto_s": min_distinto_s, "umbral_dif": umbral_dif, "cambio_max": cambio_max},
        "referencia": {"filename": meta["filename"][ref], "segundo": float(seg[ref])},
        "fotogramas_por_estado": {k: int(v) for k, v in cuenta.items()},
        "intervalos_por_tipo": {k: int(v) for k, v in inter["tipo"].value_counts().to_dict().items()},
        "vibracion_en_encuadre_de_referencia": vib, "ventanas_sugeridas": elegidas, "ventana_s": ventana_s,
        "marcados_en_revision_manual": n_marc,
    }
    guardar_json(cfg, "encuadre_resumen.json", res)

    print("=" * 66)
    print(f"📊 Fotogramas por estado: {res['fotogramas_por_estado']}")
    print(f"📳 Vibración respecto a la referencia (px originales, mediana → p90): "
          f"dx {vib['dx']['p50_px']}→{vib['dx']['p90_px']}  |  dy {vib['dy']['p50_px']}→{vib['dy']['p90_px']}")
    print("📍 Intervalos:")
    print(inter.to_string(index=False) if len(inter) else "   (ninguno)")
    print(f"🟢 Ventanas de {ventana_s:g} s sugeridas:", [(e["inicio_hms"], e["fin_hms"], e["fotogramas_no_ref"]) for e in elegidas])
    if marcar:
        print(f"📝 Marcados en revision_manual.csv: {n_marc}. Vuelve a ejecutar 'auditar' y 'particionar' de pipeline.py.")
    print(f"🖼️  Revisa {cfg['_evid'] / 'encuadre_hoja_intervalos.jpg'} y {cfg['_evid'] / 'encuadre_estabilidad.png'}")
    print("=" * 66)
    return {"estados": res["fotogramas_por_estado"], "referencia_s": float(seg[ref])}


# ------------------------------------------------------------------ hoja de revisión
def cmd_hoja(cfg, desde, hasta, n):
    ruta_meta = cfg["_sdir"] / "dataset_metadata.csv"
    if not ruta_meta.is_file():
        salir("Falta dataset_metadata.csv. Ejecuta pipeline.py (extraer) primero.")
    meta = pd.read_csv(ruta_meta)
    sub = meta[(meta["segundo"] >= desde) & (meta["segundo"] <= hasta)].sort_values("segundo")
    if sub.empty:
        salir("No hay fotogramas en ese rango de segundos.")
    pos = np.unique(np.linspace(0, len(sub) - 1, min(n, len(sub))).round().astype(int))
    sel = sub.iloc[pos]
    ruta = cfg["_evid"] / f"hoja_{int(desde)}_{int(hasta)}.jpg"
    hoja([(cfg["_frames"] / r["filename"], f"{r['filename'][6:11]}  {hms(r['segundo'])[3:]}") for _, r in sel.iterrows()], ruta)
    print(f"🖼️  {ruta}  ({len(sel)} fotogramas de {len(sub)})")
    return {"hoja": ruta.name, "fotogramas": len(sel)}


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description="Estabilidad del encuadre (Protocolo v2.1)")
    sub = ap.add_subparsers(dest="comando", required=True)
    s1 = sub.add_parser("estabilidad")
    s1.add_argument("config")
    s1.add_argument("--ancho", type=int, default=360, help="ancho de análisis en px")
    s1.add_argument("--franja-superior", type=float, default=0.45, dest="frac_sup")
    s1.add_argument("--resp-min", type=float, default=0.15, help="similitud mínima con la referencia")
    s1.add_argument("--min-distinto-s", type=float, default=60.0, help="duración mínima para llamarlo 'otro encuadre'")
    s1.add_argument("--umbral-dif", type=float, default=45.0)
    s1.add_argument("--cambio-max", type=float, default=0.30)
    s1.add_argument("--referencia", type=float, default=None, help="segundo del fotograma de referencia (opcional)")
    s1.add_argument("--rango-conteo", type=float, nargs=2, default=None, metavar=("INI", "FIN"),
                    help="segundos entre los que se buscan ventanas de 5 min (por ejemplo, el bloque de prueba)")
    s1.add_argument("--ventana-s", type=float, default=300.0, help="duración de las ventanas sugeridas para el conteo manual")
    s1.add_argument("--marcar", action="store_true", help="agrega eventos y obstrucciones a revision_manual.csv")
    s2 = sub.add_parser("hoja")
    s2.add_argument("config")
    s2.add_argument("--desde", type=float, required=True)
    s2.add_argument("--hasta", type=float, required=True)
    s2.add_argument("--n", type=int, default=15)
    a = ap.parse_args()

    cfg = cargar_config(a.config)
    t0 = time.time()
    log(cfg, f"INICIO encuadre {a.comando} | {' '.join(sys.argv[1:])}")
    if a.comando == "estabilidad":
        r = cmd_estabilidad(cfg, a.ancho, a.frac_sup, a.resp_min, a.min_distinto_s, a.umbral_dif, a.cambio_max,
                            a.referencia, a.rango_conteo, a.ventana_s, a.marcar)
    else:
        r = cmd_hoja(cfg, a.desde, a.hasta, a.n)
    log(cfg, f"FIN encuadre {a.comando} | {json.dumps(r, ensure_ascii=False, default=a_json)}")
    log(cfg, f"TIEMPO {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
