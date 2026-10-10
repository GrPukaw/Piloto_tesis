#!/usr/bin/env python3
"""
pipeline.py — Piloto de tesis (Protocolo Experimental v2.1)
Optimización del flujo vehicular — Óvalo Salida a Cusco, Juliaca

Comandos (cada uno cubre una tarea de la matriz de trazabilidad):
  inventario   T01  Inventario del video: metadatos, hash SHA-256, advertencias, catalogo_videos.csv
  extraer      T01  Extracción a 1 FPS (Sec. 3.3) y creación de dataset_metadata.csv
  auditar      T02  Auditoría de calidad (Tabla 3.2) y creación de dataset_audit.csv
  particionar  T03/T04  Partición anti-fuga por group_id (Sec. 3.2) y dataset_split.csv
  todo         Ejecuta los cuatro en orden

Uso:
  python pipeline.py inventario sesion_BYPASS_20260928_V01.yaml
  python pipeline.py extraer    sesion_BYPASS_20260928_V01.yaml
  python pipeline.py auditar    sesion_BYPASS_20260928_V01.yaml --factor 0.4
  python pipeline.py particionar sesion_BYPASS_20260928_V01.yaml

Dependencias: pip install -r requirements.txt
Todo lo que se ejecuta queda registrado en sesiones/<video_id>/evidencias/.
"""
import argparse
import hashlib
import json
import math
import platform
import shutil
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

VERSION_PIPELINE = "1.0"

DEFAULTS = {
    "hora_inicio": None,
    "condiciones": "",
    "extraccion": {"fps_dataset": 1.0, "calidad_jpg": 95},
    "auditoria": {
        "segundos_descarte_inicial": 5,
        "factor_nitidez": 0.4,
        "umbral_nitidez": None,
        "ancho_analisis": 540,
    },
    "particion": {"rol": "interno", "seed": 42, "bloque_min": 10, "guarda_seg": 60, "prueba_frac": 0.2},
}

ACCIONES = {
    "ARCHIVO_ILEGIBLE": "Regenerar el fotograma",
    "ESTABILIZACION_INICIAL_CAMARA": "Exclusión de la matriz de inferencia",
    "DESENFOQUE": "Exclusión de la matriz de inferencia",
}


# ------------------------------------------------------------------ utilidades
def salir(msg):
    print(f"❌ {msg}")
    sys.exit(1)


def hms(segundos):
    s = int(segundos)
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def deep_merge(base, extra):
    out = dict(base)
    for k, v in (extra or {}).items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def a_json(obj):
    if hasattr(obj, "item"):
        return obj.item()
    return str(obj)


def cargar_config(ruta):
    ruta = Path(ruta).resolve()
    if not ruta.is_file():
        salir(f"No existe el archivo de configuración: {ruta}")
    with open(ruta, encoding="utf-8") as f:
        usuario = yaml.safe_load(f) or {}
    cfg = deep_merge(DEFAULTS, usuario)
    for k in ("sesion", "fecha", "video_num", "video", "ubicacion", "dispositivo"):
        if cfg.get(k) in (None, ""):
            salir(f"Falta '{k}' en {ruta.name}")
    try:
        fecha = datetime.strptime(str(cfg["fecha"]), "%Y-%m-%d")
    except ValueError:
        salir("'fecha' debe tener el formato AAAA-MM-DD (por ejemplo 2026-09-28).")
    h = cfg.get("hora_inicio")
    if isinstance(h, int):  # YAML interpreta 7:30 sin comillas como entero
        h = f"{h // 60:02d}:{h % 60:02d}"
    if h not in (None, ""):
        try:
            datetime.strptime(str(h), "%H:%M")
        except ValueError:
            salir("'hora_inicio' debe tener el formato HH:MM entre comillas (por ejemplo \"07:30\").")
        cfg["hora_inicio"] = str(h)
    else:
        cfg["hora_inicio"] = None

    base = ruta.parent
    cfg["_cfg_path"], cfg["_base"], cfg["_fecha"] = ruta, base, fecha
    cfg["video_id"] = f"{cfg['sesion']}_{fecha:%Y%m%d}_V{int(cfg['video_num']):02d}"
    v = Path(str(cfg["video"]))
    cfg["_video"] = v if v.is_absolute() else base / v
    sd = base / "sesiones" / cfg["video_id"]
    cfg["_sdir"], cfg["_evid"] = sd, sd / "evidencias"
    cfg["_frames"] = sd / f"frames_{float(cfg['extraccion']['fps_dataset']):g}fps"
    cfg["_evid"].mkdir(parents=True, exist_ok=True)
    return cfg


def hora_reloj(cfg, segundo):
    if not cfg["hora_inicio"]:
        return ""
    t0 = datetime.combine(cfg["_fecha"].date(), datetime.strptime(cfg["hora_inicio"], "%H:%M").time())
    return (t0 + timedelta(seconds=float(segundo))).strftime("%H:%M:%S")


def log(cfg, texto):
    with open(cfg["_evid"] / "log.txt", "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {texto}\n")


def guardar_json(cfg, nombre, datos):
    ruta = cfg["_evid"] / nombre
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(datos, f, ensure_ascii=False, indent=2, default=a_json)
    return ruta


def cargar_json(cfg, nombre):
    ruta = cfg["_evid"] / nombre
    if ruta.is_file():
        with open(ruta, encoding="utf-8") as f:
            return json.load(f)
    return None


def registrar_entorno(cfg, comando):
    shutil.copy(cfg["_cfg_path"], cfg["_evid"] / "config_usado.yaml")
    guardar_json(cfg, "entorno.json", {
        "pipeline": VERSION_PIPELINE, "ultimo_comando": comando,
        "fecha_ejecucion": datetime.now().isoformat(timespec="seconds"),
        "python": sys.version.split()[0], "opencv": cv2.__version__, "numpy": np.__version__,
        "pandas": pd.__version__, "pyyaml": yaml.__version__, "matplotlib": matplotlib.__version__,
        "sistema": platform.platform(),
    })


def sha256_archivo(ruta, bloque=8 * 1024 * 1024):
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        while True:
            b = f.read(bloque)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def leer_imagen(ruta, flag=cv2.IMREAD_GRAYSCALE):
    try:
        datos = np.fromfile(str(ruta), dtype=np.uint8)
    except OSError:
        return None
    if datos.size == 0:
        return None
    return cv2.imdecode(datos, flag)


def guardar_jpg(ruta, imagen, calidad=95):
    ok, buf = cv2.imencode(".jpg", imagen, [cv2.IMWRITE_JPEG_QUALITY, int(calidad)])
    if not ok:
        return False
    try:
        buf.tofile(str(ruta))
    except OSError:
        return False
    return True


def barra(it, **kw):
    return tqdm(it, **kw) if tqdm else it


# ------------------------------------------------------------------ T01: inventario
def cmd_inventario(cfg):
    v = cfg["_video"]
    if not v.is_file():
        salir(f"No existe el video: {v}")
    cap = cv2.VideoCapture(str(v))
    if not cap.isOpened():
        salir(f"OpenCV no pudo abrir el video: {v}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ancho, alto = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    cap.release()
    if not fps or math.isnan(fps) or fps <= 0 or n <= 0:
        salir("No se pudieron leer FPS o cantidad de cuadros del video (metadatos inválidos).")

    print("🔐 Calculando SHA-256 del video (puede tardar un poco)...")
    sha = sha256_archivo(v)
    dur = n / fps
    orient = "vertical" if alto > ancho else "horizontal"

    adv = []
    if orient == "vertical":
        adv.append("Video vertical (el protocolo prevé orientación horizontal, Tabla 3.1)")
    if min(ancho, alto) < 1080:
        adv.append("Resolución inferior a 1080p (mínimo del protocolo)")
    if fps < 29.5:
        adv.append("FPS inferior a 30 (mínimo del protocolo)")
    if dur < 7200:
        adv.append(f"Duración {hms(dur)} inferior a las 2 h del protocolo (sesión exploratoria)")
    if not cfg["hora_inicio"]:
        adv.append("hora_inicio no declarada: no se calcula hora_reloj")

    info = {
        "video_id": cfg["video_id"], "fecha": f"{cfg['_fecha']:%Y-%m-%d}", "hora_inicio": cfg["hora_inicio"],
        "ubicacion": cfg["ubicacion"], "dispositivo": cfg["dispositivo"], "condiciones": cfg["condiciones"],
        "archivo": v.name, "sha256": sha, "tamano_mb": round(v.stat().st_size / 1e6, 1),
        "ancho": ancho, "alto": alto, "orientacion": orient, "fps": round(fps, 3), "cuadros": n,
        "duracion_s": round(dur, 1), "duracion_hms": hms(dur), "codec_fourcc": fcc, "advertencias": adv,
    }
    guardar_json(cfg, "inventario.json", info)

    # catálogo acumulado de todos los videos (evidencia de T01: horas de video)
    cat_ruta = cfg["_base"] / "catalogo_videos.csv"
    fila = {k: info[k] for k in ("video_id", "fecha", "hora_inicio", "ubicacion", "dispositivo", "archivo", "sha256",
                                 "ancho", "alto", "orientacion", "fps", "cuadros", "duracion_s", "duracion_hms", "tamano_mb")}
    fila["advertencias"] = " | ".join(adv)
    cat = pd.read_csv(cat_ruta) if cat_ruta.is_file() else pd.DataFrame()
    if not cat.empty:
        cat = cat[cat["video_id"] != cfg["video_id"]]
    cat = pd.concat([cat, pd.DataFrame([fila])], ignore_index=True)
    cat.to_csv(cat_ruta, index=False, encoding="utf-8-sig")

    print("=" * 66)
    print(f"📹 {cfg['video_id']}  |  {v.name}")
    print(f"📐 {ancho}x{alto} ({orient})  |  ⏱️  {fps:.3f} FPS  |  ⌛ {hms(dur)}  |  💾 {info['tamano_mb']} MB")
    print(f"🔑 SHA-256: {sha}")
    for a in adv:
        print(f"⚠️  {a}")
    print(f"🧾 Catálogo: {cat_ruta}  ({cat['duracion_s'].sum() / 3600:.2f} h de video registradas)")
    print("=" * 66)
    return {"video_id": cfg["video_id"], "duracion": hms(dur), "resolucion": f"{ancho}x{alto}", "sha256": sha[:12]}


# ------------------------------------------------------------------ T01: extracción
def sonda_pts(ruta, n=60):
    """True si OpenCV entrega marcas de tiempo (pts) crecientes para este video."""
    cap = cv2.VideoCapture(str(ruta))
    ts = []
    for _ in range(n):
        if not cap.grab():
            break
        ts.append(cap.get(cv2.CAP_PROP_POS_MSEC))
    cap.release()
    return len(ts) >= 10 and ts[-1] > 0 and all(b > a for a, b in zip(ts, ts[1:]))


def cmd_extraer(cfg, sobrescribir=False, tiempo="auto"):
    inv = cargar_json(cfg, "inventario.json")
    if inv is None:
        cmd_inventario(cfg)
        inv = cargar_json(cfg, "inventario.json")
    ext_prev = cargar_json(cfg, "extraccion.json")
    if ext_prev and ext_prev["sha256_video"] != inv["sha256"] and not sobrescribir:
        salir("La carpeta de esta sesión ya contiene fotogramas de OTRO video (hash distinto). "
              "Revisa 'video' o 'video_num' en el YAML, o usa --sobrescribir si es intencional.")

    intervalo = 1.0 / float(cfg["extraccion"]["fps_dataset"])
    calidad = cfg["extraccion"]["calidad_jpg"]
    carpeta = cfg["_frames"]
    carpeta.mkdir(parents=True, exist_ok=True)

    usar_pts = sonda_pts(cfg["_video"]) if tiempo == "auto" else (tiempo == "pts")
    cap = cv2.VideoCapture(str(cfg["_video"]))
    if not cap.isOpened():
        salir(f"OpenCV no pudo abrir el video: {cfg['_video']}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    esperados = math.ceil((inv["cuadros"] / fps) / intervalo)

    print("=" * 66)
    print(f"🎞️  Extrayendo a {float(cfg['extraccion']['fps_dataset']):g} FPS  |  ~{esperados} fotogramas esperados")
    print(f"🕒 Modo de tiempo: {'marcas de tiempo reales (pts)' if usar_pts else 'índice de cuadro / FPS promedio'}")
    print(f"📂 {carpeta}")
    print("=" * 66)

    filas, idx, k, objetivo = [], 0, 0, 0
    omitidos = fallidos = huecos = 0
    t_ini = None
    t0 = time.time()
    pbar = tqdm(total=esperados, unit="frame", desc="Extrayendo") if tqdm else None

    while True:
        if not cap.grab():
            break
        t_s = None
        if usar_pts:
            t_abs = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            t_ini = t_abs if t_ini is None else t_ini
            t_s = t_abs - t_ini
            k_obj = int(math.floor(t_s / intervalo + 1e-9))  # segundo real al que pertenece este cuadro
            listo = k_obj >= k
            if listo:
                huecos += k_obj - k  # segundos sin ningún cuadro (hueco en el video)
                k = k_obj
        else:
            listo = idx >= objetivo
        if listo:
            ok, frame = cap.retrieve()
            if ok:
                nombre = f"frame_{k:05d}.jpg"
                ruta = carpeta / nombre
                if ruta.exists() and not sobrescribir:
                    omitidos += 1
                elif not guardar_jpg(ruta, frame, calidad):
                    fallidos += 1
                    print(f"⚠️ No se pudo escribir {ruta}")
                seg = round(k * intervalo, 3)
                filas.append({
                    "frame_id": f"{cfg['video_id']}_F{k:05d}", "video_id": cfg["video_id"], "filename": nombre,
                    "segundo": seg, "timestamp_hms": hms(seg), "hora_reloj": hora_reloj(cfg, seg),
                    "indice_cuadro_video": idx,
                    "t_video_ms": round(t_s * 1000, 1) if usar_pts else round(cap.get(cv2.CAP_PROP_POS_MSEC), 1),
                    "ancho": frame.shape[1], "alto": frame.shape[0],
                    "resolucion": f"{frame.shape[1]}x{frame.shape[0]}",
                    "fuente": cfg["dispositivo"], "ubicacion": cfg["ubicacion"], "estado": "PENDIENTE",
                })
                k += 1
                objetivo = round(k * intervalo * fps)
                if pbar:
                    pbar.update(1)
            else:
                fallidos += 1
        idx += 1
    cap.release()
    if pbar:
        pbar.close()
    if not filas:
        salir("No se extrajo ningún fotograma.")

    df = pd.DataFrame(filas)
    df.to_csv(cfg["_sdir"] / "dataset_metadata.csv", index=False, encoding="utf-8-sig")
    obsoletos = [cfg["_sdir"] / "dataset_audit.csv", cfg["_sdir"] / "dataset_split.csv"]
    if sobrescribir:
        obsoletos.append(cfg["_sdir"] / "nitidez.csv")
    borrados = [o.name for o in obsoletos if o.exists()]
    for o in obsoletos:
        o.unlink(missing_ok=True)

    # deriva temporal: compara el segundo nominal con la marca de tiempo real del cuadro
    deriva = None
    t = df["t_video_ms"].to_numpy() / 1000.0
    if t.max() > 0 and np.all(np.diff(t) >= 0):
        d = t - df["segundo"].to_numpy()
        deriva = {"max_abs_s": round(float(np.abs(d).max()), 3), "final_s": round(float(d[-1]), 3)}

    seg_total = time.time() - t0
    res = {
        "sha256_video": inv["sha256"], "video": cfg["_video"].name, "fps_dataset": cfg["extraccion"]["fps_dataset"],
        "calidad_jpg": calidad, "esperados": esperados, "extraidos": len(df), "omitidos_existentes": omitidos,
        "fallidos": fallidos, "modo_tiempo": "pts" if usar_pts else "indice", "segundos_sin_cuadros": huecos,
        "deriva_temporal": deriva, "tiempo_s": round(seg_total, 1),
        "carpeta": str(carpeta),
    }
    guardar_json(cfg, "extraccion.json", res)

    print("=" * 66)
    print(f"✅ Extraídos: {len(df)}  (esperados ~{esperados})  |  ⏱️  {seg_total:.0f} s")
    if omitidos:
        print(f"⏭️  Ya existían: {omitidos}")
    if fallidos:
        print(f"⚠️  Fallidos: {fallidos}  ← revisar")
    if abs(len(df) - esperados) > 2:
        print("⚠️  Cantidad distinta de lo esperado: posible video con cuadros corruptos o metadatos inexactos.")
    if huecos:
        print(f"⚠️  {huecos} segundos del video no tienen ningún cuadro (hueco): los nombres de archivo saltan esos segundos.")
    if deriva:
        print(f"🕒 Desvío entre el segundo asignado y la marca de tiempo real: máx {deriva['max_abs_s']} s (final {deriva['final_s']} s)")
        if deriva["max_abs_s"] > 1.5:
            print("⚠️  Desvío > 1.5 s: el video tiene FPS variable y este modo no lo corrige. "
                  "Usa '--tiempo pts' o reconvierte con ffmpeg.")
    if borrados:
        print(f"♻️  Se reinició la sesión: se eliminaron {', '.join(borrados)}. Ejecuta 'auditar' y 'particionar' de nuevo.")
    print(f"🧾 {cfg['_sdir'] / 'dataset_metadata.csv'}")
    print("=" * 66)
    return {"extraidos": len(df), "esperados": esperados, "fallidos": fallidos}


# ------------------------------------------------------------------ T02: auditoría
def medir(ruta, ancho):
    g = leer_imagen(ruta)
    if g is None:
        return np.nan, np.nan
    h, w = g.shape
    if w != ancho:
        g = cv2.resize(g, (ancho, max(1, round(h * ancho / w))), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(g, cv2.CV_64F).var()), float(g.mean())


def hoja_contacto(carpeta, filas, ruta_salida, ancho_mini=300, columnas=5):
    minis = []
    for _, r in filas.iterrows():
        img = leer_imagen(carpeta / r["filename"], cv2.IMREAD_COLOR)
        if img is None:
            img = np.full((400, 300, 3), 128, np.uint8)
        h, w = img.shape[:2]
        img = cv2.resize(img, (ancho_mini, round(h * ancho_mini / w)), interpolation=cv2.INTER_AREA)
        etiqueta = f"{r['filename'][6:11]}  n={r['nitidez']:.0f}  {r['marca']}"
        cv2.rectangle(img, (0, 0), (ancho_mini, 20), (0, 0, 0), -1)
        cv2.putText(img, etiqueta, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        minis.append(img)
    alto_mini = max(m.shape[0] for m in minis)
    filas_n = math.ceil(len(minis) / columnas)
    lienzo = np.full((filas_n * alto_mini, columnas * ancho_mini, 3), 255, np.uint8)
    for i, m in enumerate(minis):
        y, x = (i // columnas) * alto_mini, (i % columnas) * ancho_mini
        lienzo[y:y + m.shape[0], x:x + m.shape[1]] = m
    guardar_jpg(ruta_salida, lienzo, 85)


def cmd_auditar(cfg, factor=None, umbral=None, recalcular=False):
    ruta_meta = cfg["_sdir"] / "dataset_metadata.csv"
    if not ruta_meta.is_file():
        salir("Falta dataset_metadata.csv. Ejecuta primero el comando 'extraer'.")
    a = cfg["auditoria"]
    factor = a["factor_nitidez"] if factor is None else factor
    umbral = a["umbral_nitidez"] if umbral is None else umbral

    meta = pd.read_csv(ruta_meta)
    for c in ("nitidez", "luminancia", "motivo_descarte"):
        if c in meta.columns:
            meta = meta.drop(columns=c)

    # nitidez y luminancia (con caché para poder probar varios umbrales sin recalcular)
    cache_ruta = cfg["_sdir"] / "nitidez.csv"
    cache = pd.read_csv(cache_ruta) if cache_ruta.is_file() and not recalcular else None
    if cache is None or set(meta["filename"]) - set(cache["filename"]):
        print("📏 Midiendo nitidez y luminancia de cada fotograma...")
        vals = [medir(cfg["_frames"] / n, a["ancho_analisis"]) for n in barra(meta["filename"], desc="Midiendo", unit="img")]
        cache = pd.DataFrame({"filename": meta["filename"], "nitidez": [v[0] for v in vals],
                              "luminancia": [v[1] for v in vals]})
        cache.to_csv(cache_ruta, index=False)
    meta = meta.merge(cache, on="filename", how="left")

    motivos = [[] for _ in range(len(meta))]
    ilegible = meta["nitidez"].isna().to_numpy()
    estab = (meta["segundo"] < a["segundos_descarte_inicial"]).to_numpy()
    for i in np.where(ilegible)[0]:
        motivos[i].append("ARCHIVO_ILEGIBLE")
    for i in np.where(estab)[0]:
        motivos[i].append("ESTABILIZACION_INICIAL_CAMARA")

    ref = meta.loc[~ilegible & ~estab, "nitidez"]
    if ref.empty:
        salir("No hay fotogramas legibles para auditar.")
    mediana = float(ref.median())
    umbral_usado = float(umbral) if umbral is not None else float(factor) * mediana
    borroso = (~ilegible) & (~estab) & (meta["nitidez"].to_numpy() < umbral_usado)
    for i in np.where(borroso)[0]:
        motivos[i].append("DESENFOQUE")

    # revisión manual (oclusión severa, sombras): archivo revision_manual.csv con columnas filename,motivo
    manual_ruta = cfg["_sdir"] / "revision_manual.csv"
    n_manual = 0
    if manual_ruta.is_file():
        man = pd.read_csv(manual_ruta)
        if "filename" not in man.columns:
            salir("revision_manual.csv debe tener la columna 'filename' (y opcionalmente 'motivo').")
        pos = {n: i for i, n in enumerate(meta["filename"])}
        for _, r in man.iterrows():
            if r["filename"] not in pos:
                print(f"⚠️  revision_manual.csv: no existe {r['filename']}")
                continue
            m = str(r.get("motivo", "REVISION_MANUAL")) if "motivo" in man.columns else "REVISION_MANUAL"
            if m in ("", "nan"):
                m = "REVISION_MANUAL"
            motivos[pos[r["filename"]]].append(m)
            ACCIONES.setdefault(m, "Exclusión de la matriz de inferencia")
            n_manual += 1

    meta["motivo_descarte"] = [";".join(m) if m else "NINGUNO" for m in motivos]
    meta["estado"] = ["DESCARTADO" if m else "VALIDO" for m in motivos]

    aud = [{"frame_id": r["frame_id"], "video_id": r["video_id"], "filename": r["filename"],
            "motivo_descarte": m, "accion_correctiva": ACCIONES.get(m, "Exclusión de la matriz de inferencia")}
           for (_, r), ms in zip(meta.iterrows(), motivos) for m in ms]
    aud_df = pd.DataFrame(aud, columns=["frame_id", "video_id", "filename", "motivo_descarte", "accion_correctiva"])

    meta.to_csv(ruta_meta, index=False, encoding="utf-8-sig")
    aud_df.to_csv(cfg["_sdir"] / "dataset_audit.csv", index=False, encoding="utf-8-sig")

    # candidatos a revisión manual por luminancia atípica (p2–p98), sin descartar automáticamente
    val = meta[meta["estado"] == "VALIDO"]
    p2, p98 = val["luminancia"].quantile([0.02, 0.98])
    cand = val[(val["luminancia"] < p2) | (val["luminancia"] > p98)][["filename", "segundo", "luminancia", "nitidez"]]
    cand = cand.assign(motivo_sugerido="LUMINANCIA_ATIPICA")
    cand.to_csv(cfg["_evid"] / "candidatos_revision_manual.csv", index=False, encoding="utf-8-sig")

    # evidencias gráficas
    fig, ax = plt.subplots(figsize=(8, 4.5))
    legibles = meta[~ilegible & ~estab]
    bordes = np.histogram_bin_edges(legibles["nitidez"], bins=60)
    ax.hist(legibles.loc[legibles["estado"] == "VALIDO", "nitidez"], bins=bordes, color="#2E74B5", label="Válidos")
    ax.hist(legibles.loc[legibles["estado"] == "DESCARTADO", "nitidez"], bins=bordes, color="#C0392B", label="Descartados")
    ax.axvline(umbral_usado, color="black", ls="--", label=f"Umbral = {umbral_usado:.1f}")
    ax.axvline(mediana, color="gray", ls=":", label=f"Mediana = {mediana:.1f}")
    ax.set_xlabel("Nitidez (varianza del Laplaciano)")
    ax.set_ylabel("Fotogramas")
    ax.set_title(f"Auditoría de desenfoque — {cfg['video_id']}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(cfg["_evid"] / "auditoria_histograma_nitidez.png", dpi=150)
    plt.close(fig)

    orden = legibles.sort_values("nitidez")
    peores = orden.head(10)
    medio = orden.iloc[len(orden) // 2 - 2: len(orden) // 2 + 3]
    muestra = pd.concat([peores.assign(marca=np.where(peores["estado"] == "DESCARTADO", "DESC", "OK")),
                         medio.assign(marca="MEDIANA")])
    hoja_contacto(cfg["_frames"], muestra, cfg["_evid"] / "auditoria_borrosos_vs_mediana.jpg")

    por_motivo = {}
    for ms in motivos:
        for m in ms:
            por_motivo[m] = por_motivo.get(m, 0) + 1
    validos = int((meta["estado"] == "VALIDO").sum())
    res = {
        "total": len(meta), "validos": validos, "descartados": len(meta) - validos,
        "porcentaje_validos": round(100 * validos / len(meta), 2), "descartes_por_motivo": por_motivo,
        "criterio_desenfoque": {"metodo": "varianza del Laplaciano", "ancho_analisis_px": a["ancho_analisis"],
                                "factor": factor if umbral is None else None, "umbral_usado": round(umbral_usado, 2),
                                "mediana_sesion": round(mediana, 2)},
        "nitidez_percentiles": {f"p{p}": round(float(ref.quantile(p / 100)), 2) for p in (1, 5, 25, 50, 75, 95)},
        "revision_manual_aplicada": n_manual,
        "candidatos_luminancia_atipica": len(cand),
    }
    guardar_json(cfg, "auditoria_resumen.json", res)

    print("=" * 66)
    print(f"📏 Mediana de nitidez: {mediana:.1f}  |  🎯 Umbral aplicado: {umbral_usado:.1f}")
    for m, c in por_motivo.items():
        print(f"   · {m}: {c}")
    print(f"📈 Válidos: {validos} de {len(meta)}  ({res['porcentaje_validos']} %)")
    print(f"🔍 Candidatos por luminancia atípica (revisión manual): {len(cand)}")
    print(f"🖼️  Revisa {cfg['_evid'] / 'auditoria_borrosos_vs_mediana.jpg'} antes de aceptar el umbral.")
    print("=" * 66)
    return {"validos": validos, "total": len(meta), "umbral": round(umbral_usado, 1)}


# ------------------------------------------------------------------ T03/T04: partición
def cmd_particionar(cfg):
    ruta_meta = cfg["_sdir"] / "dataset_metadata.csv"
    if not ruta_meta.is_file():
        salir("Falta dataset_metadata.csv. Ejecuta 'extraer' y 'auditar' primero.")
    meta = pd.read_csv(ruta_meta)
    if "estado" not in meta.columns or (meta["estado"] == "PENDIENTE").any():
        salir("La sesión aún no está auditada. Ejecuta 'auditar' antes de particionar.")
    p = cfg["particion"]
    rol, seed = p["rol"], int(p["seed"])
    valido = (meta["estado"] == "VALIDO").to_numpy()
    avisos = []

    if rol == "test_externo":
        # sesión completa reservada para prueba: no participa en entrenamiento ni validación
        grupo = np.full(len(meta), cfg["video_id"], dtype=object)
        bloque = np.zeros(len(meta), dtype=int)
        split = np.where(valido, "test", "descartado").astype(object)
        roles = {cfg["video_id"]: "test"}
        bloques_info = [(cfg["video_id"], 0, float(meta["segundo"].max()), "test")]
        sep_min = None
    elif rol == "interno":
        dur = float(meta["segundo"].max()) + 1.0 / float(cfg["extraccion"]["fps_dataset"])
        tam = float(p["bloque_min"]) * 60.0
        nb = max(1, int(dur // tam))  # el último bloque absorbe el resto
        bloque = np.minimum((meta["segundo"].to_numpy() // tam).astype(int), nb - 1)
        grupo = np.array([f"{cfg['video_id']}_B{b + 1:02d}" for b in bloque], dtype=object)

        rng = np.random.default_rng(seed)
        orden = rng.permutation(nb)
        n_test = max(1, round(float(p["prueba_frac"]) * nb)) if nb >= 2 else 0
        test_b, resto = orden[:n_test], orden[n_test:]
        n_val = max(1, round(float(p["prueba_frac"]) * len(resto))) if len(resto) >= 2 else 0
        val_b, train_b = resto[:n_val], resto[n_val:]
        if nb < 2:
            avisos.append("Un solo bloque: no es posible separar entrenamiento y prueba. Usa una sesión externa de prueba.")
        elif n_val == 0:
            avisos.append("Pocos bloques: no hay validación interna. Reduce bloque_min o usa más videos.")
        rol_bloque = {int(b): "test" for b in test_b} | {int(b): "val" for b in val_b} | {int(b): "train" for b in train_b}

        split = np.array([rol_bloque[int(b)] for b in bloque], dtype=object)
        # franja de guarda alrededor de cada frontera entre bloques
        seg = meta["segundo"].to_numpy()
        for j in range(1, nb):
            frontera = j * tam
            split[np.abs(seg - frontera) < float(p["guarda_seg"]) / 2.0] = "guarda"
        split[~valido] = "descartado"
        roles = {f"{cfg['video_id']}_B{b + 1:02d}": r for b, r in rol_bloque.items()}
        bloques_info = []
        for b in range(nb):
            sel = bloque == b
            bloques_info.append((f"B{b + 1:02d}", float(seg[sel].min()), float(seg[sel].max()) + 1, rol_bloque[b]))

        # separación mínima entre prueba y el resto, medida sobre fotogramas válidos
        t_test, t_otro = seg[split == "test"], seg[(split == "train") | (split == "val")]
        sep_min = float(np.min(np.abs(t_test[:, None] - t_otro[None, :]))) if len(t_test) and len(t_otro) else None
    else:
        salir("particion.rol debe ser 'interno' o 'test_externo'.")

    out = pd.DataFrame({"frame_id": meta["frame_id"], "filename": meta["filename"], "video_id": meta["video_id"],
                        "group_id": grupo, "bloque": bloque + 1, "segundo": meta["segundo"], "split": split})
    out.to_csv(cfg["_sdir"] / "dataset_split.csv", index=False, encoding="utf-8-sig")

    # verificación anti-fuga: cada group_id debe pertenecer a una sola partición
    usados = out[out["split"].isin(["train", "val", "test"])]
    solapados = int((usados.groupby("group_id")["split"].nunique() > 1).sum())
    if solapados:
        salir(f"Solapamiento de group_id entre particiones: {solapados}. Revisa la configuración.")

    # evidencia gráfica: línea de tiempo de la partición
    colores = {"train": "#2E74B5", "val": "#E69F00", "test": "#C0392B"}
    fig, ax = plt.subplots(figsize=(9, 2.2))
    for nombre, ini, fin, r in bloques_info:
        ax.barh(0, (fin - ini) / 60, left=ini / 60, color=colores[r], edgecolor="white")
        ax.text((ini + fin) / 120, 0, f"{nombre}\n{r}", ha="center", va="center", color="white", fontsize=8)
    if rol == "interno":
        for j in range(1, nb):
            g = float(p["guarda_seg"]) / 120.0
            ax.axvspan(j * tam / 60 - g, j * tam / 60 + g, color="black", alpha=0.6)
    ax.set_yticks([])
    ax.set_xlabel("Minutos de video (franjas negras = guarda)")
    ax.set_title(f"Partición por group_id — {cfg['video_id']}")
    fig.tight_layout()
    fig.savefig(cfg["_evid"] / "particion_linea_tiempo.png", dpi=150)
    plt.close(fig)

    conteo = out["split"].value_counts().to_dict()
    res = {
        "rol": rol, "semilla": seed, "bloque_min": p["bloque_min"] if rol == "interno" else None,
        "guarda_seg": p["guarda_seg"] if rol == "interno" else None, "grupos": roles,
        "fotogramas_por_particion": conteo, "solapamiento_de_grupos": solapados,
        "separacion_minima_prueba_s": sep_min, "avisos": avisos,
    }
    guardar_json(cfg, "particion_resumen.json", res)

    print("=" * 66)
    print(f"🧪 Partición ({rol}, semilla {seed})")
    for g, r in roles.items():
        print(f"   · {g}: {r}")
    print("   Fotogramas:", {k: int(v) for k, v in conteo.items()})
    print(f"✅ Solapamiento de grupos: {solapados}")
    if sep_min is not None:
        print(f"↔️  Separación mínima entre prueba y entrenamiento/validación: {sep_min:.0f} s")
    for a_ in avisos:
        print(f"⚠️  {a_}")
    if rol == "interno":
        print("⚠️  Con un solo video, la partición por bloques sobreestima la generalización (Sec. 3.2 del protocolo).")
    print("=" * 66)
    return {"solapamiento": solapados, "fotogramas": {k: int(v) for k, v in conteo.items()}}


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description="Pipeline del piloto de tesis (Protocolo Experimental v2.1)")
    sub = ap.add_subparsers(dest="comando", required=True)
    for nombre in ("inventario", "extraer", "auditar", "particionar", "todo"):
        s = sub.add_parser(nombre)
        s.add_argument("config", help="archivo YAML de la sesión")
        if nombre in ("extraer", "todo"):
            s.add_argument("--sobrescribir", action="store_true", help="reescribe fotogramas existentes")
            s.add_argument("--tiempo", choices=["auto", "pts", "indice"], default="auto",
                           help="auto = marcas de tiempo reales si el video las entrega; indice = método anterior")
        if nombre in ("auditar", "todo"):
            s.add_argument("--factor", type=float, default=None, help="umbral = factor x mediana de nitidez")
            s.add_argument("--umbral", type=float, default=None, help="umbral absoluto de nitidez")
            s.add_argument("--recalcular", action="store_true", help="vuelve a medir la nitidez")
    a = ap.parse_args()

    cfg = cargar_config(a.config)
    registrar_entorno(cfg, a.comando)
    t0 = time.time()
    log(cfg, f"INICIO {a.comando} | {' '.join(sys.argv[1:])}")
    pasos = ["inventario", "extraer", "auditar", "particionar"] if a.comando == "todo" else [a.comando]
    for paso in pasos:
        if paso == "inventario":
            r = cmd_inventario(cfg)
        elif paso == "extraer":
            r = cmd_extraer(cfg, getattr(a, "sobrescribir", False), getattr(a, "tiempo", "auto"))
        elif paso == "auditar":
            r = cmd_auditar(cfg, getattr(a, "factor", None), getattr(a, "umbral", None), getattr(a, "recalcular", False))
        else:
            r = cmd_particionar(cfg)
        log(cfg, f"FIN {paso} | {json.dumps(r, ensure_ascii=False, default=a_json)}")
    log(cfg, f"TIEMPO TOTAL {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
