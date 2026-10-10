#!/usr/bin/env python3
"""
percepcion.py — Piloto de tesis (Protocolo Experimental v2.1)
Etapa de percepción: muestra de etiquetado (T05) y línea base de YOLOv8 (T06a)

Comandos:
  muestra      T05   Selecciona fotogramas para etiquetar, respetando la partición (dataset_split.csv)
  linea_base   T06a  Corre YOLOv8 preentrenado (COCO, sin ajuste) sobre una muestra y compara resoluciones
  preetiquetar T05   Genera preetiquetas y los zips de importación para CVAT (formato YOLO 1.1)

Uso (en la misma carpeta que pipeline.py):
  python percepcion.py muestra    sesion_BYPASS_20260928_V01.yaml --n 200
  python percepcion.py linea_base sesion_BYPASS_20260928_V01.yaml --n 60 --imgsz 640 1280
  python percepcion.py preetiquetar sesion_BYPASS_20260928_V01.yaml --imgsz 1280 --conf 0.30

Dependencias: pip install ultralytics   (instala PyTorch; solo hace falta para linea_base)
"""
import argparse
import json
import shutil
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipeline import a_json, cargar_config, guardar_json, guardar_jpg, leer_imagen, log, salir

# Clases COCO relacionadas con vehículos (id -> nombre)
COCO_VEHICULOS = {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
COLORES = {1: (0, 165, 255), 2: (255, 120, 0), 3: (0, 0, 230), 5: (0, 180, 0), 7: (200, 0, 200)}  # BGR


def muestreo_sistematico(indices, n, rng):
    """n elementos repartidos de forma pareja en la lista ordenada, con inicio aleatorio (semilla fija)."""
    m = len(indices)
    if n >= m:
        return list(indices)
    paso = m / n
    inicio = rng.uniform(0, paso)
    pos = sorted({min(m - 1, int(inicio + i * paso)) for i in range(n)})
    return [indices[p] for p in pos]


# ------------------------------------------------------------------ T05: muestra de etiquetado
def cmd_muestra(cfg, n):
    ruta = cfg["_sdir"] / "dataset_split.csv"
    if not ruta.is_file():
        salir("Falta dataset_split.csv. Ejecuta 'particionar' en pipeline.py primero.")
    sp = pd.read_csv(ruta)
    sp = sp[sp["split"].isin(["train", "val", "test"])].sort_values("segundo")
    if sp.empty:
        salir("No hay fotogramas válidos en train, val o test.")

    # cuotas: 20 % prueba; del 80 % restante, 20 % validación y 80 % entrenamiento
    cuotas = {"test": round(0.20 * n), "val": round(0.16 * n)}
    cuotas["train"] = n - cuotas["test"] - cuotas["val"]
    rng = np.random.default_rng(42)

    partes = []
    for s, q in cuotas.items():
        sub = sp[sp["split"] == s]
        if sub.empty:
            print(f"⚠️  No hay fotogramas en '{s}': se omite esa cuota.")
            continue
        partes.append(sub.loc[muestreo_sistematico(list(sub.index), q, rng)])
    muestra = pd.concat(partes).sort_values(["split", "segundo"]).reset_index(drop=True)

    raiz = cfg["_sdir"] / "etiquetado" / "images"
    for s in muestra["split"].unique():
        (raiz / s).mkdir(parents=True, exist_ok=True)
    for _, r in muestra.iterrows():
        shutil.copy2(cfg["_frames"] / r["filename"], raiz / r["split"] / r["filename"])

    muestra[["frame_id", "filename", "video_id", "group_id", "split", "segundo"]].to_csv(
        cfg["_sdir"] / "muestra_etiquetado.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(9, 2.4))
    for s, c in (("train", "#2E74B5"), ("val", "#E69F00"), ("test", "#C0392B")):
        t = muestra.loc[muestra["split"] == s, "segundo"] / 60
        ax.scatter(t, np.zeros(len(t)), s=18, color=c, label=f"{s} ({len(t)})")
    ax.set_yticks([])
    ax.set_xlabel("Minutos de video")
    ax.set_title(f"Muestra de etiquetado — {cfg['video_id']}")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.45))
    fig.tight_layout()
    fig.savefig(cfg["_evid"] / "muestra_distribucion.png", dpi=150)
    plt.close(fig)

    por_split = muestra["split"].value_counts().to_dict()
    sep = {s: round(float(np.median(np.diff(muestra.loc[muestra["split"] == s, "segundo"]))), 1)
           for s in por_split if (muestra["split"] == s).sum() > 1}
    res = {"n_solicitado": n, "n_obtenido": len(muestra), "por_particion": por_split,
           "separacion_mediana_s": sep, "semilla": 42, "carpeta": str(raiz)}
    guardar_json(cfg, "muestra_resumen.json", res)

    print("=" * 66)
    print(f"🏷️  Muestra de etiquetado: {len(muestra)} fotogramas  {por_split}")
    print(f"↔️  Separación mediana entre fotogramas (s): {sep}")
    print(f"📂 {raiz}  (subcarpetas train / val / test)")
    print("=" * 66)
    return res


# ------------------------------------------------------------------ T06a: línea base
def dibujar(img, filas, nombres):
    out = img.copy()
    esc = max(0.5, img.shape[1] / 1080 * 0.8)
    for _, f in filas.iterrows():
        c = COLORES.get(int(f["clase_id"]), (255, 255, 255))
        p1, p2 = (int(f["x1"]), int(f["y1"])), (int(f["x2"]), int(f["y2"]))
        cv2.rectangle(out, p1, p2, c, max(2, int(esc * 3)))
        cv2.putText(out, f"{nombres[int(f['clase_id'])]} {f['conf']:.2f}", (p1[0], max(12, p1[1] - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, esc * 0.7, c, max(1, int(esc * 2)), cv2.LINE_AA)
    return out


def con_titulo(img, texto, ancho):
    h, w = img.shape[:2]
    im = cv2.resize(img, (ancho, round(h * ancho / w)), interpolation=cv2.INTER_AREA)
    cv2.rectangle(im, (0, 0), (ancho, 22), (0, 0, 0), -1)
    cv2.putText(im, texto, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return im


def cmd_linea_base(cfg, n, modelo, tamanos, conf, device):
    try:
        import torch
        from ultralytics import YOLO
        import ultralytics
    except ImportError:
        salir("Falta ultralytics. Instálalo con:  pip install ultralytics")

    ruta_meta = cfg["_sdir"] / "dataset_metadata.csv"
    if not ruta_meta.is_file():
        salir("Falta dataset_metadata.csv. Ejecuta pipeline.py (extraer y auditar) primero.")
    meta = pd.read_csv(ruta_meta)
    validos = meta[meta["estado"] == "VALIDO"].sort_values("segundo")
    if validos.empty:
        salir("No hay fotogramas válidos. Ejecuta 'auditar' primero.")

    rng = np.random.default_rng(42)
    muestra = validos.loc[muestreo_sistematico(list(validos.index), n, rng)].reset_index(drop=True)
    imgs = [leer_imagen(cfg["_frames"] / f, cv2.IMREAD_COLOR) for f in muestra["filename"]]
    if any(i is None for i in imgs):
        salir("Algún fotograma de la muestra no se pudo leer.")

    dev = device if device != "auto" else (0 if torch.cuda.is_available() else "cpu")
    nombre_dev = torch.cuda.get_device_name(0) if dev != "cpu" else "CPU"
    print("=" * 66)
    print(f"🧠 Modelo: {modelo}  |  💻 {nombre_dev}  |  🖼️  {len(imgs)} fotogramas  |  imgsz {tamanos}  |  conf {conf}")
    print("=" * 66)

    mod = YOLO(modelo)
    nombres = mod.names
    carpeta = cfg["_sdir"] / "linea_base"
    carpeta.mkdir(exist_ok=True)

    mod.predict(imgs[0], imgsz=tamanos[0], conf=conf, device=dev, classes=list(COCO_VEHICULOS), verbose=False)  # calentamiento
    detecciones, resumen_filas = {}, []
    for sz in tamanos:
        filas, t_total = [], 0.0
        for ini in range(0, len(imgs), 8):
            t0 = time.perf_counter()
            res = mod.predict(imgs[ini:ini + 8], imgsz=sz, conf=conf, iou=0.7, device=dev,
                              classes=list(COCO_VEHICULOS), verbose=False)
            t_total += time.perf_counter() - t0
            for k, r in enumerate(res):
                b = r.boxes
                xyxy, cf, cl = b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy().astype(int)
                for (x1, y1, x2, y2), c, k_id in zip(xyxy, cf, cl):
                    filas.append({"filename": muestra.loc[ini + k, "filename"], "imgsz": sz, "clase_id": int(k_id),
                                  "clase": nombres[int(k_id)], "conf": float(c), "x1": float(x1), "y1": float(y1),
                                  "x2": float(x2), "y2": float(y2), "alto_px": float(y2 - y1), "ancho_px": float(x2 - x1)})
        df = pd.DataFrame(filas, columns=["filename", "imgsz", "clase_id", "clase", "conf", "x1", "y1", "x2", "y2",
                                          "alto_px", "ancho_px"])
        df.to_csv(carpeta / f"detecciones_imgsz{sz}.csv", index=False, encoding="utf-8-sig")
        detecciones[sz] = df
        por_clase = df["clase"].value_counts().to_dict()
        resumen_filas.append({
            "imgsz": sz, "detecciones": len(df), "por_fotograma": round(len(df) / len(imgs), 2),
            "fotogramas_sin_detecciones": int(len(imgs) - df["filename"].nunique()),
            "conf_media": round(float(df["conf"].mean()), 3) if len(df) else None,
            "alto_px_p10": round(float(df["alto_px"].quantile(0.10)), 1) if len(df) else None,
            "alto_px_mediana": round(float(df["alto_px"].median()), 1) if len(df) else None,
            "ms_por_fotograma": round(1000 * t_total / len(imgs), 1),
            **{f"n_{nombres[i]}": int(por_clase.get(nombres[i], 0)) for i in COCO_VEHICULOS},
        })
        print(f"   imgsz {sz}: {len(df)} detecciones ({len(df) / len(imgs):.1f} por fotograma), "
              f"{1000 * t_total / len(imgs):.0f} ms/fotograma")

    tabla = pd.DataFrame(resumen_filas)
    tabla.to_csv(cfg["_evid"] / "linea_base_resumen_por_resolucion.csv", index=False, encoding="utf-8-sig")

    # gráfico: detecciones por clase y resolución
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ancho_b = 0.8 / len(tamanos)
    for j, sz in enumerate(tamanos):
        v = [int(tabla.loc[tabla["imgsz"] == sz, f"n_{nombres[i]}"].iloc[0]) for i in COCO_VEHICULOS]
        ax.bar(np.arange(len(v)) + j * ancho_b, v, ancho_b, label=f"imgsz {sz}")
    ax.set_xticks(np.arange(len(COCO_VEHICULOS)) + ancho_b * (len(tamanos) - 1) / 2)
    ax.set_xticklabels([nombres[i] for i in COCO_VEHICULOS])
    ax.set_ylabel("Detecciones")
    ax.set_title(f"Línea base YOLOv8 sin ajuste — {len(imgs)} fotogramas")
    ax.legend()
    fig.tight_layout()
    fig.savefig(cfg["_evid"] / "linea_base_detecciones_por_clase.png", dpi=150)
    plt.close(fig)

    # hoja visual: fotograma con menos, mediana y más detecciones, comparando resoluciones
    ref = detecciones[max(tamanos)]
    cuenta = ref.groupby("filename").size().reindex(muestra["filename"], fill_value=0)
    orden = cuenta.sort_values()
    elegidos = [orden.index[0], orden.index[len(orden) // 2], orden.index[-1]]
    columnas = []
    for f in elegidos:
        i = int(muestra.index[muestra["filename"] == f][0])
        piezas = [con_titulo(dibujar(imgs[i], detecciones[sz][detecciones[sz]["filename"] == f], nombres),
                             f"{f[6:11]}  imgsz {sz}  n={int((detecciones[sz]['filename'] == f).sum())}", 380)
                  for sz in tamanos]
        columnas.append(np.vstack(piezas))  # una columna por fotograma, una fila por resolución
    alto = max(c.shape[0] for c in columnas)
    columnas = [np.pad(c, ((0, alto - c.shape[0]), (0, 0), (0, 0)), constant_values=255) for c in columnas]
    guardar_jpg(cfg["_evid"] / "linea_base_ejemplos.jpg", np.hstack(columnas), 88)

    sha_pesos = None
    try:
        import hashlib
        from pathlib import Path
        pesos = Path(modelo)
        if pesos.is_file():
            sha_pesos = hashlib.sha256(pesos.read_bytes()).hexdigest()
    except OSError:
        pass
    guardar_json(cfg, "linea_base_resumen.json", {
        "fecha_ejecucion": datetime.now().isoformat(timespec="seconds"), "modelo": modelo, "sha256_pesos": sha_pesos,
        "ultralytics": ultralytics.__version__, "torch": torch.__version__, "dispositivo": nombre_dev,
        "n_fotogramas": len(imgs), "conf": conf, "iou_nms": 0.7, "clases_coco": COCO_VEHICULOS,
        "semilla_muestra": 42, "por_resolucion": resumen_filas,
        "nota": "Línea base sin etiquetas propias: solo cuenta detecciones por clase COCO; no se calcula mAP.",
    })

    print("=" * 66)
    print(tabla[["imgsz", "detecciones", "por_fotograma", "fotogramas_sin_detecciones", "conf_media",
                 "alto_px_p10", "ms_por_fotograma"]].to_string(index=False))
    print(f"🖼️  Revisa {cfg['_evid'] / 'linea_base_ejemplos.jpg'}")
    print("=" * 66)
    return {"n": len(imgs), "por_resolucion": [{k: r[k] for k in ("imgsz", "detecciones", "ms_por_fotograma")}
                                                for r in resumen_filas]}


# ------------------------------------------------------------------ T05: preetiquetas para CVAT
# Etiquetas finas que se anotan en CVAT (Tabla 3.3 del protocolo)
ETIQUETAS_FINAS = ["moto", "torito", "tricimoto", "moto_carga", "auto", "camioneta", "combi", "custer",
                   "bus", "camion", "trailer", "otro"]
# Clase de entrenamiento de cada etiqueta fina (None = no se entrena, solo cuenta en el total vehicular)
MAPEO_ENTRENAMIENTO = {"moto": "moto", "torito": "torito", "tricimoto": "tricimoto", "moto_carga": "moto_carga",
                       "auto": "auto", "camioneta": "camioneta", "combi": "combi_custer", "custer": "combi_custer",
                       "bus": "bus", "camion": "camion", "trailer": "trailer", "otro": None}
# Preetiquetado provisional: solo las clases de COCO que existen; el resto se corrige a mano
COCO_A_FINA = {2: "auto", 3: "moto", 5: "bus", 7: "camion"}
COLORES_CVAT = ["#d62728", "#ff7f0e", "#bcbd22", "#8c564b", "#1f77b4", "#17becf", "#2ca02c", "#98df8a",
                "#9467bd", "#e377c2", "#7f7f7f", "#000000"]


def escribir(ruta, texto):
    Path(ruta).write_text(texto, encoding="utf-8", newline="\n")


def cmd_preetiquetar(cfg, modelo, imgsz, conf, device):
    try:
        import torch
        from ultralytics import YOLO
    except ImportError:
        salir("Falta ultralytics. Instálalo con:  pip install ultralytics")
    ruta_m = cfg["_sdir"] / "muestra_etiquetado.csv"
    if not ruta_m.is_file():
        salir("Falta muestra_etiquetado.csv. Ejecuta primero:  python percepcion.py muestra <config> --n 200")
    muestra = pd.read_csv(ruta_m)
    base = cfg["_sdir"] / "etiquetado"
    dev = device if device != "auto" else (0 if torch.cuda.is_available() else "cpu")
    mod = YOLO(modelo)

    # archivos de apoyo: mapeo de clases (proyecto) y etiquetas para CVAT (sesión)
    entren = sorted({v for v in MAPEO_ENTRENAMIENTO.values() if v}, key=lambda c: [
        "moto", "torito", "tricimoto", "moto_carga", "auto", "camioneta", "combi_custer", "bus", "camion", "trailer"].index(c))
    lineas = ["# Mapeo de etiquetas finas (CVAT) a clases de entrenamiento — Protocolo v2.1, Tabla 3.3",
              "# 'null' = no se entrena; el vehículo cuenta en el total vehicular pero no en el desglose por tipo.",
              "clases_entrenamiento:"] + [f"  {i}: {c}" for i, c in enumerate(entren)] + ["mapeo:"]
    lineas += [f"  {k}: {v if v else 'null'}" for k, v in MAPEO_ENTRENAMIENTO.items()]
    ruta_map = cfg["_base"] / "mapeo_clases.yaml"
    if not ruta_map.is_file():
        escribir(ruta_map, "\n".join(lineas) + "\n")
    etiquetas = []
    for i, nombre in enumerate(ETIQUETAS_FINAS):
        e = {"name": nombre, "color": COLORES_CVAT[i], "type": "rectangle", "attributes": []}
        if nombre == "trailer":
            e["attributes"] = [{"name": "subtipo", "mutable": False, "input_type": "select",
                                "default_value": "otro", "values": ["ranfla", "cisterna", "plataforma", "otro"]}]
        etiquetas.append(e)
    base.mkdir(exist_ok=True)
    escribir(base / "cvat_etiquetas.json", json.dumps(etiquetas, ensure_ascii=False, indent=2) + "\n")

    resumen, total = {}, {}
    for split, sub in muestra.groupby("split"):
        carpeta = base / "preetiquetas" / split
        (carpeta / "obj_train_data").mkdir(parents=True, exist_ok=True)
        nombres_img, conteo = [], {}
        sub = sub.reset_index(drop=True)
        for ini in range(0, len(sub), 8):
            lote = sub.iloc[ini:ini + 8]
            imgs = [leer_imagen(base / "images" / split / f, cv2.IMREAD_COLOR) for f in lote["filename"]]
            if any(i is None for i in imgs):
                salir(f"No se pudo leer una imagen de etiquetado/images/{split}. Ejecuta 'muestra' de nuevo.")
            res = mod.predict(imgs, imgsz=imgsz, conf=conf, iou=0.6, device=dev,
                              classes=list(COCO_A_FINA), verbose=False)
            for f, im, r in zip(lote["filename"], imgs, res):
                h, w = im.shape[:2]
                filas = []
                b = r.boxes
                for (x1, y1, x2, y2), k in zip(b.xyxy.cpu().numpy(), b.cls.cpu().numpy().astype(int)):
                    fina = COCO_A_FINA[int(k)]
                    conteo[fina] = conteo.get(fina, 0) + 1
                    filas.append(f"{ETIQUETAS_FINAS.index(fina)} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                                 f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
                escribir(carpeta / "obj_train_data" / (Path(f).stem + ".txt"), "\n".join(filas) + ("\n" if filas else ""))
                nombres_img.append(f"data/obj_train_data/{f}")
        escribir(carpeta / "obj.names", "\n".join(ETIQUETAS_FINAS) + "\n")
        escribir(carpeta / "obj.data", f"classes = {len(ETIQUETAS_FINAS)}\nnames = obj.names\ntrain = train.txt\nbackup = backup/\n")
        escribir(carpeta / "train.txt", "\n".join(nombres_img) + "\n")
        with zipfile.ZipFile(base / f"cvat_import_{split}.zip", "w", zipfile.ZIP_DEFLATED) as z:
            for arch in ("obj.data", "obj.names", "train.txt"):
                z.write(carpeta / arch, arch)
            for t in sorted((carpeta / "obj_train_data").glob("*.txt")):
                z.write(t, f"obj_train_data/{t.name}")
        resumen[split] = {"imagenes": len(sub), "preetiquetas": int(sum(conteo.values())), "por_clase": conteo}
        for k_, v_ in conteo.items():
            total[k_] = total.get(k_, 0) + v_

    escribir(base / "LEEME_CVAT.txt", """PASOS EN CVAT (app.cvat.ai)  — los nombres de menú pueden variar un poco según la versión
1. Crea una cuenta y un PROYECTO (por ejemplo 'Piloto_tesis').
2. En el proyecto: Constructor de etiquetas -> pestaña 'Raw' -> pega el contenido de cvat_etiquetas.json -> Done.
   (Si el Raw no lo acepta, agrega a mano las 12 etiquetas con los mismos nombres, en el mismo orden.)
3. Crea 3 TAREAS dentro del proyecto: 'train', 'val' y 'test'. En cada una sube las imágenes de
   etiquetado/images/<train|val|test>. No mezcles particiones.
4. En cada tarea: menú de la tarea (Actions) -> Upload annotations -> formato 'YOLO 1.1' ->
   selecciona cvat_import_<partición>.zip. Las cajas aparecen ya dibujadas.
5. Corrige a mano: cambia la etiqueta (torito, tricimoto, moto_carga, combi, custer, trailer...), ajusta cajas,
   agrega los vehículos que falten y borra las cajas falsas. En los tráileres indica el subtipo.
6. Al terminar cada tarea: Actions -> Export task dataset -> 'YOLO 1.1' (sin imágenes), y también
   'CVAT for images 1.1' como respaldo (conserva el subtipo). Guarda los zips en etiquetado/exportados/<partición>/.
7. Criterio de etiquetado: una caja por vehículo completo (el tráiler es UNA sola caja, cabina + remolque);
   vehículos cortados por el borde se etiquetan solo si se ve más de la mitad; los detenidos también se etiquetan.
""")
    guardar_json(cfg, "preetiquetas_resumen.json", {
        "fecha_ejecucion": datetime.now().isoformat(timespec="seconds"), "modelo": modelo, "imgsz": imgsz, "conf": conf,
        "iou_nms": 0.6, "mapeo_coco_a_fina": {str(k): v for k, v in COCO_A_FINA.items()},
        "etiquetas_finas": ETIQUETAS_FINAS, "por_particion": resumen, "total_por_clase": total,
        "nota": "Preetiquetas provisionales con clases COCO; toritos, tricimotos, motos de carga, combis, custers "
                "y tráileres se corrigen manualmente."})
    print("=" * 66)
    for sp, r in resumen.items():
        print(f"🏷️  {sp}: {r['imagenes']} imágenes, {r['preetiquetas']} preetiquetas  {r['por_clase']}")
    print(f"📦 Zips: {base}\\cvat_import_<train|val|test>.zip   |   📄 {base / 'LEEME_CVAT.txt'}")
    print(f"📄 {ruta_map.name} y cvat_etiquetas.json generados")
    print("=" * 66)
    return {"particiones": {k_: v_["preetiquetas"] for k_, v_ in resumen.items()}}


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description="Etapa de percepción del piloto (Protocolo v2.1)")
    sub = ap.add_subparsers(dest="comando", required=True)
    s1 = sub.add_parser("muestra")
    s1.add_argument("config")
    s1.add_argument("--n", type=int, default=200, help="fotogramas a etiquetar")
    s2 = sub.add_parser("linea_base")
    s2.add_argument("config")
    s2.add_argument("--n", type=int, default=60, help="fotogramas de la muestra")
    s2.add_argument("--modelo", default="yolov8s.pt")
    s2.add_argument("--imgsz", type=int, nargs="+", default=[640, 1280])
    s2.add_argument("--conf", type=float, default=0.25)
    s2.add_argument("--device", default="auto", help="auto, cpu o índice de GPU (0)")
    s3 = sub.add_parser("preetiquetar")
    s3.add_argument("config")
    s3.add_argument("--modelo", default="yolov8s.pt")
    s3.add_argument("--imgsz", type=int, default=1280)
    s3.add_argument("--conf", type=float, default=0.30)
    s3.add_argument("--device", default="auto")
    a = ap.parse_args()

    cfg = cargar_config(a.config)
    t0 = time.time()
    log(cfg, f"INICIO {a.comando} | {' '.join(sys.argv[1:])}")
    if a.comando == "muestra":
        r = cmd_muestra(cfg, a.n)
    elif a.comando == "preetiquetar":
        r = cmd_preetiquetar(cfg, a.modelo, a.imgsz, a.conf, a.device)
    else:
        r = cmd_linea_base(cfg, a.n, a.modelo, a.imgsz, a.conf, a.device)
    log(cfg, f"FIN {a.comando} | {json.dumps(r, ensure_ascii=False, default=a_json)}")
    log(cfg, f"TIEMPO {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
