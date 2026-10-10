#!/usr/bin/env python3
"""
dataset.py — Piloto de tesis (Protocolo Experimental v2.1)
Convierte lo exportado de CVAT (YOLO 1.1) en el dataset de entrenamiento (T05).

Qué hace
  · Lee el zip YOLO 1.1 exportado de cada tarea (train, val, test) en etiquetado\\exportados\\<partición>\\
  · Convierte las etiquetas finas de CVAT a las clases de entrenamiento con mapeo_clases.yaml
    (por ejemplo, combi y custer -> combi_custer; 'otro' no se entrena)
  · Permite fusionar clases con pocas instancias (--fusionar trailer:camion)
  · Crea sesiones\\<video_id>\\dataset_yolo\\ (imágenes, etiquetas, data.yaml, clases.txt) y dataset_yolo.zip
  · Genera el reporte de instancias por clase y partición (evidencia para la exposición)
  · Verifica que no haya fuga entre particiones

Uso (en la carpeta de pipeline.py):
  python dataset.py convertir sesion_BYPASS_20260928_V01.yaml
  python dataset.py convertir sesion_BYPASS_20260928_V01.yaml --fusionar trailer:camion tricimoto:torito
"""
import argparse
import json
import re
import shutil
import sys
import time
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from pipeline import a_json, cargar_config, guardar_json, log, salir

PARTICIONES = ("train", "val", "test")


def leer_export(ruta_zip):
    """(nombres, {stem: [(clase_id, cx, cy, w, h), ...]}) de un zip YOLO 1.1 de CVAT; None si no lo es."""
    with zipfile.ZipFile(ruta_zip) as z:
        archivos = z.namelist()
        if "obj.names" not in archivos:
            return None
        nombres = [l.strip() for l in z.read("obj.names").decode("utf-8-sig").splitlines() if l.strip()]
        etiquetas = {}
        for n in archivos:
            if re.match(r"^obj_[^/]*_data/[^/]+\.txt$", n):
                filas = []
                for ln in z.read(n).decode("utf-8-sig").splitlines():
                    p = ln.split()
                    if len(p) >= 5:
                        filas.append((int(float(p[0])), *map(float, p[1:5])))
                etiquetas[Path(n).stem] = filas
    return nombres, etiquetas


def zip_yolo_de(carpeta):
    """El zip YOLO 1.1 más reciente de la carpeta (ignora los de 'CVAT for images')."""
    zips = sorted(Path(carpeta).glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
    for z in zips:
        try:
            r = leer_export(z)
        except zipfile.BadZipFile:
            continue
        if r is not None:
            return z, r
    return None, None


def cmd_convertir(cfg, fusionar, minimo_train):
    base = cfg["_sdir"] / "etiquetado"
    ruta_map = cfg["_base"] / "mapeo_clases.yaml"
    ruta_muestra = cfg["_sdir"] / "muestra_etiquetado.csv"
    for r in (ruta_map, ruta_muestra):
        if not r.is_file():
            salir(f"Falta {r.name}. Se genera con 'muestra' y 'preetiquetar' de percepcion.py.")
    with open(ruta_map, encoding="utf-8") as f:
        mp = yaml.safe_load(f)
    clases_ent = [mp["clases_entrenamiento"][k] for k in sorted(mp["clases_entrenamiento"])]
    mapeo = mp["mapeo"]

    # fusiones: origen:destino sobre clases de entrenamiento
    fus = {}
    for par in fusionar or []:
        if ":" not in par:
            salir(f"--fusionar espera origen:destino (por ejemplo trailer:camion); recibí '{par}'.")
        o, d = par.split(":", 1)
        if o not in clases_ent or d not in clases_ent:
            salir(f"Clase desconocida en '{par}'. Clases de entrenamiento: {clases_ent}")
        fus[o] = d
    finales = [c for c in clases_ent if c not in fus]
    idx = {c: i for i, c in enumerate(finales)}

    muestra = pd.read_csv(ruta_muestra)
    esperado = muestra.groupby("split")["filename"].apply(set).to_dict()
    destino = cfg["_sdir"] / "dataset_yolo"
    if destino.exists():
        shutil.rmtree(destino)
    for s in PARTICIONES:
        (destino / "images" / s).mkdir(parents=True)
        (destino / "labels" / s).mkdir(parents=True)

    cuentas = {s: {c: 0 for c in finales} for s in PARTICIONES}
    resumen_img, avisos, desconocidas, otro_n, fuente = {}, [], set(), 0, {}
    for s in PARTICIONES:
        carpeta = base / "exportados" / s
        zip_path, lectura = zip_yolo_de(carpeta) if carpeta.is_dir() else (None, None)
        if lectura is None:
            salir(f"No encontré un zip YOLO 1.1 de CVAT en {carpeta}. Exporta la tarea '{s}' (YOLO 1.1) y guárdala ahí.")
        nombres, etiquetas = lectura
        fuente[s] = zip_path.name
        imgs = sorted((base / "images" / s).glob("*.jpg"))
        sin_etiqueta, vacias = [], 0
        for img in imgs:
            if img.stem not in etiquetas:
                sin_etiqueta.append(img.name)
            filas_out = []
            for cid, cx, cy, w, h in etiquetas.get(img.stem, []):
                if cid >= len(nombres):
                    salir(f"{zip_path.name}: clase {cid} fuera de obj.names.")
                fina = nombres[cid]
                if fina not in mapeo:
                    desconocidas.add(fina)
                    continue
                ent = mapeo[fina]
                if ent is None:
                    otro_n += 1
                    continue
                ent = fus.get(ent, ent)
                x1, y1, x2, y2 = max(0.0, cx - w / 2), max(0.0, cy - h / 2), min(1.0, cx + w / 2), min(1.0, cy + h / 2)
                if x2 - x1 <= 0 or y2 - y1 <= 0:
                    continue
                filas_out.append(f"{idx[ent]} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}")
                cuentas[s][ent] += 1
            if not filas_out:
                vacias += 1
            shutil.copy2(img, destino / "images" / s / img.name)
            (destino / "labels" / s / (img.stem + ".txt")).write_text("\n".join(filas_out) + ("\n" if filas_out else ""), encoding="utf-8")
        faltan = sorted(esperado.get(s, set()) - {i.name for i in imgs})
        resumen_img[s] = {"imagenes": len(imgs), "sin_objetos": vacias, "sin_archivo_de_etiquetas": len(sin_etiqueta),
                          "esperadas_en_muestra": len(esperado.get(s, set())), "faltantes_en_carpeta": len(faltan)}
        if sin_etiqueta:
            avisos.append(f"{s}: {len(sin_etiqueta)} imágenes no aparecen en la exportación (¿tarea sin guardar o exportación parcial?).")
        if faltan:
            avisos.append(f"{s}: faltan {len(faltan)} imágenes de la muestra en etiquetado\\images\\{s}.")
    if desconocidas:
        salir(f"Etiquetas de CVAT que no están en mapeo_clases.yaml: {sorted(desconocidas)}. "
              "Corrige el nombre en CVAT o agrégalo al mapeo.")

    # fuga entre particiones: ningún fotograma ni grupo en más de una partición
    nombres_por_s = {s: {p.name for p in (destino / "images" / s).glob("*.jpg")} for s in PARTICIONES}
    repetidos = (nombres_por_s["train"] & nombres_por_s["val"]) | (nombres_por_s["train"] & nombres_por_s["test"]) | \
                (nombres_por_s["val"] & nombres_por_s["test"])
    if repetidos:
        salir(f"Fotogramas repetidos entre particiones: {sorted(repetidos)[:5]}")
    ruta_split = cfg["_sdir"] / "dataset_split.csv"
    solapamiento = None
    if ruta_split.is_file():
        sp = pd.read_csv(ruta_split)
        pos = {n: s for s, ns in nombres_por_s.items() for n in ns}
        sp = sp[sp["filename"].isin(pos)].assign(carpeta=lambda d: d["filename"].map(pos))
        solapamiento = int((sp.groupby("group_id")["carpeta"].nunique() > 1).sum())
        if solapamiento:
            salir(f"Un mismo group_id aparece en más de una partición ({solapamiento}). Vuelve a ejecutar 'muestra'.")

    # data.yaml, clases.txt y zip
    (destino / "clases.txt").write_text("\n".join(finales) + "\n", encoding="utf-8")
    (destino / "data.yaml").write_text(
        "path: .\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n" +
        "\n".join(f"  {i}: {c}" for i, c in enumerate(finales)) + "\n", encoding="utf-8")
    shutil.make_archive(str(cfg["_sdir"] / "dataset_yolo"), "zip", destino)

    # reporte
    tabla = pd.DataFrame(cuentas).T[finales]
    tabla["total"] = tabla.sum(axis=1)
    tabla.loc["TOTAL"] = tabla.sum()
    tabla.to_csv(cfg["_evid"] / "dataset_instancias_por_clase.csv", encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ancho = 0.27
    for k, (s, c) in enumerate((("train", "#2E74B5"), ("val", "#E69F00"), ("test", "#C0392B"))):
        ax.bar(np.arange(len(finales)) + k * ancho, [cuentas[s][f] for f in finales], ancho, label=s, color=c)
    ax.set_xticks(np.arange(len(finales)) + ancho)
    ax.set_xticklabels(finales, rotation=30, ha="right")
    ax.set_ylabel("Instancias etiquetadas")
    ax.set_title(f"Dataset del piloto — instancias por clase y partición ({cfg['video_id']})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(cfg["_evid"] / "dataset_instancias_por_clase.png", dpi=150)
    plt.close(fig)

    pocas = [c for c in finales if cuentas["train"][c] < minimo_train]
    sin_val_test = [c for c in finales if cuentas["train"][c] > 0 and (cuentas["val"][c] == 0 or cuentas["test"][c] == 0)]
    guardar_json(cfg, "dataset_resumen.json", {
        "clases_entrenamiento": finales, "fusiones": fus, "zips_usados": fuente, "imagenes": resumen_img,
        "instancias": {s: cuentas[s] for s in PARTICIONES}, "etiquetas_otro_descartadas": otro_n,
        "clases_con_pocas_instancias_en_train": {c: cuentas["train"][c] for c in pocas}, "minimo_train": minimo_train,
        "clases_sin_ejemplos_en_val_o_test": sin_val_test, "solapamiento_de_grupos": solapamiento, "avisos": avisos})

    print("=" * 66)
    print(f"📦 Dataset: {destino}   |   🗜️  {cfg['_sdir'] / 'dataset_yolo.zip'}")
    for s in PARTICIONES:
        print(f"   {s}: {resumen_img[s]['imagenes']} imágenes ({resumen_img[s]['sin_objetos']} sin objetos), "
              f"{sum(cuentas[s].values())} instancias  [{fuente[s]}]")
    print(tabla.to_string())
    print(f"🏷️  Etiquetas 'otro' (no entrenadas): {otro_n}")
    if pocas:
        lista_pocas = ", ".join(c + " (" + str(cuentas["train"][c]) + ")" for c in pocas)
        print(f"⚠️  Menos de {minimo_train} instancias en train: {lista_pocas}")
        print("    Considera fusionarlas, por ejemplo:  --fusionar trailer:camion tricimoto:torito")
    if sin_val_test:
        print(f"⚠️  Sin ejemplos en val o test (no se podrán evaluar): {', '.join(sin_val_test)}")
    for a in avisos:
        print(f"⚠️  {a}")
    if solapamiento is not None:
        print(f"✅ Solapamiento de grupos entre particiones: {solapamiento}")
    print("=" * 66)
    return {"instancias_train": sum(cuentas["train"].values()), "clases": len(finales)}


def main():
    ap = argparse.ArgumentParser(description="Dataset de entrenamiento a partir de CVAT (Protocolo v2.1)")
    sub = ap.add_subparsers(dest="comando", required=True)
    s = sub.add_parser("convertir")
    s.add_argument("config")
    s.add_argument("--fusionar", nargs="*", default=[], help="origen:destino, por ejemplo trailer:camion")
    s.add_argument("--minimo-train", type=int, default=30, help="instancias mínimas por clase en train antes de avisar")
    a = ap.parse_args()
    cfg = cargar_config(a.config)
    t0 = time.time()
    log(cfg, f"INICIO dataset {a.comando} | {' '.join(sys.argv[1:])}")
    r = cmd_convertir(cfg, a.fusionar, a.minimo_train)
    log(cfg, f"FIN dataset {a.comando} | {json.dumps(r, ensure_ascii=False, default=a_json)}")
    log(cfg, f"TIEMPO {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
