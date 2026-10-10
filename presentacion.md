---
marp: true
theme: default
paginate: true
header: "Piloto Exploratorio: Percepción Vehicular con YOLOv8"
footer: "Universidad Nacional de Juliaca — Sesión BYPASS_20260928_V01"
style: |
  section {
    font-family: 'Helvetica Neue', Arial, sans-serif;
    padding: 40px;
  }
  h1 { color: #1e3a8a; }
  h2 { color: #2563eb; }
  table { font-size: 0.8em; }
  .highlight { background-color: #f3f4f6; padding: 10px; border-radius: 5px; }
---

# Piloto Exploratorio: Percepción Vehicular con YOLOv8 en el Bypass
## De la Extracción de Fotogramas al Modelo Ajustado (Sesión `BYPASS_20260928_V01`)

**Tesista:** Hector Carlos Flores Torres[cite: 10]  
**Área:** Visión por Computadora y Control de Tráfico Urbano  
**Fecha:** Septiembre / Octubre 2026  

---

## Diapositiva 1: Contexto General y Auditoría del Protocolo

### Comparativa de Parámetros
* **Resolución de Captura:** Protocolo v2.1 ($4K$ a $30\text{ FPS}$) vs. Piloto Real ($1080\times1920$ vertical a $29.5\text{ FPS}$)[cite: 10].
* **Duración:** Protocolo ($2\text{ h}$) vs. Piloto Real ($40\text{ min } 51\text{ s}$)[cite: 10].
* **Entrada del Detector:** Recortes $800\times450$ vs. Fotograma completo a $1280\text{ px}$[cite: 10].

### Trazabilidad y Seguridad
* **Hash SHA-256 del Video:** `f1906f6abf...` (Garantiza reproducibilidad del experimento)[cite: 10, 11].

> **Criterio Metodológico:** Declarar transparentemente las desviaciones del piloto frente al protocolo evita la sobregeneralización de resultados y valida la viabilidad previa al despliegue masivo[cite: 10].

---

## Diapositiva 2: Extracción y Control de Calidad de Fotogramas

### Métricas de Extracción
* **Fotogramas procesados:** $2,452$ extraídos a $1\text{ FPS}$ ($183\text{ s}$)[cite: 10].
* **Fotogramas válidos:** $99.8\%$ ($2,447$ de $2,452$)[cite: 10].
* **Descartes:** $5$ fotogramas por inestabilidad de la cámara[cite: 10].

### Métricas de Nitidez y Estabilidad
* **Varianza Laplaciana (Mediana):** $1504.2$ | **Umbral Crítico (40%):** $601.7$[cite: 10].
* **Vibración de Cámara (P90):** $\pm 79\text{ px}$ horizontal y $\pm 178\text{ px}$ vertical[cite: 10].
* **Deriva Temporal:** Detectada deriva de hasta $20.158\text{ s}$ por grabación en Tasa de Fotogramas Variable (VFR)[cite: 11].

---

## Diapositiva 3: Estrategia de Partición Temporal Anti-Solapamiento

### Distribución del Dataset
* **Entrenamiento (Train):** $1,107$ fotogramas[cite: 10].
* **Validación (Val):** $541$ fotogramas[cite: 10].
* **Prueba (Test):** $622$ fotogramas[cite: 10].
* **Bloques de Guarda (Guardbands):** $177$ fotogramas de aislamiento[cite: 10].

### Reglas de Control Integradas
* **Semilla Aleatoria:** `42`[cite: 10].
* **Solapamiento entre Grupos:** $0$[cite: 10].
* **Separación Temporal Mínima:** $\ge 60\text{ s}$ entre prueba y entrenamiento[cite: 10].

> **Justificación Técnica:** Evita la **fuga de información (*data leakage*)**[cite: 10]. Al separar en bloques temporales con guardas, se evita probar el modelo con fotogramas casi idénticos a los de entrenamiento[cite: 10].

---

## Diapositiva 4: Línea Base vs. Etiquetado Muestral en CVAT

### Evaluaciones con YOLOv8 Predeterminado
* **A $640\text{ px}$:** $17.9$ detecciones/foto ($90\text{ ms/foto}$)[cite: 10].
* **A $1280\text{ px}$:** $27.2$ detecciones/foto ($334\text{ ms/foto}$)[cite: 10].
* **Diagnóstico:** Aumentar resolución mejora detecciones pequeñas, pero el modelo genérico falla en la taxonomía local (ej. confunde toritos con autos)[cite: 10].

### Curación y Taxonomía del Dataset
* **Muestra Etiquetada:** $120$ fotogramas manuales ($77$ train / $19$ val / $24$ test)[cite: 10].
* **$8$ Clases Definidas:** `moto`, `torito`, `moto_carga`, `auto`, `camioneta`, `combi_custer`, `bus`, `camion`[cite: 10].
* **Regla de Oclusión:** Inclusión de etiquetas solo si se observa $>50\%$ de la carrocería[cite: 10].

---

## Diapositiva 5: Rendimiento del Modelo Ajustado (YOLOv8s)

### Configuración del Entrenamiento
* **Entorno:** GPU Tesla T4, `imgsz=1280`, `batch=8`, `patience=20`, hasta 100 épocas[cite: 10].

### Métricas Globales en Conjunto de Prueba
* **$\text{mAP}@0.5$:** $55.7\%$[cite: 10]
* **$\text{mAP}@0.5:0.95$:** $45.3\%$[cite: 10]
* **Precisión (P):** $50.7\%$ | **Recall (R):** $56.9\%$[cite: 10]
* **Meta Inicial del Protocolo ($\text{mAP}@0.5 > 95\%$):** No alcanzada en este piloto de muestra reducida ($120$ imágenes)[cite: 10].

### Evidencia de Análisis Visual
1. **Curva F1-Confidence (`F1_curve.png`):** Muestra el punto de equilibrio óptimo alcanzando un F1 cercano a $0.74$ a un umbral de confianza entre $0.35$ y $0.45$.
2. **Matriz de Confusión (`confusion_matrix_normalized.png`):** Resalta alto desempeño en clases frecuentes y confusión en el fondo (*background*) para clases con escasos datos.

---

## Diapositiva 6: Desglose Métrico por Clase Vehicular

### Evaluación sobre $546$ Instancias de Prueba[cite: 10]

| Clase Vehicular | Instancias en Test | $\text{AP}@0.5$ | $\text{AP}@0.5:0.95$ | Diagnóstico de Desempeño |
| :--- | :---: | :---: | :---: | :--- |
| **Torito** | $120$ | **$0.868$** | $0.726$ | **Excelente:** Alta especialización[cite: 10] |
| **Camión** | $126$ | **$0.776$** | $0.632$ | **Alto:** Geometría clara[cite: 10] |
| **Combi / Custer** | $67$ | **$0.762$** | $0.644$ | **Alto:** Patrón vehicular definido[cite: 10] |
| **Auto** | $164$ | $0.686$ | $0.596$ | **Moderado:** Confusión con camionetas[cite: 10] |
| **Moto Carga** | $7$ | $0.555$ | $0.426$ | Muestras reducidas[cite: 10] |
| **Moto** | $22$ | $0.402$ | $0.262$ | Oclusión y área pequeña[cite: 10] |
| **Camioneta** | $16$ | $0.296$ | $0.261$ | Confusión con autos[cite: 10] |
| **Bus** | $24$ | $0.108$ | $0.079$ | Bajo (Pocas muestras)[cite: 10] |

---

## Diapositiva 7: Comparativa: Modelo Base vs. Modelo Ajustado

### Impacto de la Adaptación Taxonómica
* **Vehículos de Tres Ruedas (`torito`, `moto_carga`):**  
  * *Modelo Base:* $\text{F1} = 0.00$ (Omitidos o clasificados erróneamente como autos)[cite: 10].  
  * *Modelo Ajustado:* **$\text{F1} = 0.78$**[cite: 10].
* **Vehículos Pesados (`bus`, `camion`):**  
  * *Modelo Base:* $\text{F1} = 0.37$[cite: 10].  
  * *Modelo Ajustado:* **$\text{F1} = 0.76$**[cite: 10].

> **Análisis Crítico:** El modelo base reportaba un F1 inflado ($0.94$) debido a agrupaciones genéricas inadecuadas para flujo vehicular urbano real[cite: 10]. La especialización en $8$ clases ajusta la precisión al entorno real del bypass[cite: 10].

---

## Diapositiva 8: Limitaciones e Integración al Trabajo Futuro

### Limitaciones Identificadas
1. Toma de video única en orientación vertical[cite: 10].
2. Dataset reducido para entrenamiento ($120$ fotogramas anotados)[cite: 10].
3. Variabilidad de tasa de fotogramas (VFR) en la fuente[cite: 11].

### Hoja de Ruta Técnica
1. **Ampliación del Dataset:** Muestreo multitemporal y en formato horizontal[cite: 10].
2. **Estandarización VFR a CFR:** Conversión mediante `ffmpeg` antes del procesamiento[cite: 11].
3. **Módulo de Tracking:** Evaluación de error absoluto de conteo vehicular ($5–10\text{ FPS}$)[cite: 10].
4. **Integración con Agente DQN:** Uso de matrices de flujo para entrenamiento en simulación[cite: 10].