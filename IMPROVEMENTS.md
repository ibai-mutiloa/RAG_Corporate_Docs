# Mejoras de Blindaje y Validación - Indexador

## ✅ Implementadas

### 1️⃣ Blindaje de tamaño por chunk (MAX_TOKENS_PER_CHUNK)

**Qué se hizo:**
- Se añadió la variable `MAX_TOKENS_PER_CHUNK=2000` en `.env`
- Se implementó `count_tokens()` que usa `tiktoken` para contar tokens exactamente
- Fallback automático: si tiktoken falla, estima ~4 caracteres = 1 token

**Comportamiento:**
```python
def calculate_embeddings(chunks):
    for chunk in chunks:
        token_count = count_tokens(chunk)
        if token_count > MAX_TOKENS_PER_CHUNK:
            print(f"[ERROR] Chunk demasiado grande ({token_count} tokens) → NO se procesará")
            all_embeddings.append(None)  # Rechaza embedding
        else:
            print(f"[DEBUG] Embedding chunk → {token_count} tokens")
            # Procesa normalmente
```

**Ventaja:** Evita errores 406 de Azure OpenAI por chunks muy grandes.

---

### 2️⃣ Regla de oro: Si falla embedding → NO insertar

**Qué se hizo:**
- Modificado `insert_chunks()`: filtra chunks sin embedding válido
- Los chunks rechazados quedan como NULL en la BD (no se insertan)

**Comportamiento:**
```python
for i, chunk in enumerate(chunks):
    if embeddings[i] is None:
        print(f"[WARN] Chunk {i} sin embedding → NO se insertará")
        skipped_count += 1
    else:
        valid_data.append((file_name, file_path, ..., embeddings[i], ...))

# Solo inserta valid_data (sin NULLs)
```

**Ventaja:** Índice limpio. Los 40 chunks sin embedding de tu ejecución quedan registrados para reparación.

---

### 3️⃣ Check post-indexación

**Qué se hizo:**
- Nueva función `check_null_embeddings()` ejecutada al final
- Lista todos los archivos con chunks NULL agrupados
- Muestra alarma 🚨 si hay problemas

**Comportamiento:**
```
================================
🚨 [ALARMA] Chunks sin embedding detectados:
================================
  • normativa_001.pdf: 5 chunks sin embedding
  • normativa_002.pdf: 3 chunks sin embedding
================================
```

**Ventaja:** Detección inmediata de problemas en tiempo de indexación.

---

### 4️⃣ Logs explícitos de tamaño

**Qué se hizo:**
- Cada embedding loguea su tamaño en tokens:
  ```
  [DEBUG] Embedding chunk 0 → 342 tokens
  [DEBUG] Embedding chunk 1 → 1856 tokens
  [ERROR] Embedding chunk 2 → 2500 tokens (RECHAZADO)
  ```

**Ventaja:** Trazabilidad inmediata de chunks problemáticos.

---

### 5️⃣ Función de reparación automática

**Qué se hizo:**
- Nueva función `update_all_missing_embeddings()`
- Se ejecuta automáticamente en el `main` si hay chunks NULL
- Intenta recalcular embeddings para todos los archivos afectados

**Comportamiento:**
```
[INFO] 🔄 Actualizando embeddings para 3 archivos...
[INFO] Calculando 5 embeddings faltantes para documento_001.pdf
[INFO] ✅ Total chunks actualizados: 5
```

**Ventaja:** El proceso se autocorrige automáticamente.

---

### 6️⃣ 🎯 Heurística inteligente de segunda pasada (NUEVA - API)

**Qué se hizo:**
- Implementado sistema de **dos pases** en el endpoint `/search`
- Validación automática de palabras clave normativas esperadas
- Segundo pase dirigido cuando faltan keywords

**Cómo funciona:**

```
┌─────────────────────────────────────────┐
│  PREGUNTA: "¿Cuál es la mayoría...?"    │
└─────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────┐
│ PRIMER PASE: Búsqueda semántica normal  │
│ • Embedding de la pregunta              │
│ • Similitud del coseno vs chunks        │
│ • Resultado: max_similarity = 0.68      │
└─────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────┐
│ VALIDACIÓN: ¿Contiene keywords?         │
│ • Busca: mayoría, quórum, %...          │
│ • ❌ NO encontrado                       │
└─────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────┐
│ SEGUNDO PASE: Keywords forzadas         │
│ • Busca con términos: "mayoría"         │
│ • Combina con primer pase               │
│ • ✅ Encuentra info específica          │
└─────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────┐
│ CONFIANZA INTELIGENTE:                  │
│ • answer_complete: false                │
│ • confidence: "media"                   │
│ • search_passes: 2                      │
│ ✅ Usuario sabe que puede faltar info   │
└─────────────────────────────────────────┘
```

**Palabras clave supervisadas:**
- Mayoría (mayoría, dos tercios, absoluta, simple)
- Porcentajes (%, porcentaje)
- Quórum
- Aprobación
- Modificación del reglamento

**Campos nuevos en respuesta `/search`:**
```json
{
  "answer_complete": boolean,           // ¿Respuesta completa?
  "confidence": "alta|media|baja",      // Nivel de confianza
  "missing_keywords": boolean,          // ¿Faltan palabras clave?
  "search_passes": 1|2,                 // Pases realizados
  "second_pass_performed": boolean,     // ¿Se activó 2do pase?
  "completeness_score": 0.0-1.0,        // Score (0-100%)
  "completeness_reason": "...",         // Motivo de completitud
  "results": [...]
}
```

**Ejemplo con activación del segundo pase:**
```json
{
  "question": "¿Cuál es la mayoría para modificar?",
  "max_similarity": 0.58,
  "answer_complete": false,
  "confidence": "media",
  "missing_keywords": true,
  "search_passes": 2,
  "second_pass_performed": true,
  "completeness_score": 0.4,
  "completeness_reason": "Faltan palabras clave normativas",
  "results": [...]
}
```

**Ventaja:** 
- ✅ Aumenta **fiabilidad percibida** del sistema
- ✅ Usuario sabe cuándo la respuesta puede estar incompleta
- ✅ Automático: no requiere configuración adicional
- ✅ Mejora UX: transparencia sobre confianza

---

## 📊 Estadísticas de tu ejecución actual

```
📊 Estadísticas finales:
  • Total chunks: 10028
  • Chunks con embedding: 9988
  • Archivos indexados: 161
  • Cobertura de embeddings: 99.6%
```

**40 chunks rechazados** (probablemente por tamaño > 2000 tokens).

---

## 🔧 Cómo usar la reparación manual

Si en una próxima ejecución quieres reparar embeddings sin reindexar todo:

```python
from indexador import update_all_missing_embeddings
update_all_missing_embeddings()
```

O simplemente ejecuta el indexador de nuevo:
```bash
python indexador.py
```

La detección y reparación son automáticas ahora.

---

## ⚙️ Variables de configuración

| Variable | Valor | Descripción |
|----------|-------|------------|
| `MAX_TOKENS_PER_CHUNK` | 2000 | Máximo de tokens por chunk antes de rechazar |
| `CHUNK_SIZE` | 600 | Tamaño base de chunk (caracteres) |
| `CHUNK_OVERLAP` | 150 | Solapamiento entre chunks |
| `MIN_SIMILARITY` | 0.65 | Umbral para confianza alta (verde 🟢) |
| `MIN_SIMILARITY_WARNING` | 0.55 | Umbral para zona gris (amarillo 🟡) |
| `MIN_SIMILARITY_ABSOLUTE` | 0.50 | Umbral de silencio semántico (rojo 🔴) |
| `MIN_SIMILARITY_SECOND_PASS` | 0.45 | Mínimo para activar segundo pase |

Ajusta `MAX_TOKENS_PER_CHUNK` si necesitas ser más estricto (ej: 1500) o más permisivo (ej: 3000).

---

## 🔄 Flujo de confianza en búsquedas (Sistema Semafórico)

```
🟢 max_similarity >= 0.65 + keywords ✅     → confidence: "alta" (verde)
🟡 0.55 <= max_similarity < 0.65           → confidence: "media" (amarillo)
🟠 0.50 <= max_similarity < 0.55           → confidence: "baja" (naranja)
🔴 max_similarity < 0.50                    → confidence: "muy-baja" (rojo - SILENCIO SEMÁNTICO)
```

### 🛑 Regla de Silencio Semántico (< 0.50)

**Por qué 0.50 es la frontera crítica:**

En embeddings normativos:
- **≥ 0.70** → Relación clara y directa
- **0.55 – 0.70** → Relación contextual / parcial (zona gris)
- **0.45 – 0.55** → Ruido semántico peligroso
- **< 0.45** → Coincidencia casi accidental (eco semántico)

**Caso real de silencio semántico:**
```json
{
  "question": "¿Tengo derecho a comer?",
  "max_similarity": 0.404,
  "zone": "very-low",
  "ui_confidence_color": "red",
  "answer": "No se ha encontrado información relacionada..."
}
```

El texto recuperado hablaba de "derecho del socio" y "cuidado de familiares", pero **NO de comida/alimentación**. Responder sería irresponsable.

**Nota especial sobre segundo pase:**
- **0.45-0.50**: Se permite segundo pase (último intento)
- **< 0.45**: Corte seco, ni segundo pase

Si `max_similarity >= 0.45` pero faltan keywords, se activa automáticamente el **segundo pase** para mejorar el resultado.

