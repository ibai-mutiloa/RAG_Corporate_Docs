# API de Búsqueda de Normativas

## Descripción
REST API para buscar normativas similares usando embeddings de Azure OpenAI y similitud del coseno.

## Instalación

```bash
pip install -r requirements.txt
```

## Configuración
Asegúrate de que tu archivo `.env` tenga las siguientes variables:

```
# Base de datos
DB_HOST=localhost
DB_PORT=5432
DB_NAME=normativas_db
DB_USER=normativas_user
DB_PASS=tu_password

# Azure OpenAI
AZURE_ENDPOINT=https://tu-recurso.openai.azure.com/
AZURE_API_KEY=tu_api_key
AZURE_DEPLOYMENT_NAME=text-embedding-3-small
AZURE_API_VERSION=2024-12-01-preview

# API Configuration
API_PORT=5000
API_DEBUG=False
TOP_K=5
```

## Ejecución

```bash
python api.py
```

La API estará disponible en `http://localhost:5000`

## Endpoints

### 1. Health Check
Verifica que la API esté funcionando.

**Request:**
```bash
GET /health
```

**Response:**
```json
{
  "status": "ok",
  "message": "API de búsqueda de normativas funcionando correctamente"
}
```

### 2. Buscar Normativas
Busca chunks de normativas similares a una pregunta.
Si el chat conoce el idioma activo del sitio, envíalo en `site_language` para que la búsqueda y la respuesta usen esa rama (`es` o `eu`).

**Request:**
```bash
POST /search
Content-Type: application/json

{
  "question": "¿Cuál es la normativa sobre seguridad laboral?",
  "site_language": "es",
  "top_k": 5
}
```

**Response:**
```json
{
  "question": "¿Cuál es la normativa sobre seguridad laboral?",
  "count": 5,
  "results": [
    {
      "id": 123,
      "file_name": "seguridad_laboral.pdf",
      "file_path": "normativas/seguridad_laboral.pdf",
      "folder_name": "normativas",
      "chunk_index": 0,
      "text": "Texto del chunk relevante...",
      "similarity": 0.95
    },
    ...
  ]
}
```

### 3. Estadísticas
Obtiene estadísticas de la base de datos.

**Request:**
```bash
GET /stats
```

**Response:**
```json
{
  "total_chunks": 1000,
  "chunks_with_embeddings": 950,
  "total_files": 50
}
```

## Ejemplos con curl

### Health Check
```bash
curl http://localhost:5000/health
```

### Búsqueda
```bash
curl -X POST http://localhost:5000/search \
  -H "Content-Type: application/json" \
  -d '{
    "question": "¿Cuál es la normativa sobre vacaciones?",
    "top_k": 3
  }'
```

### Estadísticas
```bash
curl http://localhost:5000/stats
```

## Ejemplos con Python

```python
import requests

# Realizar una búsqueda
response = requests.post(
    'http://localhost:5000/search',
    json={
        'question': '¿Cuáles son los requisitos de seguridad?',
        'top_k': 5
    }
)

if response.status_code == 200:
    data = response.json()
    print(f"Pregunta: {data['question']}")
    print(f"Resultados encontrados: {data['count']}")
    
    for result in data['results']:
        print(f"\nArchivo: {result['file_name']}")
        print(f"Similitud: {result['similarity']:.4f}")
        print(f"Texto: {result['text'][:200]}...")
else:
    print(f"Error: {response.json()}")
```

## Cómo funciona

1. **Recepción de la pregunta**: La API recibe una pregunta en formato JSON.

2. **Cálculo del embedding**: La pregunta se envía a Azure OpenAI para calcular su embedding (vector de 1536 dimensiones).

3. **Primer pase - Búsqueda semántica normal**: 
   - Se reescribe la pregunta para mejorar el matching semántico (variantes con lenguaje jurídico).
   - Se calcula la similitud del coseno entre el embedding de la pregunta y cada chunk:
     ```
     similitud = (A · B) / (||A|| × ||B||)
     ```
   - Los chunks se ordenan por similitud descendente.

4. **Validación de completitud**: 
   - Se verifica si los resultados contienen **palabras clave normativas** esperadas:
     - Mayoría (mayoría, dos tercios, absoluta, simple)
     - Porcentajes (%, porcentaje)
     - Quórum
     - Aprobación
     - Modificación del reglamento
   
5. **Segundo pase (si es necesario)** - Búsqueda con keywords forzadas:
   - **Condición de activación**: `similarity >= MIN_SIMILARITY_WARNING` pero no aparecen keywords esperadas
   - Se relanza la búsqueda con términos forzados: "mayoría", "quórum", "aprobación", "modificación", "reglamento"
   - Se combinan resultados de ambos pases
   - Se re-valida la completitud

6. **Determinación de confianza**:
   - **alta**: similitud >= 0.65 Y contiene keywords normativas
   - **media**: similitud >= 0.55 pero faltan keywords (segundo pase activado)
   - **baja**: similitud < 0.55

7. **Respuesta**: Se retornan los chunks más relevantes con metadatos y nivel de confianza.

## 🎯 Heurística Inteligente (Segundo Pase)

La API implementa una **heurística de dos pases** que mejora significativamente la confiabilidad:

### 🚦 Sistema Semafórico de Confianza

```
🟢 VERDE (>= 0.65)  → Alta confianza
   Respuesta completa con keywords normativas

🟡 AMARILLO (0.55-0.65) → Zona gris  
   Relación contextual/parcial, verificar fuente

🟠 NARANJA (0.50-0.55) → Baja confianza
   Cerca del ruido semántico, consultar directamente

🔴 ROJO (< 0.50) → Silencio semántico
   "No se ha encontrado información relacionada"
   Coincidencia accidental, no es conocimiento real
```

### ¿Por qué el umbral 0.50?

**Ruido semántico peligroso**: Por debajo de 0.50, los embeddings encuentran "ecos semánticos" que NO son conocimiento real.

**Ejemplo libro de texto**:
```
Pregunta: "¿Tengo derecho a comer?"
max_similarity: 0.404
Texto recuperado: "derecho del socio... cuidado de familiares... dependencia..."

❌ No habla de comida
❌ No habla de alimentación  
❌ No habla de derechos básicos

👉 Responder sería IRRESPONSABLE
```

**Respuesta del sistema con < 0.50**:
```json
{
  "confidence": "muy-baja",
  "zone": "very-low",
  "ui_confidence_color": "red",
  "ui_message": "No se ha encontrado información relevante para esta consulta",
  "answer": "No se ha encontrado información relacionada en la normativa consultada..."
}
```

### 🔄 Segundo Pase Inteligente

**Condiciones de activación**:
- `similarity >= 0.45` (por debajo, corte seco)
- Faltan keywords normativas esperadas

**Rango estratégico**:
- **0.45-0.50**: Último intento antes del silencio semántico
- **>= 0.50**: Búsqueda mejorada si faltan keywords

### ¿Por qué es importante?

```
Caso típico:
- Pregunta: "¿Cuál es la mayoría requerida para modificar el reglamento?"
- Resultado con similitud alta (0.68): chunk sobre "procedimientos de votación"
- PROBLEMA: No contiene la palabra "mayoría" específicamente
- SOLUCIÓN: Segundo pase relanza búsqueda con términos forzados
```

### Campos en la respuesta

```json
{
  "question": "...",
  "answer_complete": true|false,          // ¿Contiene info completa?
  "confidence": "alta|media|baja|muy-baja", // Nivel de confianza
  "zone": "high|gray|low|very-low",       // Zona semafórica
  "ui_confidence_color": "green|yellow|orange|red", // Color para UI
  "ui_message": "...",                    // Mensaje para mostrar al usuario
  "missing_keywords": true|false,         // ¿Faltan palabras clave?
  "search_passes": 1|2,                   // Cuántos pases se realizaron
  "second_pass_performed": true|false,    // ¿Se activó el segundo pase?
  "completeness_score": 0.0-1.0,          // Score de completitud
  "results": [...]
}
```

### Ejemplo de respuesta con segundo pase activado

```json
{
  "question": "¿Cuál es la mayoría necesaria para aprobar cambios?",
  "max_similarity": 0.58,
  "confidence": "media",
  "zone": "gray",
  "ui_confidence_color": "yellow",
  "ui_message": "Información encontrada con confianza media - verificar con fuente oficial",
  "answer_complete": false,
  "missing_keywords": true,
  "search_passes": 2,
  "second_pass_performed": true,
  "completeness_score": 0.4,
  "completeness_reason": "Faltan palabras clave normativas",
  "results": [...]
}
```

### Ejemplo de silencio semántico (< 0.50)

```json
{
  "question": "¿Tengo derecho a comer?",
  "max_similarity": 0.404,
  "confidence": "muy-baja",
  "zone": "very-low",
  "ui_confidence_color": "red",
  "ui_message": "No se ha encontrado información relevante para esta consulta",
  "answer": "No se ha encontrado información relacionada en la normativa consultada. La consulta no tiene suficiente relación semántica con los documentos disponibles.",
  "answer_complete": false,
  "search_passes": 1,
  "second_pass_performed": false,
  "results": []
}
```

En este caso, el sistema **transparentemente informa** que la respuesta podría estar incompleta o que directamente no hay información relevante, mejorando la confianza del usuario en el sistema.

## Notas

- Asegúrate de que el indexador (`indexador.py`) haya procesado los PDFs y calculado los embeddings antes de usar la API.
- La similitud del coseno devuelve valores entre -1 y 1, donde 1 indica máxima similitud.
- El parámetro `top_k` controla cuántos resultados se retornan (por defecto 5).
- La heurística de segundo pase se activa automáticamente cuando es apropiado, sin necesidad de configuración adicional.
