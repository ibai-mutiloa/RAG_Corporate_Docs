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

**Request:**
```bash
POST /search
Content-Type: application/json

{
  "question": "¿Cuál es la normativa sobre seguridad laboral?",
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

3. **Búsqueda por similitud del coseno**: 
   - Se obtienen todos los chunks de la base de datos que tienen embeddings.
   - Se calcula la similitud del coseno entre el embedding de la pregunta y cada chunk:
     ```
     similitud = (A · B) / (||A|| × ||B||)
     ```
   - Los chunks se ordenan por similitud descendente.

4. **Respuesta**: Se retornan los top K chunks más similares con sus metadatos.

## Notas

- Asegúrate de que el indexador (`indexador.py`) haya procesado los PDFs y calculado los embeddings antes de usar la API.
- La similitud del coseno devuelve valores entre -1 y 1, donde 1 indica máxima similitud.
- El parámetro `top_k` controla cuántos resultados se retornan (por defecto 5).
