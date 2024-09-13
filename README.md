# Procesamiento_de_lenguaje_Natural
Se desea automatizar y elevar el nivel de precisión del proceso de revisión de perfiles de hojas de vida (CV). Para cumplir con este objetico se debe implementar un modelo IA basado en técnicas NLP

En el mundo actual, la eficiencia y precisión en la revisión de perfiles de hojas de vida (CV) son cruciales para identificar rápidamente a los candidatos más adecuados para cualquier posición. Con el objetivo de automatizar y elevar el nivel de precisión de este proceso, proponemos la implementación de un modelo de Inteligencia Artificial (IA) basado en técnicas de Procesamiento de Lenguaje Natural (NLP) y Modelos de Lenguaje de Gran Escala (LLMs).

El proceso de revisión automatizada se centrará en extraer la siguiente información clave de cada hoja de vida:

Nombre completo del candidato
Email o teléfono de contacto
Número total de años de experiencia profesional
Formación en inteligencia artificial (Sí/No)
El modelo de IA analizará cada CV y retornará un conjunto de datos en formato JSON, que incluirá los valores obtenidos para cada especificación técnica, junto con un valor de score que indicará el nivel de precisión o el porcentaje de ajuste del valor obtenido. En los casos donde el modelo no pueda obtener un valor, se registrará un valor nulo con un score de cero (0).

Este enfoque no solo optimizará el tiempo y los recursos dedicados a la revisión de perfiles, sino que también garantizará una mayor precisión en la identificación de candidatos con las competencias necesarias. 


### Instalación de las librerias

* **rarfile**: Esta librería permite trabajar con archivos RAR en Python. Puedes abrir, extraer y listar el contenido de archivos RAR.

* **os**: Es una librería estándar de Python que proporciona una forma de interactuar con el sistema operativo. Puedes realizar operaciones como leer o escribir archivos, manipular rutas y ejecutar comandos del sistema.
  
* **pdfplumber**: Esta librería se utiliza para extraer texto, tablas y metadatos de archivos PDF. Es muy útil para procesar documentos PDF de manera programática.
  
* **re**: Es la librería de expresiones regulares de Python. Permite buscar, dividir y manipular cadenas de texto utilizando patrones definidos.
  
* **spacy**: Es una librería de procesamiento de lenguaje natural (NLP) que ofrece herramientas para el análisis de texto, como el etiquetado de partes del discurso, la lematización y el reconocimiento de entidades nombradas.
  
* **json**: Esta librería estándar de Python se utiliza para trabajar con datos en formato JSON (JavaScript Object Notation). Permite codificar y decodificar datos JSON.
  
* **transformers**: Proporcionada por Hugging Face, esta librería incluye modelos de aprendizaje profundo preentrenados para tareas de NLP, como la traducción, el resumen y la clasificación de texto.
  
* **pandas**: Es una librería poderosa para la manipulación y análisis de datos. Ofrece estructuras de datos como DataFrames, que facilitan el manejo de datos tabulares.


```Pyton
import rarfile
import os
import pdfplumber
import re
import spacy
import json
from transformers import pipeline
import pandas as pd
```
### Parte 1

Este script realiza varias tareas, incluyendo la carga de modelos de procesamiento de lenguaje natural (NLP) y de preguntas y respuestas (QA), así como la extracción de archivos de un archivo RAR. A continuación, se detalla cada sección del código.

**1. Cargar Modelo de spaCy para el Reconocimiento de Entidades**

```Pyton
# Cargar modelo de spaCy para el reconocimiento de entidades
nlp = spacy.load('en_core_web_sm')
```
* Descripción: Carga el modelo en_core_web_sm de spaCy, que es un modelo preentrenado para el reconocimiento de entidades en inglés.
* Uso: Este modelo se utiliza para identificar y clasificar entidades en un texto, como nombres de personas, organizaciones, ubicaciones, etc.

**2. Pipeline para Preguntas y Respuestas**
```Pyton
# Pipeline para preguntas y respuestas (puede ser un modelo general)
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")
```
* Descripción: Crea un pipeline para tareas de preguntas y respuestas utilizando el modelo distilbert-base-uncased-distilled-squad.
* Uso: Este pipeline se utiliza para responder preguntas basadas en un contexto dado.

**3. Ruta del Archivo RAR**
```Pyton
# Ruta del archivo RAR
rar_path = '/content/Nueva carpeta.rar'  # Reemplaza con la ruta correcta
```
* Descripción: Define la ruta del archivo RAR que se desea extraer.
* Nota: Asegúrate de reemplazar '/content/Nueva carpeta.rar' con la ruta correcta del archivo RAR.

**4. Directorio para Extraer los Archivos**
```Pyton
# Directorio para extraer los archivos
extraction_path = 'extracted_cvs'
os.makedirs(extraction_path, exist_ok=True)
```
* Descripción: Define el directorio donde se extraerán los archivos del archivo RAR y crea el directorio si no existe.
* Uso: os.makedirs se asegura de que el directorio de extracción exista antes de intentar extraer los archivos.

**5. Función para Extraer Archivos RAR**
```Pyton   
def extract_rar(rar_path, extraction_path):
    """
    Extrae archivos de un archivo RAR a un directorio especificado.

    Args:
        rar_path (str): Ruta del archivo RAR.
        extraction_path (str): Directorio donde se extraerán los archivos.

    Returns:
        list: Lista de rutas de archivos extraídos.
    """
    with rarfile.RarFile(rar_path) as rf:
        rf.extractall(extraction_path)

    # Listar todos los archivos extraídos, incluyendo subcarpetas
    extracted_files = []
    for root, dirs, files in os.walk(extraction_path):
        for file in files:
            extracted_files.append(os.path.join(root, file))

    return extracted_files
```
* Descripción: Esta función extrae archivos de un archivo RAR a un directorio especificado
  
**Parámetros:**
* rar_path (str): Ruta del archivo RAR.
* extraction_path (str): Directorio donde se extraerán los archivos.
  
* Retorno: Devuelve una lista de rutas de los archivos extraídos.
  
**Detalles:**
* Utiliza rarfile.RarFile para abrir y extraer el contenido del archivo RAR.
* os.walk se utiliza para listar todos los archivos extraídos, incluyendo los que están en subcarpetas.

### Parte 2
Este script realiza la extracción de archivos de un archivo RAR y la lectura de texto de archivos PDF. A continuación, se detalla cada sección del código.

```Pyton
cv_files = extract_rar(rar_path, extraction_path)
print(f"Archivos extraídos: {cv_files}")
```
* Descripción: Utiliza la función extract_rar para extraer archivos de un archivo RAR y luego imprime la lista de archivos extraídos.
* Uso: cv_files: Variable que almacena la lista de archivos extraídos. print(f"Archivos extraídos: {cv_files}"): Imprime la lista de archivos extraídos.

```Pyton
def read_pdf(file_path):
    """
    Lee y extrae el texto de un archivo PDF.

    Args:
        file_path (str): Ruta del archivo PDF.

    Returns:
        str: Texto extraído del archivo PDF.
    """
    text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:  # Asegura que haya texto antes de añadir
                    text += page_text + "\n"
        return text
    except Exception as e:
        print(f"Error al leer {file_path}: {e}")
        return ""

```
* Descripción: Esta función lee y extrae el texto de un archivo PDF.
* Parámetros: file_path (str): Ruta del archivo PDF.
* Retorno: Devuelve el texto extraído del archivo PDF como una cadena de caracteres.
  
**Detalles:**

* text = "": Inicializa una cadena vacía para almacenar el texto extraído.
* with pdfplumber.open(file_path) as pdf: Abre el archivo PDF utilizando pdfplumber.
* for page in pdf.pages: Itera sobre cada página del PDF.
* page_text = page.extract_text(): Extrae el texto de la página.
* if page_text: Verifica si hay texto en la página antes de añadirlo a la cadena text.
* text += page_text + "\n": Añade el texto de la página a la cadena text, seguido de un salto de línea.
* return text: Devuelve el texto extraído.
* except Exception as e: Captura cualquier excepción que ocurra durante la lectura del PDF y imprime un mensaje de error.

### Parte 3

**Función extract_name:**
* Utiliza la biblioteca spaCy para procesar el texto y extraer entidades nombradas.
* Busca entidades etiquetadas como “PERSON” y devuelve el nombre con una puntuación de confianza del 90%.
* Si no encuentra un nombre, devuelve None con una puntuación de 0.

```Pyton
def extract_name(text):
    """Extrae el nombre de una persona del texto utilizando spaCy."""
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return {"value": ent.text, "score": 0.9}
    return {"value": None, "score": 0}
```

**Función extract_contact:**
* Utiliza expresiones regulares para buscar emails y números de teléfono en el texto.
* Devuelve el primer email o número de teléfono encontrado con una puntuación de confianza del 90%.
* Si no encuentra ninguno, devuelve None con una puntuación de 0.

```Pyton
def extract_contact(text):
    """Extrae el email o número de teléfono del texto."""
    email = re.findall(r'\S+@\S+', text)
    phone = re.findall(r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', text)
    if email:
        return {"value": email[0], "score": 0.9}
    elif phone:
        return {"value": phone[0], "score": 0.9}
    else:
        return {"value": None, "score": 0}
```

**Función extract_experience:**
* Utiliza un pipeline de preguntas y respuestas (QA) para estimar los años de experiencia del candidato a partir del texto.
* Devuelve la respuesta y la puntuación de confianza.
* Si ocurre un error, devuelve None con una puntuación de 0.
  
```Pyton
def extract_experience(text):
    """Estima los años de experiencia profesional a partir del texto."""
    try:
        result = qa_pipeline({
            'context': text,
            'question': "How many years of experience does the candidate have?"
        })
        return {"value": result['answer'], "score": result['score']}
    except:
        return {"value": None, "score": 0}

```

**Función check_ai_education:**
* Busca palabras clave relacionadas con IA en el texto.
* Si encuentra alguna, devuelve “S” (Sí) con una puntuación de confianza del 90%.
* Si no encuentra ninguna, devuelve “N” (No) con una puntuación de 90%.
  
```Pyton
def check_ai_education(text):
    """Determina si el candidato tiene formación en IA."""
    keywords = ["artificial intelligence", "machine learning", "deep learning"]
    for keyword in keywords:
        if keyword.lower() in text.lower():
            return {"value": "S", "score": 0.9}
    return {"value": "N", "score": 0.9}
```

**Función Principal**
* Función process_cv:
* Procesa el texto de un CV y utiliza las funciones anteriores para extraer la información especificada.
* Devuelve un diccionario con los resultados
  
```Pyton
def process_cv(cv_text):
    """Procesa el texto de un CV y extrae la información especificada."""
    return {
        "name": extract_name(cv_text),  # Nombre completo del candidato
        "contact": extract_contact(cv_text),  # Email o teléfono de contacto
        "experience_years": extract_experience(cv_text),  # Número de años de experiencia
        "ai_education": check_ai_education(cv_text)  # Formación en IA (S/N)
    }

```

**Procesamiento de Archivos de CV** 

* Itera sobre una lista de archivos de CV (cv_files).
* Si el archivo es un PDF, lo lee y procesa utilizando la función process_cv.
* Almacena los resultados en una lista y los imprime.


```Pyton
results = []

for cv_path in cv_files:
    if cv_path.endswith('.pdf'):
        cv_text = read_pdf(cv_path)
        if cv_text:  # Si se leyó correctamente
            result = process_cv(cv_text)
            results.append(result)
            print(f"Procesado: {cv_path}")

```

**Conversión y almacenamiento de resultados:**
* Convierte los resultados a formato JSON y los imprime.
* Guarda los resultados en un archivo JSON.

```Pyton
results_json = json.dumps(results, indent=4)
print(results_json)

# Guardar el resultado en un archivo JSON
with open('results.json', 'w') as json_file:
    json_file.write(results_json)

```
![image](https://github.com/user-attachments/assets/d829e41a-b918-42c4-abc0-456b3726db0e)


### Fase 3

**Nombre del archivo JSON:**
* Asegúrate de que el nombre del archivo JSON sea correcto. En tu código, has escrito 'resultss.json', pero en el código anterior, el archivo se guardó como 'results.json'. Corrige el nombre del archivo si es necesario.

**Uso de display:**
* La función display es parte de IPython, que se usa comúnmente en Jupyter Notebooks. Si estás ejecutando este código en un entorno diferente, es posible que necesites usar print(data) en su lugar.
Aquí tienes el código corregido y validado:

```Pyton
# Cargar los datos desde el JSON
data = pd.read_json('results.json')  # Asegúrate de que el nombre del archivo sea correcto

# Mostrar el contenido en un DataFrame para una visualización más clara
display(data)  # Si estás en un Jupyter Notebook
# print(data)  # Si estás en otro entorno
```

![image](https://github.com/user-attachments/assets/f084f1e4-10d4-4bf4-b0e4-30aa9ec8705f)


**Conclusión del Proyecto de Automatización y Precisión en la Revisión de Perfiles de Hojas de Vida**

En el presente proyecto, se ha desarrollado e implementado un modelo de inteligencia artificial (IA) basado en técnicas de Procesamiento de Lenguaje Natural (NLP) y Modelos de Lenguaje de Gran Escala (LLMs) con el objetivo de automatizar y elevar el nivel de precisión en el proceso de revisión de perfiles de hojas de vida (CV). Este modelo ha sido diseñado para extraer y validar información clave de cada CV, proporcionando resultados en un formato JSON estructurado.

**Especificaciones Técnicas del Perfil:**

1. Nombre completo del candidato.
2. Email o teléfono de contacto.
3. Número total de años de experiencia profesional.
4. Formación en inteligencia artificial (Sí/No).
   
**Resultados Obtenidos:** 
El modelo IA ha sido capaz de procesar y extraer la información requerida de manera eficiente, generando un conjunto de datos en formato JSON que incluye los valores obtenidos para cada especificación técnica, así como un valor de score que indica el nivel de precisión o el porcentaje de ajuste del valor obtenido. En los casos donde el modelo no pudo obtener un valor, se registró un valor nulo con un score de cero (0).

**Conclusión:**

La implementación de este modelo IA ha demostrado ser una herramienta eficaz para la automatización y precisión en la revisión de perfiles de hojas de vida. La capacidad del modelo para extraer información relevante y proporcionar un score de precisión para cada especificación técnica permite una evaluación más objetiva y consistente de los candidatos. Este avance no solo optimiza el proceso de selección, sino que también reduce significativamente el tiempo y los recursos necesarios para la revisión manual de CVs.




