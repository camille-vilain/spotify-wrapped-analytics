# ============================================================
# CABECERA
# ============================================================
# Alumno: Nombre Apellido
# URL Streamlit Cloud: https://...streamlit.app
# URL GitHub: https://github.com/...

# ============================================================
# IMPORTS
# ============================================================
# Streamlit: framework para crear la interfaz web
# pandas: manipulación de datos tabulares
# plotly: generación de gráficos interactivos
# openai: cliente para comunicarse con la API de OpenAI
# json: para parsear la respuesta del LLM (que llega como texto JSON)
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from openai import OpenAI
import json

# ============================================================
# CONSTANTES
# ============================================================
# Modelo de OpenAI. No lo cambies.
MODEL = "gpt-4.1-mini"

# -------------------------------------------------------
# >>> SYSTEM PROMPT — TU TRABAJO PRINCIPAL ESTÁ AQUÍ <<<
# -------------------------------------------------------
# El system prompt es el conjunto de instrucciones que recibe el LLM
# ANTES de la pregunta del usuario. Define cómo se comporta el modelo:
# qué sabe, qué formato debe usar, y qué hacer con preguntas inesperadas.
#
# Puedes usar estos placeholders entre llaves — se rellenan automáticamente
# con información real del dataset cuando la app arranca:
#   {fecha_min}             → primera fecha del dataset
#   {fecha_max}             → última fecha del dataset
#   {plataformas}           → lista de plataformas (Android, iOS, etc.)
#   {reason_start_values}   → valores posibles de reason_start
#   {reason_end_values}     → valores posibles de reason_end
#
# IMPORTANTE: como el prompt usa llaves para los placeholders,
# si necesitas escribir llaves literales en el texto (por ejemplo para
# mostrar un JSON de ejemplo), usa doble llave: {{ y }}
#
SYSTEM_PROMPT = """
Eres un asistente analítico especializado en datos de escucha de Spotify.
Tu única función es analizar el historial de escucha de un usuario y responder
preguntas sobre sus hábitos musicales.

════════════════════════════════════════════════
CONTEXTO DEL DATASET
════════════════════════════════════════════════

El DataFrame `df` ya está cargado en memoria. Contiene reproducciones de
{fecha_min} a {fecha_max}. Estas son TODAS las columnas disponibles:

  Columnas originales:
    ts                (datetime, UTC) — timestamp de fin de reproducción
    ms_played         (int)           — milisegundos reproducidos
    reason_start      (str)           — valores posibles: {reason_start_values}
    reason_end        (str)           — valores posibles: {reason_end_values}
    spotify_track_uri (str)           — URI único de la canción
    platform          (str)           — plataformas posibles: {plataformas}
    shuffle           (bool)          — True si shuffle estaba activado
    skipped           (bool)          — True si se saltó (False si no)

  Columnas derivadas (ya calculadas, úsalas directamente):
    artist            (str)   — nombre del artista
    track             (str)   — nombre de la canción
    album             (str)   — nombre del álbum
    date              (date)  — solo la fecha
    year              (int)   — año
    month             (int)   — mes numérico 1-12
    month_label       (str)   — "2023-04" (para ejes temporales ordenados)
    weekday           (int)   — día de semana: 0=lunes, 6=domingo
    weekday_name      (str)   — "Monday", "Tuesday"…
    hour              (int)   — hora del día 0-23
    is_weekend        (bool)  — True si sábado o domingo
    minutes_played    (float) — minutos reproducidos
    hours_played      (float) — horas reproducidas
    semester          (str)   — "H1" (ene-jun) o "H2" (jul-dic)
    season            (str)   — "Invierno", "Primavera", "Verano", "Otoño"

NOTA: Ya se han filtrado reproducciones menores de 30 segundos.

════════════════════════════════════════════════
INSTRUCCIONES DE RESPUESTA
════════════════════════════════════════════════

Responde SIEMPRE con un JSON válido. Sin markdown, sin explicaciones fuera
del JSON. Exactamente uno de estos dos formatos:

Si la pregunta es sobre el historial de escucha:
{{
  "tipo": "grafico",
  "codigo": "...código Python que genera una figura Plotly en la variable fig...",
  "interpretacion": "...una frase breve explicando el resultado..."
}}

Si la pregunta NO es sobre el historial de escucha:
{{
  "tipo": "fuera_de_alcance",
  "codigo": "",
  "interpretacion": "Esta pregunta está fuera del alcance del asistente. Pregúntame sobre tu historial de escucha de Spotify."
}}

════════════════════════════════════════════════
INSTRUCCIONES PARA EL CÓDIGO
════════════════════════════════════════════════

1. Usa SIEMPRE Plotly. Nunca matplotlib, seaborn ni otras librerías.
   Disponibles: px, go, pd, df.

2. La variable final SIEMPRE debe llamarse `fig`.

3. Selección del tipo de gráfico:
   - Rankings / top N             → px.bar horizontal, ordenado descendente
   - Evolución temporal (meses)   → px.line con month_label en eje X
   - Distribución por hora/día    → px.bar vertical
   - Comparación entre grupos     → px.bar con barmode="group"
   - Porcentajes / proporciones   → px.pie o px.bar con etiqueta %

4. Estilo obligatorio:
   - Título descriptivo en cada gráfico
   - Etiquetas en ejes X e Y
   - Valores visibles sobre las barras cuando sea posible

5. Para top N sin especificar, usa N=10.

6. Para días de la semana, ordena lunes→domingo (weekday 0→6).

7. Para ejes temporales usa month_label y ordena por ese campo.

════════════════════════════════════════════════
EJEMPLOS
════════════════════════════════════════════════

Pregunta: "¿Cuáles son mis 5 artistas más escuchados en horas?"
Respuesta:
{{
  "tipo": "grafico",
  "codigo": "top = df.groupby('artist')['hours_played'].sum().nlargest(5).reset_index().sort_values('hours_played'); fig = px.bar(top, x='hours_played', y='artist', orientation='h', title='Top 5 artistas más escuchados', labels={{'hours_played': 'Horas', 'artist': 'Artista'}})",
  "interpretacion": "Tu artista más escuchado acumula más horas que los siguientes combinados."
}}

Pregunta: "¿Cuál es la capital de Francia?"
Respuesta:
{{
  "tipo": "fuera_de_alcance",
  "codigo": "",
  "interpretacion": "Esta pregunta está fuera del alcance del asistente. Pregúntame sobre tu historial de escucha de Spotify."
}}
"""


# ============================================================
# CARGA Y PREPARACIÓN DE DATOS
# ============================================================
# Esta función se ejecuta UNA SOLA VEZ gracias a @st.cache_data.
# Lee el fichero JSON y prepara el DataFrame para que el código
# que genere el LLM sea lo más simple posible.
#
@st.cache_data
def load_data():
    df = pd.read_json("streaming_history.json")

# Timestamps
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df["date"]         = df["ts"].dt.date
    df["year"]         = df["ts"].dt.year
    df["month"]        = df["ts"].dt.month
    df["month_label"]  = df["ts"].dt.strftime("%Y-%m")
    df["weekday"]      = df["ts"].dt.weekday
    df["weekday_name"] = df["ts"].dt.day_name()
    df["hour"]         = df["ts"].dt.hour
    df["is_weekend"]   = df["weekday"].isin([5, 6])

    # Tiempo en unidades útiles
    df["minutes_played"] = df["ms_played"] / 60_000
    df["hours_played"]   = df["ms_played"] / 3_600_000

    # Alias de columnas largas
    df["artist"] = df["master_metadata_album_artist_name"]
    df["track"]  = df["master_metadata_track_name"]
    df["album"]  = df["master_metadata_album_album_name"]

    # Filtrar reproducciones irrelevantes (<30 segundos)
    df = df[df["ms_played"] >= 30_000].copy()

    # Semestre y estación
    df["semester"] = df["month"].apply(lambda m: "H1" if m <= 6 else "H2")

    def get_season(m):
        if m in [12, 1, 2]: return "Invierno"
        if m in [3, 4, 5]:  return "Primavera"
        if m in [6, 7, 8]:  return "Verano"
        return "Otoño"
    df["season"] = df["month"].apply(get_season)

    # Normalizar booleanos
    df["skipped"] = df["skipped"].fillna(False).astype(bool)
    df["shuffle"]  = df["shuffle"].astype(bool)

    return df


def build_prompt(df):
    """
    Inyecta información dinámica del dataset en el system prompt.
    Los valores que calcules aquí reemplazan a los placeholders
    {fecha_min}, {fecha_max}, etc. dentro de SYSTEM_PROMPT.

    Si añades columnas nuevas en load_data() y quieres que el LLM
    conozca sus valores posibles, añade aquí el cálculo y un nuevo
    placeholder en SYSTEM_PROMPT.
    """
    fecha_min = df["ts"].min()
    fecha_max = df["ts"].max()
    plataformas = df["platform"].unique().tolist()
    reason_start_values = df["reason_start"].unique().tolist()
    reason_end_values = df["reason_end"].unique().tolist()

    return SYSTEM_PROMPT.format(
        fecha_min=fecha_min,
        fecha_max=fecha_max,
        plataformas=plataformas,
        reason_start_values=reason_start_values,
        reason_end_values=reason_end_values,
    )


# ============================================================
# FUNCIÓN DE LLAMADA A LA API
# ============================================================
# Esta función envía DOS mensajes a la API de OpenAI:
# 1. El system prompt (instrucciones generales para el LLM)
# 2. La pregunta del usuario
#
# El LLM devuelve texto (que debería ser un JSON válido).
# temperature=0.2 hace que las respuestas sean más predecibles.
#
# No modifiques esta función.
#
def get_response(user_msg, system_prompt):
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content


# ============================================================
# PARSING DE LA RESPUESTA
# ============================================================
# El LLM devuelve un string que debería ser un JSON con esta forma:
#
#   {"tipo": "grafico",          "codigo": "...", "interpretacion": "..."}
#   {"tipo": "fuera_de_alcance", "codigo": "",    "interpretacion": "..."}
#
# Esta función convierte ese string en un diccionario de Python.
# Si el LLM envuelve el JSON en backticks de markdown (```json...```),
# los limpia antes de parsear.
#
# No modifiques esta función.
#
def parse_response(raw):
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

    return json.loads(cleaned)


# ============================================================
# EJECUCIÓN DEL CÓDIGO GENERADO
# ============================================================
# El LLM genera código Python como texto. Esta función lo ejecuta
# usando exec() y busca la variable `fig` que el código debe crear.
# `fig` debe ser una figura de Plotly (px o go).
#
# El código generado tiene acceso a: df, pd, px, go.
#
# No modifiques esta función.
#
def execute_chart(code, df):
    local_vars = {"df": df, "pd": pd, "px": px, "go": go}
    exec(code, {}, local_vars)
    return local_vars.get("fig")


# ============================================================
# INTERFAZ STREAMLIT
# ============================================================
# Toda la interfaz de usuario. No modifiques esta sección.
#

# Configuración de la página
st.set_page_config(page_title="Spotify Analytics", layout="wide")

# --- Control de acceso ---
# Lee la contraseña de secrets.toml. Si no coincide, no muestra la app.
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("🔒 Acceso restringido")
    pwd = st.text_input("Contraseña:", type="password")
    if pwd:
        if pwd == st.secrets["PASSWORD"]:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Contraseña incorrecta.")
    st.stop()

# --- App principal ---
st.title("🎵 Spotify Analytics Assistant")
st.caption("Pregunta lo que quieras sobre tus hábitos de escucha")

# Cargar datos y construir el prompt con información del dataset
df = load_data()
system_prompt = build_prompt(df)

# Caja de texto para la pregunta del usuario
if prompt := st.chat_input("Ej: ¿Cuál es mi artista más escuchado?"):

    # Mostrar la pregunta en la interfaz
    with st.chat_message("user"):
        st.write(prompt)

    # Generar y mostrar la respuesta
    with st.chat_message("assistant"):
        with st.spinner("Analizando..."):
            try:
                # 1. Enviar pregunta al LLM
                raw = get_response(prompt, system_prompt)

                # 2. Parsear la respuesta JSON
                parsed = parse_response(raw)

                if parsed["tipo"] == "fuera_de_alcance":
                    # Pregunta fuera de alcance: mostrar solo texto
                    st.write(parsed["interpretacion"])
                else:
                    # Pregunta válida: ejecutar código y mostrar gráfico
                    fig = execute_chart(parsed["codigo"], df)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                        st.write(parsed["interpretacion"])
                    else:
                        st.warning("El código no produjo ninguna visualización. Intenta reformular la pregunta.")
                        st.code(parsed["codigo"], language="python")

            except json.JSONDecodeError:
                st.error("No he podido interpretar la respuesta. Intenta reformular la pregunta.")
            except Exception as e:
                st.error("Ha ocurrido un error al generar la visualización. Intenta reformular la pregunta.")


# ============================================================
# REFLEXIÓN TÉCNICA (máximo 30 líneas)
# ============================================================
#
# Responde a estas tres preguntas con tus palabras. Sé concreto
# y haz referencia a tu solución, no a generalidades.
# No superes las 30 líneas en total entre las tres respuestas.
#
# 1. ARQUITECTURA TEXT-TO-CODE
#    ¿Cómo funciona la arquitectura de tu aplicación? ¿Qué recibe
#    el LLM? ¿Qué devuelve? ¿Dónde se ejecuta el código generado?
#    ¿Por qué el LLM no recibe los datos directamente?
#
#    La app nunca envía los datos a la LLM. Le manda el system prompt
#    (estructura del DataFrame) y la pregunta del usuario. La LLM devuelve
#    un JSON con tres campos: tipo, codigo e interpretacion. Ese código
#    se ejecuta localmente con exec() sobre el DataFrame real y produce
#    la variable "fig" que Streamlit dibuja. No mandamos las 15000 filas
#    porque ocuparían millones de tokens, no cabrían en el contexto del
#    modelo, y no es necesario: la LLM solo necesita la estructura.
#
# 2. EL SYSTEM PROMPT COMO PIEZA CLAVE
#    ¿Qué información le das al LLM y por qué? Pon un ejemplo
#    concreto de una pregunta que funciona gracias a algo específico
#    de tu prompt, y otro de una que falla o fallaría si quitases
#    una instrucción.
#
#    El prompt describe las columnas disponibles, sus tipos, las librerías
#    del exec() y el formato JSON de respuesta. Sin la lista de columnas,
#    el LLM inventaría nombres y el exec() fallaría. Por ejemplo,
#    "¿A qué hora escuchas?" funciona porque el prompt especifica la columna
#    "hour" ya calculada. Sin el guardrail, "¿Capital de Francia?"
#    generaría código sin sentido que fallaría en ejecución.
#
# 3. EL FLUJO COMPLETO
#    Describe paso a paso qué ocurre desde que el usuario escribe
#    una pregunta hasta que ve el gráfico en pantalla.
#
#    1. load_data() prepara el DataFrame una sola vez (timestamps, columnas
#       derivadas, filtro <30s). 
#    2. El usuario escribe una pregunta.
#    3. get_response() manda system prompt + pregunta a gpt-4.1-mini y recibe
#       un JSON. 
#    4. parse_response() lo convierte en diccionario. 
#    5. Si tipo es "gráfico", execute_chart() ejecuta el código con exec() y 
#       extrae "fig".
#    6. st.plotly_chart() redibuja el gráfico con la interpretación en texto.