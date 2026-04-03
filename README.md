# 🎵 Wrapped Analytics

Asistente conversacional para explorar el historial de escucha de Spotify mediante lenguaje natural.

## ¿Qué hace?

El usuario escribe una pregunta en lenguaje natural ("¿Cuál es mi artista más escuchado?") y la app genera automáticamente una visualización como respuesta.

## Tecnologías

- [Streamlit](https://streamlit.io) — interfaz web
- [OpenAI GPT-4.1-mini](https://openai.com) — generación de código
- [Plotly](https://plotly.com) — visualizaciones
- [pandas](https://pandas.pydata.org) — manipulación de datos

## Arquitectura

La app usa una arquitectura *text-to-code*: el LLM no recibe los datos, sino la estructura del dataset. Genera código Python que se ejecuta localmente sobre el DataFrame real.

## Puesta en marcha

1. Clona el repositorio
2. Crea un entorno virtual y actívalo
3. Instala dependencias: `pip install -r requirements.txt`
4. Copia `.streamlit/secrets.toml.example` como `.streamlit/secrets.toml` y rellena la API key y la contraseña
5. Ejecuta: `streamlit run app.py`
