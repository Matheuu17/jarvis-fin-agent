import json
import os
import traceback
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from groq import Groq

from predictor import analizar_activo

load_dotenv()

app = Flask(__name__)

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    print("ADVERTENCIA: No se encontró GROQ_API_KEY en el archivo .env")

client = Groq(api_key=api_key)


def obtener_modelo_groq() -> str:
    try:
        modelos_cuenta = [m.id for m in client.models.list().data]
        preferidos = [
            "meta-llama/llama-4-maverick-17b-128e-instruct",
            "meta-llama/llama-4-scout-17b-16e-instruct",
            "openai/gpt-oss-120b",
            "qwen/qwen3-32b",
            "llama3-70b-8192",
        ]
        for candidato in preferidos:
            if candidato in modelos_cuenta:
                return candidato
        for m_id in modelos_cuenta:
            if "whisper" not in m_id and "guard" not in m_id and "tts" not in m_id:
                return m_id
    except Exception:
        pass
    return "meta-llama/llama-4-scout-17b-16e-instruct"


MODELO_ACTIVO = obtener_modelo_groq()
print(f"[GROQ] Modelo activo: {MODELO_ACTIVO}")

SYSTEM_INSTRUCTION = """
Eres J.A.R.V.I.S., un agente inteligente especializado en análisis financiero cuantitativo.
Responde siempre en español, de forma concisa, directa y con tono analítico profesional (máximo 2 o 3 oraciones, pensado para ser leído por voz).
Tienes 4 modelos neuronales LSTM entrenados en PyTorch:
- IXIC: Nasdaq Composite
- GSPC: S&P 500
- DJI: Dow Jones Industrial
- EURUSD: Par de divisas Euro / Dólar

SOLO invoca la herramienta 'consultar_modelo_financiero' cuando el usuario pida analizar, predecir o consultar recomendaciones sobre alguno de estos activos o el mercado.
Si el usuario hace preguntas conceptuales sobre inversiones, gestión de riesgo o Deep Learning, responde con tu propio conocimiento sin llamar herramientas.
Cuando uses la herramienta, cita los valores reales devueltos y aclara brevemente que es una proyección estadística del modelo LSTM.
"""

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "consultar_modelo_financiero",
            "description": "Ejecuta la red neuronal LSTM correspondiente en PyTorch para predecir el siguiente cierre de un activo financiero y emitir una señal.",
            "parameters": {
                "type": "object",
                "properties": {
                    "activo_id": {
                        "type": "string",
                        "enum": ["IXIC", "GSPC", "DJI", "EURUSD"],
                        "description": "Identificador del activo: IXIC (Nasdaq), GSPC (S&P 500), DJI (Dow Jones) o EURUSD (Euro/Dólar).",
                    },
                    "umbral_pct": {
                        "type": "number",
                        "description": "Porcentaje mínimo de variación para emitir señal de compra o venta (por defecto 0.25).",
                    },
                },
                "required": ["activo_id"],
            },
        },
    }
]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "").strip()

    if not user_message:
        return jsonify({"reply": "No recibí ninguna instrucción.", "tool_used": False})

    messages = [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": user_message},
    ]

    try:
        ultimo_market_data = None
        activos_consultados = []

        # Bucle agéntico: permite 1, 2 o más consultas encadenadas a PyTorch
        for _ in range(5):
            response = client.chat.completions.create(
                model=MODELO_ACTIVO,
                messages=messages,
                tools=TOOLS_SCHEMA,
                tool_choice="auto",
                temperature=0.3,
            )

            response_message = response.choices[0].message
            tool_calls = response_message.tool_calls

            # Si ya no pide más herramientas, devolvemos la respuesta final
            if not tool_calls:
                return jsonify({
                    "reply": response_message.content or "Análisis completado.",
                    "tool_used": len(activos_consultados) > 0,
                    "activos_consultados": activos_consultados,
                    "market_data": ultimo_market_data,
                })

            # Agregamos el turno del asistente de forma limpia
            messages.append({
                "role": "assistant",
                "content": response_message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ],
            })

            # Ejecutamos cada modelo LSTM solicitado
            for tool_call in tool_calls:
                if tool_call.function.name == "consultar_modelo_financiero":
                    args = json.loads(tool_call.function.arguments or "{}")
                    activo_id = args.get("activo_id", "IXIC")
                    umbral = args.get("umbral_pct", 0.25)

                    resultado = analizar_activo(activo_id=activo_id, umbral_pct=umbral)
                    ultimo_market_data = resultado
                    if resultado["activo"] not in activos_consultados:
                        activos_consultados.append(resultado["activo"])

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": "consultar_modelo_financiero",
                        "content": json.dumps(resultado),
                    })

        return jsonify({
            "reply": "Se alcanzó el límite de iteraciones del agente.",
            "tool_used": len(activos_consultados) > 0,
            "activos_consultados": activos_consultados,
            "market_data": ultimo_market_data,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "reply": f"Error en el núcleo del agente: {str(e)}",
            "tool_used": False,
            "market_data": None,
        }), 500


if __name__ == "__main__":
    print("Iniciando J.A.R.V.I.S. Multi-Activo en http://127.0.0.1:5000 ...")
    app.run(debug=True, port=5000)  