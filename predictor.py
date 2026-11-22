import joblib
import numpy as np
import torch
import torch.nn as nn
import yfinance as yf


class LSTMFinanciera(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=32, num_layers=2, output_dim=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


ACTIVOS_INFO = {
    "IXIC": {"ticker": "^IXIC", "nombre": "Nasdaq Composite (^IXIC)", "decimales": 2},
    "GSPC": {"ticker": "^GSPC", "nombre": "S&P 500 (^GSPC)", "decimales": 2},
    "DJI": {"ticker": "^DJI", "nombre": "Dow Jones Industrial (^DJI)", "decimales": 2},
    "EURUSD": {"ticker": "EURUSD=X", "nombre": "Euro / Dólar (EUR/USD)", "decimales": 4},
}

MODELOS_CARGADOS = {}
SCALERS_CARGADOS = {}

for clave in ACTIVOS_INFO:
    modelo = LSTMFinanciera()
    modelo.load_state_dict(torch.load(f"modelos/modelo_{clave}.pt", map_location="cpu", weights_only=True))
    modelo.eval()
    MODELOS_CARGADOS[clave] = modelo
    SCALERS_CARGADOS[clave] = joblib.load(f"modelos/scaler_{clave}.pkl")


def analizar_activo(activo_id: str = "IXIC", umbral_pct: float = 0.25) -> dict:
    activo_id = activo_id.upper().replace("^", "").replace("=X", "")
    if activo_id not in ACTIVOS_INFO:
        activo_id = "IXIC"

    info = ACTIVOS_INFO[activo_id]
    modelo = MODELOS_CARGADOS[activo_id]
    scaler = SCALERS_CARGADOS[activo_id]
    dec = info["decimales"]

    df = yf.download(info["ticker"], period="3mo", interval="1d", progress=False)
    cierres = df["Close"].dropna().values[-29:].reshape(-1, 1)

    ultimo_cierre = float(cierres[-1][0])
    ventana_norm = scaler.transform(cierres)
    tensor_entrada = torch.tensor(ventana_norm, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        pred_norm = modelo(tensor_entrada).numpy()

    proyeccion = float(scaler.inverse_transform(pred_norm)[0][0])
    variacion_pct = ((proyeccion - ultimo_cierre) / ultimo_cierre) * 100

    # Para divisas (EUR/USD) un umbral de 0.15% es más realista que en índices
    umbral_efectivo = 0.15 if activo_id == "EURUSD" else umbral_pct

    if variacion_pct >= umbral_efectivo:
        senal = "COMPRAR"
    elif variacion_pct <= -umbral_efectivo:
        senal = "VENDER"
    else:
        senal = "MANTENER"

    return {
        "activo_id": activo_id,
        "activo": info["nombre"],
        "ticker": info["ticker"],
        "ultimo_cierre": round(ultimo_cierre, dec),
        "proyeccion": round(proyeccion, dec),
        "variacion_pct": round(variacion_pct, 2),
        "senal": senal,
        "decimales": dec,
    }