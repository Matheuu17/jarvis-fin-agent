import os
import joblib
import numpy as np
import torch
import torch.nn as nn
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

os.makedirs("modelos", exist_ok=True)

ACTIVOS = {
    "IXIC": "^IXIC",       # Nasdaq Composite
    "GSPC": "^GSPC",       # S&P 500
    "DJI": "^DJI",         # Dow Jones
    "EURUSD": "EURUSD=X",  # Euro / Dólar
}

VENTANA = 29
EPOCHS = 60
LR = 0.005


class LSTMFinanciera(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=32, num_layers=2, output_dim=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def crear_secuencias(datos, ventana):
    X, y = [], []
    for i in range(len(datos) - ventana):
        X.append(datos[i : i + ventana])
        y.append(datos[i + ventana])
    return np.array(X), np.array(y)


for clave, ticker in ACTIVOS.items():
    print(f"\n--- Entrenando modelo LSTM para {clave} ({ticker}) ---")
    df = yf.download(ticker, period="2y", interval="1d", progress=False)
    cierres = df["Close"].dropna().values.reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    datos_norm = scaler.fit_transform(cierres)

    X, y = crear_secuencias(datos_norm, VENTANA)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    modelo = LSTMFinanciera()
    criterio = nn.MSELoss()
    optimizador = torch.optim.Adam(modelo.parameters(), lr=LR)

    modelo.train()
    for epoch in range(EPOCHS):
        optimizador.zero_grad()
        pred = modelo(X_tensor)
        loss = criterio(pred, y_tensor)
        loss.backward()
        optimizador.step()

    torch.save(modelo.state_dict(), f"modelos/modelo_{clave}.pt")
    joblib.dump(scaler, f"modelos/scaler_{clave}.pkl")
    print(f"[OK] Guardados: modelos/modelo_{clave}.pt y modelos/scaler_{clave}.pkl (Loss final: {loss.item():.6f})")