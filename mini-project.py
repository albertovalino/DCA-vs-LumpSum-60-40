import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

df = pd.read_excel("cartera_60_40.xlsx", sheet_name="MERGE")
# Convierte "YYYY-MM" -> fin de mes (o inicio si prefieres)
df["date_m"] = pd.to_datetime(df["date_m"], format="%Y-%m")\
                .dt.to_period("M")\
                .dt.to_timestamp(how="end")   # pon how="start" si quieres inicio

# --- sanity checks ---
df = df.sort_values("date_m").reset_index(drop=True)
assert df[["equity_close","bond_close"]].notna().all().all()

# --- 1) Retornos mensuales ---
df["r_eq"] = df["equity_close"].pct_change()
df["r_bd"] = df["bond_close"].pct_change()

# --- 2) Cartera 60/40 Lump Sum (rebalanceo anual en enero) ---
V0 = 100.0
E = np.empty(len(df)); B = np.empty(len(df)); V = np.empty(len(df))
E[0], B[0] = 0.6*V0, 0.4*V0
V[0] = V0
for t in range(1, len(df)):
    E[t] = E[t-1]*(1 + df.loc[t,"r_eq"])
    B[t] = B[t-1]*(1 + df.loc[t,"r_bd"])
    V[t] = E[t] + B[t]
    if df.loc[t,"date_m"].month == 1: # rebalanceo anual en enero
        E[t], B[t] = 0.6*V[t], 0.4*V[t]

df["V_LS"]  = V
df["r_LS"]  = df["V_LS"].pct_change()
roll_max  = np.maximum.accumulate(df["V_LS"].ffill().values)
df["DD_LS"] = df["V_LS"]/roll_max - 1.0

# --- 3) Cartera 60/40 DCA (aportar 1 cada mes, sin rebalanceo) ---
c = 1.0
E2 = np.empty(len(df)); B2 = np.empty(len(df)); K = np.empty(len(df))
E2[0] = 0.6*c*(1 + (0 if np.isnan(df.loc[0,"r_eq"]) else df.loc[0,"r_eq"]))
B2[0] = 0.4*c*(1 + (0 if np.isnan(df.loc[0,"r_bd"]) else df.loc[0,"r_bd"]))
K[0]  = E2[0] + B2[0]
for t in range(1, len(df)):
    re = 0 if np.isnan(df.loc[t,"r_eq"]) else df.loc[t,"r_eq"]
    rb = 0 if np.isnan(df.loc[t,"r_bd"]) else df.loc[t,"r_bd"]
    E2[t] = E2[t-1]*(1+re) + 0.6*c
    B2[t] = B2[t-1]*(1+rb) + 0.4*c
    K[t]  = E2[t] + B2[t]

df["V_DCA"] = K
# DD de DCA (opcional)
roll_max2 = np.maximum.accumulate(df["V_DCA"].ffill().values)
df["DD_DCA"]= df["V_DCA"]/roll_max2 - 1.0

# --- 4) Métricas ---
n = df["r_LS"].dropna().shape[0]
years = (df["date_m"].iloc[-1] - df["date_m"].iloc[0]).days/365.25
CAGR_LS = (df["V_LS"].iloc[-1]/V0)**(1/years) - 1
VOL_LS  = df["r_LS"].dropna().std()*np.sqrt(12)
MDD_LS  = df["DD_LS"].min()

# prob. de perder a 12 meses (LS)
roll_12 = (1+df["r_LS"].dropna()).rolling(12).apply(np.prod, raw=True) - 1
loss_12_prob = (roll_12 < 0).mean() if len(roll_12.dropna())>0 else np.nan

# DCA "CAGR" aproximado (sobre capital aportado)
months = len(df)
CAGR_DCA = (df["V_DCA"].iloc[-1]/(months*c))**(12/months) - 1
MDD_DCA  = df["DD_DCA"].min()

# --- 5) Gráficos (matplotlib, uno por figura, sin estilos ni colores) ---
out = Path("outputs"); out.mkdir(exist_ok=True)

# (i) Equity curve
plt.figure()
plt.plot(df["date_m"], df["V_LS"], label="Lump Sum")
plt.plot(df["date_m"], df["V_DCA"], label="DCA")
plt.title("60/40: Lump Sum vs DCA (valor acumulado)")
plt.xlabel("Fecha"); plt.ylabel("Valor")
plt.legend()
plt.tight_layout(); plt.savefig(out/"equity_curve.png", dpi=150); plt.close()

# (ii) Drawdown (LS)
plt.figure()
plt.plot(df["date_m"], df["DD_LS"])
plt.title("Drawdown – Lump Sum (60/40)")
plt.xlabel("Fecha"); plt.ylabel("Drawdown")
plt.tight_layout(); plt.savefig(out/"drawdown_ls.png", dpi=150); plt.close()

# --- 6) 3 insights (imprime y guarda) ---
insights = [
    f"1) DCA reduce el drawdown máximo de {MDD_LS:.1%} (LS) a {MDD_DCA:.1%} con un CAGR aprox de {CAGR_DCA:.2%}.",
    f"2) El CAGR de Lump Sum es {CAGR_LS:.2%} con vol anualizada {VOL_LS:.2%}.",
    f"3) Probabilidad de perder en ventanas de 12 meses (LS): {loss_12_prob:.1%}."
]
print("\nINSIGHTS"); print("\n".join(insights))
(Path(out/"report.md")).write_text(
    "## Insights\n" + "\n".join(f"- {s}" for s in insights) +
    "\n\n## Gráficos\n- equity_curve.png\n- drawdown_ls.png\n",
    encoding="utf-8"
)
print(f"\nExportado en: {out.resolve()}")
