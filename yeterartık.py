import streamlit as st
import pandas as pd
from ortools.sat.python import cp_model

# =========================
# SAYFA AYARI
# =========================
st.set_page_config(
    page_title="Montaj Hattı Dengeleme",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# CSS TASARIM
# =========================
st.markdown("""
<style>
[data-testid="stSidebar"] {
    background-color: #eef1f6;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #202336;
}

.main-title {
    font-size: 48px;
    font-weight: 800;
    color: #2d3040;
    margin-top: 90px;
}

.subtitle {
    font-size: 18px;
    color: #1f2937;
    margin-top: 10px;
    margin-bottom: 30px;
}

.stButton > button {
    border-radius: 8px;
    padding: 12px 22px;
    font-weight: 600;
    border: 1px solid #d1d5db;
    background-color: white;
}

.stButton > button:hover {
    border-color: #ff4b4b;
    color: #ff4b4b;
}

div[data-testid="stMetricValue"] {
    font-size: 28px;
}

.block-container {
    padding-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

# =========================
# VERİ
# =========================
I = range(1, 64)
J = range(1, 37)
W = range(1, 37)

t_raw = {
    1: 2.43, 2: 9.79, 3: 2.12, 4: 9.92, 5: 4.66, 6: 11.58,
    7: 1.01, 8: 1.44, 9: 9.66, 10: 10.30, 11: 0.49, 12: 7.13,
    13: 7.18, 14: 2.44, 15: 3.58, 16: 4.90, 17: 3.21, 18: 7.78,
    19: 11.27, 20: 11.35, 21: 0.80, 22: 3.31, 23: 9.83, 24: 0.80,
    25: 4.61, 26: 5.20, 27: 11.89, 28: 6.30, 29: 13.32, 30: 0.98,
    31: 14.20, 32: 6.13, 33: 0.98, 34: 14.49, 35: 3.14, 36: 12.12,
    37: 1.07, 38: 5.14, 39: 5.63, 40: 0.57, 41: 10.13, 42: 0.90,
    43: 1.39, 44: 1.43, 45: 0.51, 46: 10.74, 47: 5.65, 48: 7.38,
    49: 1.71, 50: 15.09, 51: 7.31, 52: 6.93, 53: 10.72, 54: 1.31,
    55: 6.45, 56: 2.39, 57: 0.89, 58: 11.06, 59: 8.02, 60: 6.48,
    61: 3.13, 62: 0.53, 63: 7.74
}

SCALE = 100
t = {i: int(round(t_raw[i] * SCALE)) for i in t_raw}
P = [(i, i + 1) for i in range(1, 63)]

# =========================
# SIDEBAR
# =========================
st.sidebar.markdown("## ⚙️ Ayarlar")

with st.sidebar.expander("🏗️ Hat Parametreleri", expanded=True):
    L = st.number_input("Maksimum Yürüme Mesafesi (L)", 1, 20, 4)
    D = st.number_input("Hedef Üretim Miktarı (D)", 1, 100, 32)
    T = st.number_input("Vardiya Süresi (T - dk)", 1, 1000, 510)

with st.sidebar.expander("⚖️ Optimizasyon Kısıtları", expanded=True):
    U_MAX = st.slider("Maks. Operatör Doluluğu (U_MAX)", 0.50, 1.00, 1.00, 0.01)

st.sidebar.markdown("---")

epsilon_choice = st.sidebar.slider(
    "Detaylı Rapor İçin Operatör Seç",
    min_value=1,
    max_value=36,
    value=29
)

time_limit = st.sidebar.slider(
    "Senaryo Başına Süre Limiti (sn)",
    min_value=5,
    max_value=120,
    value=30
)

d = {j: {k: 2 * abs(j - k) for k in J} for j in J}
BIG_M = sum(t.values())

# =========================
# MODEL
# =========================
@st.cache_data(show_spinner=False)
def solve_model(exact_workers, time_limit, L, D, T, U_MAX):
    model = cp_model.CpModel()

    x = {(i, j): model.NewBoolVar(f"x_{i}_{j}") for i in I for j in J}
    y = {(w, j): model.NewBoolVar(f"y_{w}_{j}") for w in W for j in J}
    z = {w: model.NewBoolVar(f"z_{w}") for w in W}

    l = {j: model.NewIntVar(0, BIG_M, f"l_{j}") for j in J}
    q = {(w, j): model.NewIntVar(0, BIG_M, f"q_{w}_{j}") for w in W for j in J}

    C = model.NewIntVar(0, BIG_M, "C")

    for i in I:
        model.Add(sum(x[i, j] for j in J) == 1)

    for i, h in P:
        model.Add(sum(j * x[i, j] for j in J) <= sum(j * x[h, j] for j in J))

    for j in J:
        model.Add(l[j] == sum(t[i] * x[i, j] for i in I))
        model.Add(sum(y[w, j] for w in W) == 1)
        model.Add(l[j] <= C)

    for w in W:
        for j in J:
            model.Add(y[w, j] <= z[w])
            model.Add(q[w, j] <= l[j])
            model.Add(q[w, j] <= BIG_M * y[w, j])
            model.Add(q[w, j] >= l[j] - BIG_M * (1 - y[w, j]))

    for w in W:
        model.Add(sum(q[w, j] for j in J) <= C)

    for w in W:
        for j in J:
            for k in J:
                if j < k and 2 * abs(j - k) > L:
                    model.Add(y[w, j] + y[w, k] <= 1)

    model.Add(sum(z[w] for w in W) == exact_workers)

    model.Minimize(C)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_search_workers = 8

    status = solver.Solve(model)

    if status not in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        return None

    C_value = solver.Value(C) / SCALE

    res = {
        "C": C_value,
        "used_workers": sum(solver.Value(z[w]) for w in W),
        "ops_of_station": {j: [] for j in J},
        "station_loads": {j: solver.Value(l[j]) / SCALE for j in J},
        "stations_of_worker": {w: [] for w in W},
        "worker_load_per_product": {},
        "worker_load_per_shift": {},
        "worker_U": {},
        "reachable_output": T / C_value,
        "meets_target": T / C_value >= D
    }

    for i in I:
        for j in J:
            if solver.Value(x[i, j]) == 1:
                res["ops_of_station"][j].append(i)

    for w in W:
        total = sum(solver.Value(q[w, j]) for j in J) / SCALE
        res["worker_load_per_product"][w] = total
        res["worker_load_per_shift"][w] = D * total
        res["worker_U"][w] = 100 * ((D / T) * total)

        for j in J:
            if solver.Value(y[w, j]) == 1:
                res["stations_of_worker"][w].append(j)

    return res

# =========================
# ARAYÜZ
# =========================
st.markdown(
    '<div class="main-title">🏭 Montaj Hattı Dengeleme & Operatör Atama Sistemi</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Google OR-Tools CP-SAT Solver tabanlı gelişmiş optimizasyon arayüzü.</div>',
    unsafe_allow_html=True
)

run = st.button("🚀 Tüm Senaryoları Hesapla ve Analiz Et")

if run:
    results = {}

    progress = st.progress(0)
    info = st.empty()

    for eps in range(1, 37):
        info.info(f"{eps} operatörlü senaryo çözülüyor...")
        results[eps] = solve_model(eps, time_limit, L, D, T, U_MAX)
        progress.progress(eps / 36)

    info.success("Tüm senaryolar tamamlandı.")

    feasible = [e for e, r in results.items() if r is not None]

    if feasible:
        ideal_C = min(results[e]["C"] for e in feasible)
        ideal_Z = min(results[e]["used_workers"] for e in feasible)
        nadir_C = max(results[e]["C"] for e in feasible)
        nadir_Z = max(results[e]["used_workers"] for e in feasible)

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("İdeal Nokta", f"({ideal_C:.2f}, {ideal_Z:.2f})")
        c2.metric("Nadir Nokta", f"({nadir_C:.2f}, {nadir_Z:.2f})")
        c3.metric("En İyi Üretim", f"{max(results[e]['reachable_output'] for e in feasible):.2f}")
        c4.metric("Hedef", f"{D} adet")

    rows = []

    for eps in range(1, 37):
        r = results[eps]

        if r is None:
            rows.append([eps, "Infeasible", "-", "-", "-"])
        else:
            rows.append([
                f"{eps:.2f}",
                f"{r['C']:.2f}",
                f"{r['used_workers']:.2f}",
                f"{r['reachable_output']:.2f}",
                "Evet" if r["meets_target"] else "Hayır"
            ])

    st.subheader("📊 Epsilon Senaryo Tablosu")

    df = pd.DataFrame(
        rows,
        columns=["Epsilon", "F1 (C)", "F2 (Z)", "Ulaşılabilir Üretim", "Hedef?"]
    )

    st.dataframe(df, use_container_width=True, hide_index=True)

    st.subheader("📋 Detaylı Senaryo Raporu")

    selected = results.get(epsilon_choice)

    if selected is None:
        st.warning(f"{epsilon_choice} operatör için uygun çözüm bulunamadı.")
    else:
        report = []

        report.append("=" * 78)
        report.append(f"DETAYLI SENARYO RAPORU | Operatör Sayısı = {epsilon_choice}")
        report.append("=" * 78)

        report.append(f"Çevrim Süresi (C)                       : {selected['C']:.2f} dk/ürün")
        report.append(f"Kullanılan Operatör Sayısı              : {selected['used_workers']}")
        report.append(f"Maksimum İzin Verilen Operatör Doluluğu : %{U_MAX * 100:.2f}")
        report.append(f"Ulaşılabilir Üretim                     : {selected['reachable_output']:.2f} adet/vardiya")
        report.append(f"Hedef Üretim ({D} adet) Sağlanıyor mu?  : {'Evet' if selected['meets_target'] else 'Hayır'}")

        report.append("\n[1] Operasyon -> İstasyon Atamaları")
        for j in J:
            report.append(f"İstasyon {j}: {selected['ops_of_station'][j]}")

        report.append("\n[2] İstasyon Yükleri")
        for j in J:
            report.append(f"İstasyon {j}: {selected['station_loads'][j]:.2f} dk")

        report.append("\n[3] Operatör -> İstasyon Atamaları")
        for w in W:
            if selected["stations_of_worker"][w]:
                report.append(f"Operatör {w}: {selected['stations_of_worker'][w]}")

        report.append("\n[4] Operatör Toplam Yükleri ve U Değerleri")
        for w in W:
            if selected["stations_of_worker"][w]:
                report.append(
                    f"Operatör {w}: ürün başı yük = "
                    f"{selected['worker_load_per_product'][w]:.2f} dk, "
                    f"vardiya yükü = {selected['worker_load_per_shift'][w]:.2f} dk, "
                    f"U = %{selected['worker_U'][w]:.2f}"
                )

        report.append("\n[5] Mesafe Kontrolü")
        any_pair = False

        for w in W:
            sts = selected["stations_of_worker"][w]
            if len(sts) >= 2:
                for a in range(len(sts)):
                    for b in range(a + 1, len(sts)):
                        j = sts[a]
                        k = sts[b]
                        mesafe = 2 * abs(j - k)
                        durum = "Uygun" if mesafe <= L else "İhlal"
                        any_pair = True
                        report.append(
                            f"Operatör {w}: İstasyon {j}-{k}, "
                            f"mesafe = {mesafe}, durum = {durum}"
                        )

        if not any_pair:
            report.append("Birden fazla istasyona atanmış operatör yok.")

        report.append("=" * 78)

        st.code("\n".join(report), language="text")

else:
    st.empty()
