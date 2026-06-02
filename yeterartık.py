import streamlit as st
import pandas as pd
from ortools.sat.python import cp_model

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

d = {j: {k: 2 * abs(j - k) for k in J} for j in J}

L = 4
D = 32
T = 510
U_MAX = 0.95
BIG_M = sum(t.values())


@st.cache_data(show_spinner=False)
def solve_model(exact_workers=None, time_limit=30):
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
        model.Add(
            sum(j * x[i, j] for j in J)
            <=
            sum(j * x[h, j] for j in J)
        )

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
                if j < k and d[j][k] > L:
                    model.Add(y[w, j] + y[w, k] <= 1)

    if exact_workers is not None:
        model.Add(sum(z[w] for w in W) == exact_workers)

    model.Minimize(C)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_search_workers = 8

    status = solver.Solve(model)

    if status not in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        return None

    C_value = solver.Value(C) / SCALE

    solution = {
        "C": C_value,
        "used_workers": sum(solver.Value(z[w]) for w in W),
        "stations_of_worker": {w: [] for w in W},
        "ops_of_station": {j: [] for j in J},
        "station_loads": {j: solver.Value(l[j]) / SCALE for j in J},
        "worker_load_per_product": {
            w: sum(solver.Value(q[w, j]) for j in J) / SCALE for w in W
        },
        "worker_load_per_shift": {
            w: D * sum(solver.Value(q[w, j]) for j in J) / SCALE for w in W
        },
        "worker_U": {
            w: 100 * ((D / T) * (sum(solver.Value(q[w, j]) for j in J) / SCALE))
            for w in W
        },
        "reachable_output": T / C_value if C_value > 0 else float("inf"),
        "meets_target": (T / C_value >= D - 1e-6) if C_value > 0 else True,
    }

    for i in I:
        for j in J:
            if solver.Value(x[i, j]) == 1:
                solution["ops_of_station"][j].append(i)

    for w in W:
        for j in J:
            if solver.Value(y[w, j]) == 1:
                solution["stations_of_worker"][w].append(j)

    return solution


def build_summary_df(results):
    rows = []
    feasible_eps = [e for e, res in results.items() if res is not None]

    ideal = None
    nadir = None

    if feasible_eps:
        ideal = (
            min(results[e]["C"] for e in feasible_eps),
            min(results[e]["used_workers"] for e in feasible_eps)
        )

        nadir = (
            max(results[e]["C"] for e in feasible_eps),
            max(results[e]["used_workers"] for e in feasible_eps)
        )

    for eps in sorted(results.keys()):
        res = results[eps]

        if res is None:
            rows.append({
                "Epsilon": f"{eps:.2f}",
                "F1 (C)": "Infeasible",
                "F2 (Z)": "-",
                "Ulaşılabilir Üretim": "-",
                "Hedef?": "-"
            })
        else:
            rows.append({
                "Epsilon": f"{eps:.2f}",
                "F1 (C)": f"{res['C']:.2f}",
                "F2 (Z)": f"{res['used_workers']:.2f}",
                "Ulaşılabilir Üretim": f"{res['reachable_output']:.2f}",
                "Hedef?": "Evet" if res["meets_target"] else "Hayır"
            })

    return ideal, nadir, pd.DataFrame(rows)


def build_text_report(results, epsilon_choice):
    if epsilon_choice not in results or results[epsilon_choice] is None:
        return f"Epsilon = {epsilon_choice} için uygun çözüm bulunamadı."

    res = results[epsilon_choice]

    lines = []

    lines.append("=" * 78)
    lines.append(f"DETAYLI SENARYO RAPORU | Operatör Sayısı = {epsilon_choice}")
    lines.append("=" * 78)

    lines.append(f"Çevrim Süresi (C)                       : {res['C']:.2f} dk/ürün")
    lines.append(f"Kullanılan Operatör Sayısı              : {res['used_workers']}")
    lines.append(f"Maksimum İzin Verilen Operatör Doluluğu : %{U_MAX * 100:.2f}")
    lines.append(f"Ulaşılabilir Üretim                     : {res['reachable_output']:.2f} adet/vardiya")
    lines.append(f"Hedef Üretim ({D} adet) Sağlanıyor mu?  : {'Evet' if res['meets_target'] else 'Hayır'}")

    lines.append("\n[1] Operasyon -> İstasyon Atamaları")
    for j in J:
        lines.append(f"İstasyon {j}: {res['ops_of_station'][j]}")

    lines.append("\n[2] İstasyon Yükleri")
    for j in J:
        lines.append(f"İstasyon {j}: {res['station_loads'][j]:.2f} dk")

    lines.append("\n[3] Operatör -> İstasyon Atamaları")
    for w in W:
        if len(res["stations_of_worker"][w]) > 0:
            lines.append(f"Operatör {w}: {res['stations_of_worker'][w]}")

    lines.append("\n[4] Operatör Toplam Yükleri ve U Değerleri")
    for w in W:
        if len(res["stations_of_worker"][w]) > 0:
            lines.append(
                f"Operatör {w}: "
                f"ürün başı yük = {res['worker_load_per_product'][w]:.2f} dk, "
                f"vardiya yükü = {res['worker_load_per_shift'][w]:.2f} dk, "
                f"U = %{res['worker_U'][w]:.2f}"
            )

    lines.append("\n[5] Mesafe Kontrolü")

    any_pair = False

    for w in W:
        sts = res["stations_of_worker"][w]

        if len(sts) >= 2:
            for a in range(len(sts)):
                for b in range(a + 1, len(sts)):
                    j = sts[a]
                    k = sts[b]

                    any_pair = True
                    status = "Uygun" if d[j][k] <= L else "İhlal"

                    lines.append(
                        f"Operatör {w}: "
                        f"İstasyon {j}-{k}, "
                        f"mesafe = {d[j][k]}, "
                        f"durum = {status}"
                    )

    if not any_pair:
        lines.append("Birden fazla istasyona atanmış operatör yok.")

    lines.append("=" * 78)

    return "\n".join(lines)


st.set_page_config(
    page_title="Hat Dengeleme CP-SAT",
    layout="wide"
)

st.title("Hat Dengeleme CP-SAT Optimizasyon Arayüzü")

with st.sidebar:
    st.header("Parametreler")

    min_eps = st.number_input(
        "Başlangıç operatör sayısı",
        min_value=1,
        max_value=36,
        value=1,
        step=1
    )

    max_eps = st.number_input(
        "Bitiş operatör sayısı",
        min_value=1,
        max_value=36,
        value=36,
        step=1
    )

    time_limit = st.number_input(
        "Her senaryo için süre limiti (sn)",
        min_value=1,
        max_value=300,
        value=30,
        step=1
    )

    epsilon_choice = st.number_input(
        "Detaylı rapor operatör sayısı",
        min_value=1,
        max_value=36,
        value=29,
        step=1
    )

    run_button = st.button("Modeli Çalıştır", type="primary")

if min_eps > max_eps:
    st.error("Başlangıç operatör sayısı, bitiş operatör sayısından büyük olamaz.")
    st.stop()

if run_button:
    results = {}

    progress = st.progress(0)
    status_box = st.empty()

    eps_values = list(range(int(min_eps), int(max_eps) + 1))

    for idx, eps in enumerate(eps_values, start=1):
        status_box.info(f"{eps} operatörlü senaryo çözülüyor...")

        results[eps] = solve_model(
            exact_workers=eps,
            time_limit=int(time_limit)
        )

        progress.progress(idx / len(eps_values))

    status_box.success("Çözüm tamamlandı.")

    ideal, nadir, summary_df = build_summary_df(results)

    if ideal is not None:
        col1, col2 = st.columns(2)

        col1.metric(
            "Ideal Nokta",
            f"({ideal[0]:.2f}, {ideal[1]:.2f})"
        )

        col2.metric(
            "Nadir Nokta",
            f"({nadir[0]:.2f}, {nadir[1]:.2f})"
        )
    else:
        st.warning("Hiçbir epsilon değeri için çözüm bulunamadı.")

    st.subheader("Epsilon Senaryo Tablosu")
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    st.subheader("Detaylı Senaryo Raporu")
    report = build_text_report(results, int(epsilon_choice))

    st.code(report, language="text")

else:
    st.info("Sol menüden parametreleri seçip 'Modeli Çalıştır' butonuna bas.")