import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from setup import Air, Alumina, Geometry, CalcHeatCoeffs, ImplicitEuler

PLOT_EXPER = True  # overlay experimental CSV
CSV_PATH = "cascetta_results.csv"

def theta_of(T, Tmin, Tmax):
    return (T - Tmin) / (Tmax - Tmin)


def find_tmax_and_final_state(
    mode_name, Tf0, Ts0, u, T_inlet,
    dt, dz, por, fluid, solid, geom,
    Tmin, Tmax, max_time=20000.0,
    outlet_idx=None,
    charge_rule=True
):
    """Evolve until Cascetta’s stopping time (θ_outlet ≥ 0.1 for charge, ≤ 0.9 for discharge)."""
    solver = ImplicitEuler()
    Tf = Tf0.copy()
    Ts = Ts0.copy()
    t = 0.0
    if outlet_idx is None:
        outlet_idx = -1 if u >= 0 else 0

    while t < max_time:
        T_mean = 0.5 * (Tf.mean() + Ts.mean())

        rho_f = fluid.density(T_mean)
        cp_f  = fluid.specific_heat(T_mean)
        rho_s = solid.density(T_mean)
        cp_s  = solid.specific_heat(T_mean)
        h     = CalcHeatCoeffs.calc_h(T_mean, abs(u), geom.particle_diameter, por, fluid)
        ha    = h * geom.specific_surface_area

        solver = ImplicitEuler()
        a, b, c, d = solver.assemble_TDMA_Tf(
            Tf, Ts, dt, dz, u, por, rho_f, cp_f, rho_s, cp_s, ha, T_inlet
        )
        Tf_new = solver.TDMA(a, b, c, d)
        Ts_new = solver.recover_Ts(Tf_new, Ts, dt, por, rho_s, cp_s, ha)

        t += dt
        th_out = theta_of(Tf_new[outlet_idx], Tmin, Tmax)
        if (charge_rule and th_out >= 0.1) or ((not charge_rule) and th_out <= 0.9):
            return t, Tf_new, Ts_new

        Tf, Ts = Tf_new, Ts_new

    print(f"{mode_name}: did not reach threshold before cap; using cap.")
    return max_time, Tf, Ts


def main():
    # === Geometry & materials ===
    bed_height = 1.8
    porosity = 0.39
    Rint = 0.584 / 2
    dp = 0.008
    mdot = 0.2  # kg/s

    Tmin = 51 + 273.15
    Tmax = 192 + 273.15
    T_ref = 0.5 * (Tmin + Tmax)
    # T_ref = Tmin

    fluid = Air()
    solid = Alumina()
    geom = Geometry(bed_height, porosity, dp, Rint)
    solver = ImplicitEuler()

    # === Properties ===
    rho_f = fluid.density(T_ref)
    cp_f = fluid.specific_heat(T_ref)
    rho_s = solid.density(T_ref)
    cp_s = solid.specific_heat(T_ref)

    A_cs = np.pi * Rint**2
    v_sup = mdot / (rho_f * A_cs)
    u_mag = v_sup / porosity

    # === Heat transfer ===
    h = CalcHeatCoeffs.calc_h(T_ref, u_mag, dp, porosity, fluid)
    ha = h * geom.specific_surface_area

    # === Grid & time step ===
    N = 500
    z = np.linspace(0.0, bed_height, N)
    dz = z[1] - z[0]
    dt = 0.05
    xs = z / bed_height

    # ---------- STAGE 1: CHARGING ----------
    u_charge = +u_mag
    T_in_charge = Tmax
    Tf0_charge = np.full(N, Tmin)
    Ts0_charge = np.full(N, Tmin)

    tmax_ch, Tf_end, Ts_end = find_tmax_and_final_state(
        "charge", Tf0_charge, Ts0_charge, u_charge, T_in_charge,
        dt, dz, porosity, fluid, solid, geom,
        Tmin, Tmax, max_time=20000.0,
        outlet_idx=-1, charge_rule=True
    )
    print(f"Charging t_max = {tmax_ch:.2f} s")

    # ---------- STAGE 2: DISCHARGING (to get t_max) ----------
    u_dis = -u_mag
    T_in_dis = Tmin
    outlet_idx = (N - 1) if u_dis >= 0 else 0

    Tf = Tf_end.copy()
    Ts = Ts_end.copy()

    tmax_dis, _, _ = find_tmax_and_final_state(
        "discharge", Tf.copy(), Ts.copy(), u_dis, T_in_dis,
        dt, dz, porosity, fluid, solid, geom,
        Tmin, Tmax, max_time=20000.0,
        outlet_idx=outlet_idx, charge_rule=False
    )
    print(f"Discharging t_max = {tmax_dis:.2f} s")

    # ============ Sample profiles for CHARGE and DISCHARGE ============
    taus = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    # --- CHARGE ---
    t_targets_ch = taus * tmax_ch
    profiles_ch = []
    Tf_ch = Tf0_charge.copy()
    Ts_ch = Ts0_charge.copy()
    t = 0.0
    next_idx = 0
    while next_idx < len(t_targets_ch) and t <= tmax_ch + 1e-12:
        if t >= t_targets_ch[next_idx] - 1e-12:
            profiles_ch.append(Tf_ch.copy())
            next_idx += 1
            if next_idx == len(t_targets_ch):
                break
        T_mean = 0.5 * (Tf_ch.mean() + Ts_ch.mean())
        rho_f = fluid.density(T_mean)
        cp_f  = fluid.specific_heat(T_mean)
        rho_s = solid.density(T_mean)
        cp_s  = solid.specific_heat(T_mean)
        h     = CalcHeatCoeffs.calc_h(T_mean, abs(u_charge), geom.particle_diameter, porosity, fluid)
        ha    = h * geom.specific_surface_area

        a, b, c, d = solver.assemble_TDMA_Tf(
            Tf_ch, Ts_ch, dt, dz, u_charge, porosity, rho_f, cp_f, rho_s, cp_s, ha, T_in_charge
        )
        Tf_new = solver.TDMA(a, b, c, d)
        Ts_new = solver.recover_Ts(Tf_new, Ts_ch, dt, porosity, rho_s, cp_s, ha)
        Tf_ch, Ts_ch = Tf_new, Ts_new
        t += dt

    if len(profiles_ch) < len(taus):
        profiles_ch.append(Tf_ch.copy())

    # --- DISCHARGE ---
    t_targets_dis = taus * tmax_dis
    profiles_dis = []
    Tf = Tf_end.copy()
    Ts = Ts_end.copy()
    t = 0.0
    next_idx = 0
    while next_idx < len(t_targets_dis) and t <= tmax_dis + 1e-12:
        if t >= t_targets_dis[next_idx] - 1e-12:
            profiles_dis.append(Tf.copy())
            next_idx += 1
            if next_idx == len(t_targets_dis):
                break
        T_mean = 0.5 * (Tf.mean() + Ts.mean())
        rho_f = fluid.density(T_mean)
        cp_f  = fluid.specific_heat(T_mean)
        rho_s = solid.density(T_mean)
        cp_s  = solid.specific_heat(T_mean)
        h     = CalcHeatCoeffs.calc_h(T_mean, abs(u_dis), geom.particle_diameter, porosity, fluid)
        ha    = h * geom.specific_surface_area

        a, b, c, d = solver.assemble_TDMA_Tf(
            Tf, Ts, dt, dz, u_dis, porosity, rho_f, cp_f, rho_s, cp_s, ha, T_in_dis
        )
        Tf_new = solver.TDMA(a, b, c, d)
        Ts_new = solver.recover_Ts(Tf_new, Ts, dt, porosity, rho_s, cp_s, ha)
        Tf, Ts = Tf_new, Ts_new
        t += dt

    if len(profiles_dis) < len(taus):
        profiles_dis.append(Tf.copy())

    # ---------- Plotting ----------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.5), sharey=True)
    cmap = plt.get_cmap("tab10")

    colour_map_c = {}
    for k, Tprof in enumerate(profiles_ch):
        ax1.plot(xs, theta_of(Tprof, Tmin, Tmax), lw=2, color=cmap(k % 10), label=f"τ={taus[k]:.1f}")
    if PLOT_EXPER:
        try:
            df = pd.read_csv(CSV_PATH, sep=";")
            if set(["xL_c","theta_c","tau_c"]).issubset(df.columns):
                df_c = df[["xL_c", "theta_c", "tau_c"]].dropna()
                df_c = df_c.rename(columns={"xL_c":"xL","theta_c":"theta","tau_c":"tau"})
                df_c = df_c.apply(pd.to_numeric, errors="coerce").dropna()
                groups_c = dict(tuple(df_c.groupby(df_c["tau"].round(1))))
                for tau_k, g in sorted(groups_c.items(), key=lambda kv: kv[0]):
                    ci = colour_map_c.setdefault(float(tau_k),
                            (max(colour_map_c.values(), default=-1)+1) % 10)
                    ax1.scatter(g["xL"].values, g["theta"].values, s=30,
                                facecolor="none", edgecolor=cmap(ci))
        except FileNotFoundError:
            print(f"Warning: CSV '{CSV_PATH}' not found")

    ax1.set_xlabel("x/L")
    ax1.set_ylabel(r"$\theta$")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, alpha=0.5)
    ax1.set_title("Schumann Model (Charge)")
    ax1.legend(loc="best", ncol=3, fontsize=9)

    # DISCHARGE panel
    colour_map_dc = {}
    for k, Tprof in enumerate(profiles_dis):
        ax2.plot(xs, theta_of(Tprof, Tmin, Tmax), lw=2, color=cmap(k % 10), label=f"τ={taus[k]:.1f}")
    if PLOT_EXPER:
        try:
            if 'df' not in locals():
                df = pd.read_csv(CSV_PATH, sep=";")
            if set(["xL_dc","theta_dc","tau_dc"]).issubset(df.columns):
                df_dc = df[["xL_dc", "theta_dc", "tau_dc"]].dropna()
                df_dc = df_dc.rename(columns={"xL_dc":"xL","theta_dc":"theta","tau_dc":"tau"})
                df_dc = df_dc.apply(pd.to_numeric, errors="coerce").dropna()
                groups_dc = dict(tuple(df_dc.groupby(df_dc["tau"].round(1))))
                for tau_k, g in sorted(groups_dc.items(), key=lambda kv: kv[0]):
                    ci = colour_map_dc.setdefault(float(tau_k),
                            (max(colour_map_dc.values(), default=-1)+1) % 10)
                    ax2.scatter(g["xL"].values, g["theta"].values, s=30,
                                facecolor="none", edgecolor=cmap(ci))
        except FileNotFoundError:
            print(f"Warning: CSV '{CSV_PATH}' not found")

    ax2.set_xlabel("x/L")
    ax2.set_xlim(0, 1)
    ax2.grid(True, alpha=0.5)
    ax2.set_title("Schumann Model (Discharge)")
    ax2.legend(loc="best", ncol=3, fontsize=9)

    # unified legend style (optional)
    # legend_elems = [
    #     Line2D([0], [0], color="k", lw=2, label="CFD"),
    #     Line2D([0], [0], marker="o", color="k", markerfacecolor="none",
    #            linestyle="None", label="Exper")
    # ]

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
