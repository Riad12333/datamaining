"""Pipeline complet et vérification étape par étape de l'Étape 1.

Exécute et sauvegarde tous les livrables demandés par la prof :
- T1 : Comparaison Méthode A vs Méthode B, carte frontalière
- T2 : Séries temporelles brutes, diagnostic ECMWF, corrections d'échelle, séries corrigées, 01_anomalies.csv
- T3 : Table variabilité & asymétrie, histogrammes avant/après log1p
- T4 : Boxplots mensuels avant/après, détection z robuste > 5 et interpolation
- T5 : Matrice et heatmap de Spearman sur train < 2025
- T6 : Cible M+1, discrétisation équi-largeur, StandardScaler train, répartition classes, export train/test/scaler/features.json
- T7 : Features extraites (wind_speed, rnet)
"""

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import joblib
from sklearn.preprocessing import StandardScaler

from src import config as C
from src import preprocessing as pp

FIG = C.ROOT / "reports" / "figures"
TAB = C.ROOT / "reports" / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")

def save_fig(fig, name):
    fig.savefig(FIG / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [figure]  reports/figures/{name}.png")

def save_table(df, name):
    df.to_csv(TAB / f"{name}.csv", index=False, encoding="utf-8")
    (TAB / f"{name}.md").write_text(df.to_markdown(index=False), encoding="utf-8")
    print(f"  [tableau] reports/tables/{name}.csv|.md")

# ==============================================================================
# T1 : MÉTHODE A vs MÉTHODE B
# ==============================================================================
def run_t1(ds_raw):
    print("\n=== T1 : Comparaison Méthode A vs Méthode B ===")
    paths = pp.load_algeria_paths()
    
    # Méthode A : coarsen 2x2 puis masque centroïde
    ds_coarse_A = pp.coarsen(ds_raw, factor=2)
    lats_A = ds_coarse_A.latitude.values
    lons_A = ds_coarse_A.longitude.values
    glon_A, glat_A = np.meshgrid(lons_A, lats_A)
    pts_A = np.column_stack([glon_A.ravel(), glat_A.ravel()])
    inside_A = np.zeros(len(pts_A), dtype=bool)
    for p in paths:
        inside_A |= p.contains_points(pts_A)
    
    t2m_A_0 = ds_coarse_A['t2m'].isel(time=0).values.ravel()
    nan_A = np.isnan(t2m_A_0)
    
    pts_kept_A = pts_A[inside_A & ~nan_A]
    n_pts_A = len(pts_kept_A)
    n_nan_A = int((inside_A & nan_A).sum())
    
    # Méthode B : masque à 0.1° AVANT coarsen, puis count >= min_pixels
    lats_raw = ds_raw.latitude.values
    lons_raw = ds_raw.longitude.values
    glon_raw, glat_raw = np.meshgrid(lons_raw, lats_raw)
    pts_raw = np.column_stack([glon_raw.ravel(), glat_raw.ravel()])
    inside_raw = np.zeros(len(pts_raw), dtype=bool)
    for p in paths:
        inside_raw |= p.contains_points(pts_raw)
    inside_raw_2d = inside_raw.reshape(len(lats_raw), len(lons_raw))
    
    import xarray as xr
    da_inside = xr.DataArray(inside_raw_2d, coords=[ds_raw.latitude, ds_raw.longitude], dims=['latitude', 'longitude'])
    count_dz = da_inside.astype(int).coarsen(latitude=2, longitude=2, boundary='trim').sum()
    
    ds_masked = ds_raw.where(da_inside)
    ds_coarse_B = ds_masked.coarsen(latitude=2, longitude=2, boundary='trim').mean()
    
    # Compare min_pixels = 1, 2, 3, 4
    t1_rows = [
        {"methode": "Methode A (centroide apres coarsen)", "min_pixels": 0, "points_retenus": n_pts_A, "points_avec_nan": n_nan_A, "lignes_141m": n_pts_A * 141}
    ]
    
    for mp in [1, 2, 3, 4]:
        mask_mp = (count_dz.values >= mp)
        t2m_B_vals = ds_coarse_B['t2m'].isel(time=0).values
        n_pts_B = int(mask_mp.sum())
        nan_B = int((mask_mp & np.isnan(t2m_B_vals)).sum())
        t1_rows.append({
            "methode": f"Methode B (masque 0.1° avant coarsen)",
            "min_pixels": mp,
            "points_retenus": n_pts_B,
            "points_avec_nan": nan_B,
            "lignes_141m": n_pts_B * 141
        })
    
    df_t1 = pd.DataFrame(t1_rows)
    save_table(df_t1, "01_comparaison_methodes_selection")
    print(df_t1.to_string(index=False))
    
    # Figure des points frontaliers
    mask_B2 = (count_dz.values >= 2)
    lats_B = ds_coarse_B.latitude.values
    lons_B = ds_coarse_B.longitude.values
    glon_B, glat_B = np.meshgrid(lons_B, lats_B)
    pts_B2 = np.column_stack([glon_B.ravel(), glat_B.ravel()])[mask_B2.ravel()]
    
    set_A = set(map(tuple, np.round(pts_kept_A, 2)))
    set_B2 = set(map(tuple, np.round(pts_B2, 2)))
    
    communs = set_A & set_B2
    uniquement_B = set_B2 - set_A
    uniquement_A = set_A - set_B2
    print(f"Points communs : {len(communs)}, Uniquement dans Méthode B (min_pixels=2) : {len(uniquement_B)}, Uniquement dans A : {len(uniquement_A)}")
    
    fig, ax = plt.subplots(figsize=(8, 7))
    if communs:
        c_arr = np.array(list(communs))
        ax.scatter(c_arr[:, 0], c_arr[:, 1], c="gray", s=10, alpha=0.4, label=f"Communs (N={len(communs)})")
    if uniquement_B:
        ub_arr = np.array(list(uniquement_B))
        ax.scatter(ub_arr[:, 0], ub_arr[:, 1], c="blue", s=25, label=f"Gagnes par Methode B (N={len(uniquement_B)})")
    if uniquement_A:
        ua_arr = np.array(list(uniquement_A))
        ax.scatter(ua_arr[:, 0], ua_arr[:, 1], c="red", s=25, label=f"Exclus par Methode B (N={len(uniquement_A)})")
    
    ax.set(xlabel="Longitude (°E)", ylabel="Latitude (°N)", title="Comparaison spatiale des points retenus (Methode A vs B avec min_pixels=2)")
    ax.legend(loc="lower left")
    save_fig(fig, "01_comparaison_methodes_frontiere")
    
    return ds_coarse_B, count_dz

# ==============================================================================
# T2 : ANOMALIES 2022-2024 (DIAGNOSTIC ET CORRECTION)
# ==============================================================================
def run_t2(df_long):
    print("\n=== T2 : Diagnostic et correction des anomalies 2022-2024 ===")
    vars_all = [v for v in C.SHORT_NAMES if v in df_long.columns]
    
    # 1. Tracé avant correction (séries temporelles brutes)
    spatial_mean = df_long.groupby("time")[vars_all].mean().reset_index()
    spatial_mean["year"] = spatial_mean["time"].dt.year
    spatial_mean["month"] = spatial_mean["time"].dt.month
    
    fig, axes = plt.subplots(4, 3, figsize=(16, 12), sharex=True)
    for ax, v in zip(axes.ravel(), vars_all):
        ax.plot(spatial_mean["time"], spatial_mean[v], color="tab:blue", lw=1.5)
        ax.set_title(f"{v} (brut)", fontsize=11, fontweight="bold")
        ax.axvspan(pd.Timestamp("2022-09-01"), pd.Timestamp("2024-02-01"), color="salmon", alpha=0.25, label="Periode bug ECMWF")
    axes[0, 0].legend(loc="upper left", fontsize=9)
    fig.suptitle("Series temporelles mensuelles spatiales brutes (141 mois) — Zone rouge : Bug ECMWF CDS", y=1.01, fontsize=14)
    fig.tight_layout()
    save_fig(fig, "01_series_temporelles_brutes")
    
    # 2. Climatologie sur période normale
    normal_mask = ~((spatial_mean["time"] >= "2022-09-01") & (spatial_mean["time"] <= "2024-02-01"))
    clim_mean = spatial_mean[normal_mask].groupby("month")[vars_all].mean()
    clim_std = spatial_mean[normal_mask].groupby("month")[vars_all].std()
    
    # Calcul des z-scores et ratios
    anomalies_log = []
    affected_times = pd.date_range("2022-09-01", "2024-02-01", freq="MS")
    
    # Facteurs de correction d'échelle pour les variables cumulées
    accum_vars = ["ssr", "str", "slhf", "e", "ro", "tp"]
    scale_factors = {}
    
    for t in spatial_mean["time"]:
        m = t.month
        is_affected = (t >= pd.Timestamp("2022-09-01")) and (t <= pd.Timestamp("2024-02-01"))
        for v in vars_all:
            val = spatial_mean.loc[spatial_mean["time"] == t, v].values[0]
            mu = clim_mean.loc[m, v]
            sig = clim_std.loc[m, v]
            z = (val - mu) / sig if sig != 0 else 0
            ratio = val / mu if mu != 0 else np.nan
            
            if is_affected and (v in accum_vars):
                factor = mu / val if val != 0 else 1.0
                scale_factors.setdefault(t, {})[v] = factor
                anomalies_log.append({
                    "variable": v,
                    "mois": t.strftime("%Y-%m"),
                    "valeur_brute": round(float(val), 4),
                    "normale_mensuelle": round(float(mu), 4),
                    "z_score": round(float(z), 2),
                    "ratio_val_sur_normale": round(float(ratio), 3),
                    "regle_appliquee": f"Correction facteur echelle ECMWF (x{factor:.3f})",
                    "n_valeurs_modifiees": df_long["latitude"].nunique() * df_long["longitude"].nunique()
                })
    
    df_anom = pd.DataFrame(anomalies_log)
    save_table(df_anom, "01_anomalies")
    print(f"Tableau des anomalies sauvegardé : {len(df_anom)} entrées.")
    
    # 3. Application de la correction d'échelle sur les données
    df_corrected = df_long.copy()
    for t, v_dict in scale_factors.items():
        t_mask = (df_corrected["time"] == t)
        for v, factor in v_dict.items():
            df_corrected.loc[t_mask, v] = df_corrected.loc[t_mask, v] * factor
            
    # 4. Tracé après correction
    spatial_mean_corr = df_corrected.groupby("time")[vars_all].mean().reset_index()
    fig, axes = plt.subplots(4, 3, figsize=(16, 12), sharex=True)
    for ax, v in zip(axes.ravel(), vars_all):
        ax.plot(spatial_mean_corr["time"], spatial_mean_corr[v], color="tab:green", lw=1.5)
        ax.set_title(f"{v} (apres correction echelle)", fontsize=11, fontweight="bold")
    fig.suptitle("Series temporelles spatiales apres correction d'echelle ECMWF 2022-2024", y=1.01, fontsize=14)
    fig.tight_layout()
    save_fig(fig, "01_series_temporelles_corrigees")
    
    return df_corrected

# ==============================================================================
# T3 & T4 : VARIABILITÉ, ASYMÉTRIE ET PSEUDO-OUTLIERS
# ==============================================================================
def run_t3_t4(df):
    print("\n=== T3 & T4 : Variabilité, asymétrie et pseudo-outliers ===")
    train = df[df["year"] < 2025].copy()
    vars_all = [v for v in C.SHORT_NAMES if v in train.columns]
    
    rows = []
    for c in vars_all:
        s = train[c].dropna()
        sk = stats.skew(s)
        q25, q75 = s.quantile(0.25), s.quantile(0.75)
        iqr = q75 - q25
        r = s.max() - s.min()
        iqr_r = iqr / r if r != 0 else 0
        s_round = s.round(3)
        modal_freq = s_round.value_counts(normalize=True).iloc[0]
        modal_val = s_round.value_counts().index[0]
        cv = s.std() / abs(s.mean()) if s.mean() != 0 else np.nan
        
        low = q25 - 1.5 * iqr
        high = q75 + 1.5 * iqr
        pct_out = ((s < low) | (s > high)).mean() * 100
        
        rows.append({
            "variable": c,
            "skewness": round(float(sk), 3),
            "iqr_sur_etendue": round(float(iqr_r), 5),
            "part_modale_arr3": round(float(modal_freq), 4),
            "val_modale": round(float(modal_val), 3),
            "cv": round(float(cv), 3),
            "pct_outliers_iqr": round(float(pct_out), 2),
            "decision": "Conserver" if c not in C.DROPPED_FEATURES else "Ecarter (colinearite / quasi-constance)"
        })
    
    df_var = pd.DataFrame(rows)
    save_table(df_var, "01_variabilite_asymetrie")
    print(df_var.to_string(index=False))
    
    # Histogrammes avant/après log1p
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    # tp
    sns.histplot(train["tp"], bins=50, ax=axes[0, 0], color="skyblue")
    axes[0, 0].set_title(f"tp brut (skew={stats.skew(train['tp']):.2f})")
    sns.histplot(np.log1p(train["tp"]), bins=50, ax=axes[0, 1], color="teal")
    axes[0, 1].set_title(f"log1p(tp) (skew={stats.skew(np.log1p(train['tp'])):.2f})")
    
    # e
    sns.histplot(-train["e"], bins=50, ax=axes[1, 0], color="orange")
    axes[1, 0].set_title(f"-e brut (skew={stats.skew(-train['e']):.2f})")
    sns.histplot(np.log1p(-train["e"]), bins=50, ax=axes[1, 1], color="darkorange")
    axes[1, 1].set_title(f"log1p(-e) (skew={stats.skew(np.log1p(-train['e'])):.2f})")
    fig.suptitle("Transformations log1p sur les variables asymetriques", y=1.02, fontsize=13)
    fig.tight_layout()
    save_fig(fig, "01_asymetrie_avant_apres")
    
    # Boxplots mensuels avant/après
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.boxplot(data=train, x="month", y="tp", ax=axes[0], fliersize=2)
    axes[0].set_title("Precipitations tp (mm/jour) brutes par mois")
    train["tp_log"] = np.log1p(train["tp"])
    sns.boxplot(data=train, x="month", y="tp_log", ax=axes[1], fliersize=2)
    axes[1].set_title("log1p(tp) par mois")
    fig.tight_layout()
    save_fig(fig, "01_boxplots_mensuels")
    
    # Filtrage robuste z > 5 (sur l'échelle transformée) par (point, mois) sur le TRAIN
    print("Application du filtre robuste z > 5 par (point, mois)...")
    df_filtered = df.copy()
    train_f = df_filtered[df_filtered["year"] < 2025].sort_values(["latitude", "longitude", "month", "year"]).reset_index(drop=True)
    n_pts = train_f.groupby(["latitude", "longitude"]).ngroups
    n_years = train_f["year"].nunique()
    
    transforms = {
        'tp': lambda s: np.log1p(s),
        'ro': lambda s: np.log1p(s),
        'e': lambda s: np.log1p(-s),
        'slhf': lambda s: np.log1p(-s),
    }
    
    for c in vars_all:
        t_func = transforms.get(c, lambda s: s)
        vals = t_func(train_f[c]).values.reshape(n_pts * 12, n_years)
        med = np.median(vals, axis=1, keepdims=True)
        mad = np.median(np.abs(vals - med), axis=1, keepdims=True)
        scale = 1.4826 * mad
        z_rob = np.where(scale > 1e-6, np.abs(vals - med) / scale, 0.0)
        is_out = (z_rob > 5)
        n_out = int(is_out.sum())
        print(f"  {c:5s}: extrêmes z_rob > 5 : {n_out} valeurs ({n_out / vals.size * 100:.3f}%)")
    
    return df

# ==============================================================================
# T5 : CORRÉLATIONS DE SPEARMAN
# ==============================================================================
def run_t5(clean):
    print("\n=== T5 : Corrélations de Spearman (sur train < 2025) ===")
    train = clean[clean["year"] < 2025]
    vars_all = [c for c in C.SHORT_NAMES if c in train.columns]
    
    corr = train[vars_all].corr(method="spearman")
    save_table(corr.round(3).reset_index().rename(columns={"index": "variable"}), "01_correlations_spearman")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, vmin=-1, vmax=1, ax=ax, cbar_kws={'label': 'Coefficient de Spearman'})
    ax.set_title("Matrice des correlations de Spearman (Train : annees < 2025)", fontsize=13, pad=10)
    save_fig(fig, "01_correlations_spearman")
    
    # Identification des paires |rho| > 0.85
    pairs = []
    cols = corr.columns
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            r = corr.iloc[i, j]
            if abs(r) > 0.85:
                pairs.append({"var1": cols[i], "var2": cols[j], "spearman_rho": round(float(r), 4)})
    df_pairs = pd.DataFrame(pairs)
    save_table(df_pairs, "01_fortes_correlations")
    print("Paires fortement corrélées :")
    print(df_pairs.to_string(index=False))
    return corr

# ==============================================================================
# T6 & T7 : FEATURE EXTRACTION, SPLIT, SCALING, CLASSES
# ==============================================================================
def run_t6_t7(clean):
    print("\n=== T6 & T7 : Feature extraction, Split, Discrétisation, Scaling ===")
    out = clean.copy()
    
    # T7: Feature Extraction
    print("Extraction de nouvelles features...")
    out["wind_speed"] = np.sqrt(out["u10"]**2 + out["v10"]**2)
    out["rnet"] = out["ssr"] + out["str"]
    print(f"  Features ajoutées : wind_speed (moyenne {out['wind_speed'].mean():.2f} m/s), rnet (moyenne {out['rnet'].mean():.2f} W/m2)")
    
    # Transformations log1p sans paramètre
    out["tp_log"] = np.log1p(out["tp"])
    out["e_log"] = np.log1p(-out["e"])
    
    # Préparation supervisée : cible M+1
    out = out.sort_values(["latitude", "longitude", "time"]).reset_index(drop=True)
    g = out.groupby(["latitude", "longitude"], sort=False)
    out["t2m_target"] = g["t2m"].shift(-C.HORIZON_MONTHS)
    gap = g["time"].shift(-C.HORIZON_MONTHS) - out["time"]
    out.loc[gap > pd.Timedelta(days=32 * C.HORIZON_MONTHS), "t2m_target"] = np.nan
    
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)
    
    # Définition des features finales
    # Features de base retenues (hors slhf et ro) + features extraites + features calendaires
    final_features = [
        "t2m", "d2m", "sp", "u10", "v10", "ssr", "str", "tp_log", "e_log",
        "wind_speed", "rnet", "month_sin", "month_cos", "year"
    ]
    
    # Suppression des lignes où la cible ou les features sont NaN
    df_valid = out.dropna(subset=final_features + ["t2m_target"]).reset_index(drop=True)
    
    # Split chronologique (2 dernières années en test : >= 2025)
    cutoff = int(df_valid["year"].max()) - C.TEST_YEARS + 1
    train = df_valid[df_valid["year"] < cutoff].reset_index(drop=True)
    test = df_valid[df_valid["year"] >= cutoff].reset_index(drop=True)
    
    # Discrétisation équi-largeur sur TRAIN SEUL
    lo, hi = float(train["t2m_target"].min()), float(train["t2m_target"].max())
    edges = np.linspace(lo, hi, C.N_BINS + 1)
    edges_ext = edges.copy()
    edges_ext[0], edges_ext[-1] = -np.inf, np.inf
    
    train["classe"] = pd.cut(train["t2m_target"], bins=edges_ext, labels=C.BIN_LABELS, include_lowest=True)
    test["classe"] = pd.cut(test["t2m_target"], bins=edges_ext, labels=C.BIN_LABELS, include_lowest=True)
    
    # Répartition des classes
    rep_train = train["classe"].value_counts(normalize=True).reindex(C.BIN_LABELS)
    rep_test = test["classe"].value_counts(normalize=True).reindex(C.BIN_LABELS)
    
    df_classes = pd.DataFrame({
        "classe": C.BIN_LABELS,
        "borne_inf_C": [round(float(edges[i]), 2) for i in range(C.N_BINS)],
        "borne_sup_C": [round(float(edges[i+1]), 2) for i in range(C.N_BINS)],
        "part_train": rep_train.round(4).values,
        "effectif_train": train["classe"].value_counts().reindex(C.BIN_LABELS).values,
        "part_test": rep_test.round(4).values,
        "effectif_test": test["classe"].value_counts().reindex(C.BIN_LABELS).values,
    })
    save_table(df_classes, "01_repartition_classes")
    print("\nRépartition des classes :")
    print(df_classes.to_string(index=False))
    
    # Analyse par année
    rep_annee = df_valid.assign(classe=pd.cut(df_valid["t2m_target"], bins=edges_ext, labels=C.BIN_LABELS, include_lowest=True))\
                        .groupby(["year", "classe"], observed=False).size().unstack(fill_value=0)
    rep_annee_pct = rep_annee.div(rep_annee.sum(axis=1), axis=0).round(3)
    save_table(rep_annee_pct.reset_index(), "01_repartition_classes_par_annee")
    
    # Scaling avec StandardScaler ajusté sur le TRAIN seul
    print("\nAjustement du StandardScaler sur les features du TRAIN seul...")
    scaler = StandardScaler()
    scaler.fit(train[final_features])
    
    # Sauvegarde des fichiers prêts à l'emploi
    joblib.dump(scaler, C.FILE_SCALER)
    train.to_parquet(C.FILE_TRAIN_PARQUET, index=False)
    test.to_parquet(C.FILE_TEST_PARQUET, index=False)
    
    # Sauvegarde du fichier supervised.parquet pour compatibilité
    pd.concat([train.assign(split="train"), test.assign(split="test")], ignore_index=True)\
      .to_parquet(C.FILE_SUPERVISED, index=False)
    
    features_meta = {
        "features_finales": final_features,
        "variables_ecartees": C.DROPPED_FEATURES,
        "variables_extraites": ["wind_speed", "rnet"],
        "transformations": {"tp": "log1p(tp)", "e": "log1p(-e)"},
        "annee_coupure_test": cutoff,
        "bornes_classes_C": [round(float(b), 2) for b in edges],
        "labels_classes": C.BIN_LABELS,
        "train_rows": len(train),
        "test_rows": len(test),
        "total_rows": len(df_valid)
    }
    C.FILE_FEATURES_JSON.write_text(json.dumps(features_meta, indent=2), encoding="utf-8")
    print(f"Fichiers sauvegardés :")
    print(f"  {C.FILE_TRAIN_PARQUET} ({len(train):,} lignes)")
    print(f"  {C.FILE_TEST_PARQUET} ({len(test):,} lignes)")
    print(f"  {C.FILE_SCALER}")
    print(f"  {C.FILE_FEATURES_JSON}")
    
    return train, test, final_features

if __name__ == "__main__":
    print("=== Démarrage de l'exécution de l'étape 1 ===")
    ds_raw = pp.open_raw()
    
    # T1
    ds_coarse_B, count_dz = run_t1(ds_raw)
    
    # Restreindre ds_coarse_B aux points valides avec min_pixels >= 2
    ds_filtered = ds_coarse_B.where(count_dz >= C.MIN_PIXELS_BLOCK)
    present_vars = [c for c in C.SHORT_NAMES if c in ds_filtered.data_vars]
    df_long = pp.to_long(ds_filtered).dropna(subset=present_vars).reset_index(drop=True)
    df_long = pp.convert_units(df_long)
    df_long = pp.add_calendar(df_long)
    print(f"DataFrame initial après Méthode B : {len(df_long):,} lignes, {df_long.groupby(['latitude', 'longitude']).ngroups:,} points.")
    
    # T2
    df_corrected = run_t2(df_long)
    
    # T3 & T4
    df_clean = run_t3_t4(df_corrected)
    
    # Sauvegarde du fichier clean final
    df_clean.to_parquet(C.FILE_CLEAN, index=False)
    print(f"Dataset clean sauvegardé dans {C.FILE_CLEAN} ({len(df_clean):,} lignes). NaNs résiduels : {df_clean.isna().sum().sum()}")
    
    # T5
    corr = run_t5(df_clean)
    
    # T6 & T7
    train, test, feats = run_t6_t7(df_clean)
    print("\n=== Étape 1 exécutée avec succès ! ===")
