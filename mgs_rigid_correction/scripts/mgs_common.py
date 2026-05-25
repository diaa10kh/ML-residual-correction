from __future__ import print_function

import csv
import json
import math
import os
import pickle
import random
import sys


EPS = 1.0e-12

STATIC_FEATURES = [
    "S",
    "logS",
    "ID_percent",
    "D_m",
    "L_m",
    "L_over_D",
    "penetration_m",
    "penetration_over_D",
    "velocity_m_per_s",
    "v_over_v0",
    "eta",
    "z_over_D",
    "z_over_penetration",
    "z_m",
]

RAW_RESPONSE_FEATURES = [
    "qb_fast_MPa",
    "qs_fast_kPa",
    "dqb_deta",
    "dqs_deta",
    "qb_movmean_5",
    "qs_movmean_5",
    "qb_movstd_5",
    "qs_movstd_5",
    "qb_cummean",
    "qs_cummean",
]

DYNAMIC_FEATURES = [
    "energy_ratio",
    "energy_ratio_cummax",
    "oscillation_index_qb",
    "oscillation_index_qs",
]

FEATURE_COLUMNS = STATIC_FEATURES + RAW_RESPONSE_FEATURES + DYNAMIC_FEATURES
TARGET_COLUMNS = ["target_residual_qb_MPa", "target_residual_qs_kPa"]

DEPTH_ZONES = {
    "shallow": (0.02, 0.20),
    "middle": (0.20, 0.60),
    "deep": (0.60, 0.90),
    "full": (0.02, 0.90),
}


def script_dir():
    return os.path.dirname(os.path.abspath(__file__))


def project_root():
    return os.path.abspath(os.path.join(script_dir(), os.pardir))


def project_path(*parts):
    return os.path.join(project_root(), *parts)


def resolve_project_path(path):
    if not path or os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(project_root(), path.replace("\\", os.sep)))


def mkdir_p(path):
    if path and not os.path.isdir(path):
        try:
            os.makedirs(path)
        except OSError:
            if not os.path.isdir(path):
                raise


def ensure_project_dirs():
    for rel in [
        ("data", "raw", "odb"),
        ("data", "raw", "inp"),
        ("data", "raw", "sta"),
        ("data", "raw", "msg"),
        ("data", "raw", "dat"),
        ("data", "extracted", "per_run_csv"),
        ("data", "processed", "resampled_runs"),
        ("data", "benchmark"),
        ("models",),
        ("reports", "figures"),
        ("reports", "tables"),
        ("logs",),
        ("runs",),
    ]:
        mkdir_p(project_path(*rel))


def load_config(path):
    """Load a JSON-compatible YAML file using only the Python standard library.

    The files use .yaml because that is the project convention, but their
    contents are valid JSON so Abaqus Python 2.7 can parse them.
    """
    with open(path, "r") as handle:
        return json.load(handle)


def write_json(path, data):
    mkdir_p(os.path.dirname(path))
    with open(path, "w") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def read_json(path):
    with open(path, "r") as handle:
        return json.load(handle)


def write_pickle(path, data):
    mkdir_p(os.path.dirname(path))
    with open(path, "wb") as handle:
        pickle.dump(data, handle, protocol=2)


def read_pickle(path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def as_float(value, default=None):
    if value is None or value == "":
        if default is None:
            raise ValueError("Missing numeric value")
        return float(default)
    return float(value)


def as_int(value, default=None):
    if value is None or value == "":
        if default is None:
            raise ValueError("Missing integer value")
        return int(default)
    return int(round(float(value)))


def to_str(value):
    if value is None:
        return ""
    return str(value)


def s_label(value):
    return "S%03d" % as_int(value)


def geometry_label(geometry_id, D_m, penetration_m, mode):
    if mode != "dimensions":
        return geometry_id
    D_cm = int(round(100.0 * D_m))
    penetration_dm = int(round(10.0 * penetration_m))
    return "%s_D%03d_P%03d" % (geometry_id, D_cm, penetration_dm)


def void_ratio_from_relative_density(ID_percent, e_min, e_max):
    """Invert the density-based relative-density formula used in the legacy notes.

    D = (e_max - e) / (e_max - e_min) * (1 + e_min) / (1 + e)
    """
    d = max(0.0, min(1.0, as_float(ID_percent) / 100.0))
    e_min = as_float(e_min)
    e_max = as_float(e_max)
    a = d * (e_max - e_min)
    b = 1.0 + e_min
    return (b * e_max - a) / max(a + b, EPS)


def void_ratio_settings(config, scenario_id, density_id, ID_percent):
    settings = config.get("void_ratio", {})
    e_min = as_float(settings.get("e_min"), 0.49)
    e_max = as_float(settings.get("e_max"), 0.76)
    reference_scenario_id = to_str(settings.get("reference_scenario_id", "G0_DENS_REF_V_REF"))
    reference_density_id = to_str(settings.get("reference_density_id", ""))
    reference_profile = settings.get("reference_profile", {})
    use_reference = scenario_id == reference_scenario_id
    if reference_density_id:
        use_reference = use_reference or density_id == reference_density_id
    if use_reference:
        soil_void = as_float(reference_profile.get("soil_void"), 0.615854)
        upper = as_float(reference_profile.get("upper_layer"), 0.615854)
        down = as_float(reference_profile.get("down_layer"), 0.55)
        mode = "reference_two_layer"
    else:
        uniform = void_ratio_from_relative_density(ID_percent, e_min, e_max)
        soil_void = uniform
        upper = uniform
        down = uniform
        mode = "uniform_from_ID"
    return {
        "void_ratio_mode": mode,
        "void_ratio_formula": "D=(e_max-e)/(e_max-e_min)*(1+e_min)/(1+e)",
        "void_ratio_e_min": "%.12g" % e_min,
        "void_ratio_e_max": "%.12g" % e_max,
        "use_reference_void_profile": "true" if use_reference else "false",
        "void_ratio_soil_void": "%.12g" % soil_void,
        "void_ratio_upper_layer": "%.12g" % upper,
        "void_ratio_down_layer": "%.12g" % down,
    }


def format_float(value):
    return "%.12g" % float(value)


def mohr_coulomb_settings(config, density_id, soil_model):
    fields = {
        "mc_density": "",
        "mc_E": "",
        "mc_nu": "",
        "mc_phi": "",
        "mc_psi": "",
        "mc_cohesion": "",
        "mc_plastic_strain": "",
    }
    if soil_model != "Mohr-Coulomb":
        return fields

    table = config.get("mohr_coulomb", {})
    values = table.get(density_id)
    if values is None:
        raise ValueError("Missing mohr_coulomb parameters for %s" % density_id)

    fields["mc_density"] = format_float(as_float(values.get("rho", values.get("density"))))
    fields["mc_E"] = format_float(as_float(values.get("E")))
    fields["mc_nu"] = format_float(as_float(values.get("nu")))
    fields["mc_phi"] = format_float(as_float(values.get("phi")))
    fields["mc_psi"] = format_float(as_float(values.get("psi")))
    fields["mc_cohesion"] = format_float(as_float(values.get("c", values.get("cohesion"))))
    fields["mc_plastic_strain"] = format_float(as_float(values.get("plastic_strain"), 0.0))
    return fields


def linspace(start, stop, points):
    points = int(points)
    if points <= 1:
        return [float(start)]
    step = (float(stop) - float(start)) / float(points - 1)
    return [float(start) + i * step for i in range(points)]


def eta_grid_from_config(config):
    grid = config.get("eta_grid", {}) if config else {}
    return linspace(
        as_float(grid.get("start"), 0.02),
        as_float(grid.get("stop"), 0.90),
        as_int(grid.get("points"), 200),
    )


def row_to_jsonable(row):
    out = {}
    for key, value in row.items():
        out[key] = value
    return out


def expand_matrix(config):
    reference_velocity = as_float(config.get("reference_velocity_m_per_s"), 0.5)
    penetration_eta_max = as_float(config.get("penetration_eta_max"), 0.90)
    geometry_label_mode = to_str(config.get("geometry_label_mode", "id"))
    phase = to_str(config.get("phase", "matrix"))
    soil_model = to_str(config.get("soil_model", "Hypoplastisch"))
    run_id_prefix = to_str(config.get("run_id_prefix", "")).strip("_")
    rows = []
    for geometry in config.get("geometries", []):
        geometry_id = to_str(geometry["geometry_id"])
        D_m = as_float(geometry["D_m"])
        penetration_m = as_float(geometry.get("penetration_m"), None)
        if penetration_m is None:
            L_m = as_float(geometry["L_m"])
            penetration_m = penetration_eta_max * L_m
        else:
            L_m = as_float(geometry.get("L_m"), penetration_m / penetration_eta_max)
        L_over_D = as_float(geometry.get("L_over_D"), L_m / D_m)
        penetration_over_D = penetration_m / D_m
        penetration_over_L = penetration_m / L_m
        scenario_geometry_id = geometry_label(geometry_id, D_m, penetration_m, geometry_label_mode)
        for density in config.get("densities", []):
            density_id = to_str(density["density_id"])
            ID_percent = as_float(density["ID_percent"])
            for velocity in config.get("velocities", []):
                velocity_id = to_str(velocity["velocity_id"])
                velocity_m_per_s = as_float(velocity["velocity_m_per_s"])
                base_scenario_id = "%s_%s_%s" % (scenario_geometry_id, density_id, velocity_id)
                if run_id_prefix:
                    scenario_id = "%s_%s" % (run_id_prefix, base_scenario_id)
                else:
                    scenario_id = base_scenario_id
                void_ratios = void_ratio_settings(config, base_scenario_id, density_id, ID_percent)
                mc_values = mohr_coulomb_settings(config, density_id, soil_model)
                for scaling_factor in config.get("scaling_factors", []):
                    S = as_int(scaling_factor)
                    run_id = "%s_%s" % (scenario_id, s_label(S))
                    run_dir = project_path("runs", run_id)
                    rows.append(
                        {
                            "run_id": run_id,
                            "scenario_id": scenario_id,
                            "base_scenario_id": base_scenario_id,
                            "run_id_prefix": run_id_prefix,
                            "phase": phase,
                            "geometry_id": geometry_id,
                            "density_id": density_id,
                            "velocity_id": velocity_id,
                            "D_m": "%.12g" % D_m,
                            "L_m": "%.12g" % L_m,
                            "L_over_D": "%.12g" % L_over_D,
                            "penetration_m": "%.12g" % penetration_m,
                            "penetration_over_D": "%.12g" % penetration_over_D,
                            "penetration_over_L": "%.12g" % penetration_over_L,
                            "ID_percent": "%.12g" % ID_percent,
                            "velocity_m_per_s": "%.12g" % velocity_m_per_s,
                            "v_over_v0": "%.12g" % (velocity_m_per_s / reference_velocity),
                            "S": str(S),
                            "rho_scale": "%.12g" % float(S),
                            "gravity_scale": "%.12g" % (1.0 / float(S)),
                            "pile_type": "deformable_volume",
                            "soil_model": soil_model,
                            "mc_density": mc_values["mc_density"],
                            "mc_E": mc_values["mc_E"],
                            "mc_nu": mc_values["mc_nu"],
                            "mc_phi": mc_values["mc_phi"],
                            "mc_psi": mc_values["mc_psi"],
                            "mc_cohesion": mc_values["mc_cohesion"],
                            "mc_plastic_strain": mc_values["mc_plastic_strain"],
                            "void_ratio_mode": void_ratios["void_ratio_mode"],
                            "void_ratio_formula": void_ratios["void_ratio_formula"],
                            "void_ratio_e_min": void_ratios["void_ratio_e_min"],
                            "void_ratio_e_max": void_ratios["void_ratio_e_max"],
                            "use_reference_void_profile": void_ratios["use_reference_void_profile"],
                            "void_ratio_soil_void": void_ratios["void_ratio_soil_void"],
                            "void_ratio_upper_layer": void_ratios["void_ratio_upper_layer"],
                            "void_ratio_down_layer": void_ratios["void_ratio_down_layer"],
                            "symmetry_factor": str(as_int(config.get("symmetry_factor"), 4)),
                            "run_dir": run_dir,
                            "input_file": os.path.join(run_dir, run_id + ".inp"),
                            "case_config_file": os.path.join(run_dir, "case_config.json"),
                            "odb_file": os.path.join(run_dir, run_id + ".odb"),
                            "sta_file": os.path.join(run_dir, run_id + ".sta"),
                            "postprocess_csv": project_path("data", "extracted", "per_run_csv", run_id + ".csv"),
                            "resampled_csv": project_path("data", "processed", "resampled_runs", run_id + ".csv"),
                            "status": "planned",
                        }
                    )
    if not rows:
        raise ValueError("Matrix config produced no rows")
    return rows


def dict_fieldnames(rows):
    seen = []
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.append(key)
    return seen


def write_csv(path, rows, fieldnames=None):
    mkdir_p(os.path.dirname(path))
    if fieldnames is None:
        fieldnames = dict_fieldnames(rows)
    if sys.version_info[0] >= 3:
        handle = open(path, "w", newline="")
    else:
        handle = open(path, "wb")
    try:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            encoded = {}
            for key in fieldnames:
                value = row.get(key, "")
                encoded[key] = "" if value is None else value
            writer.writerow(encoded)
    finally:
        handle.close()


def read_csv(path):
    if sys.version_info[0] >= 3:
        handle = open(path, "r", newline="")
    else:
        handle = open(path, "rb")
    try:
        return [dict(row) for row in csv.DictReader(handle)]
    finally:
        handle.close()


def mean(values):
    values = [v for v in values if is_finite(v)]
    if not values:
        return float("nan")
    return sum(values) / float(len(values))


def std(values):
    values = [v for v in values if is_finite(v)]
    if len(values) <= 1:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / float(len(values) - 1))


def is_finite(value):
    try:
        return math.isfinite(float(value))
    except AttributeError:
        try:
            value = float(value)
            return not (math.isnan(value) or math.isinf(value))
        except Exception:
            return False
    except Exception:
        return False


def sorted_numeric_rows(rows, column):
    clean = []
    for row in rows:
        try:
            row[column] = as_float(row[column])
            clean.append(row)
        except Exception:
            pass
    clean.sort(key=lambda item: item[column])
    return clean


def collapse_by_x(xs, columns, rows):
    buckets = {}
    for row in rows:
        x = row[xs]
        key = "%.12g" % x
        buckets.setdefault(key, []).append(row)
    out = []
    for key in sorted(buckets.keys(), key=lambda value: float(value)):
        bucket = buckets[key]
        row = {xs: float(key)}
        for column in columns:
            row[column] = mean([safe_float(item.get(column)) for item in bucket])
        out.append(row)
    return out


def safe_float(value, default=float("nan")):
    try:
        if value == "":
            return default
        return float(value)
    except Exception:
        return default


def interpolate(x_src, y_src, x_new):
    pairs = [(x_src[i], y_src[i]) for i in range(len(x_src)) if is_finite(x_src[i]) and is_finite(y_src[i])]
    pairs.sort()
    if len(pairs) < 2:
        return [float("nan") for _ in x_new]
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    out = []
    j = 0
    for x in x_new:
        if x < xs[0] or x > xs[-1]:
            out.append(float("nan"))
            continue
        while j < len(xs) - 2 and xs[j + 1] < x:
            j += 1
        x0, x1 = xs[j], xs[j + 1]
        y0, y1 = ys[j], ys[j + 1]
        if abs(x1 - x0) < EPS:
            out.append(y0)
        else:
            t = (x - x0) / (x1 - x0)
            out.append(y0 + t * (y1 - y0))
    return out


def gradient(values, xs):
    n = len(values)
    out = []
    for i in range(n):
        if n == 1:
            out.append(0.0)
        elif i == 0:
            out.append((values[1] - values[0]) / max(xs[1] - xs[0], EPS))
        elif i == n - 1:
            out.append((values[-1] - values[-2]) / max(xs[-1] - xs[-2], EPS))
        else:
            out.append((values[i + 1] - values[i - 1]) / max(xs[i + 1] - xs[i - 1], EPS))
    return out


def rolling(values, window, func):
    half = int(window // 2)
    out = []
    for i in range(len(values)):
        start = max(0, i - half)
        stop = min(len(values), i + half + 1)
        out.append(func(values[start:stop]))
    return out


def expanding_mean(values):
    out = []
    total = 0.0
    count = 0
    for value in values:
        if is_finite(value):
            total += value
            count += 1
        out.append(total / float(count) if count else float("nan"))
    return out


def cummax(values):
    out = []
    current = float("nan")
    for value in values:
        if is_finite(value):
            if not is_finite(current) or value > current:
                current = value
        out.append(current)
    return out


def add_curve_features(rows):
    rows_by_run = {}
    for row in rows:
        rows_by_run.setdefault(row.get("run_id", ""), []).append(row)
    out = []
    for run_id in rows_by_run:
        group = rows_by_run[run_id]
        group.sort(key=lambda row: safe_float(row.get("eta")))
        eta = [safe_float(row.get("eta")) for row in group]
        qb = [safe_float(row.get("qb_MPa")) for row in group]
        qs = [safe_float(row.get("qs_kPa")) for row in group]
        dqb = gradient(qb, eta)
        dqs = gradient(qs, eta)
        qb_movmean = rolling(qb, 5, mean)
        qs_movmean = rolling(qs, 5, mean)
        qb_movstd = rolling(qb, 5, std)
        qs_movstd = rolling(qs, 5, std)
        qb_cmean = expanding_mean(qb)
        qs_cmean = expanding_mean(qs)
        energy_ratio = []
        for row in group:
            allke = abs(safe_float(row.get("ALLKE"), 0.0))
            allie = abs(safe_float(row.get("ALLIE"), 0.0))
            energy_ratio.append(allke / max(allie, EPS))
        energy_ratio_cummax = cummax(energy_ratio)
        for i, row in enumerate(group):
            D_m = safe_float(row.get("D_m"))
            L_m = safe_float(row.get("L_m"))
            penetration_m = safe_float(row.get("penetration_m"))
            if not is_finite(penetration_m) and is_finite(L_m):
                penetration_m = 0.90 * L_m
            z_m = safe_float(row.get("z_m"))
            S = max(safe_float(row.get("S"), 1.0), 1.0)
            row["logS"] = "%.12g" % math.log(S)
            row["z_over_D"] = "%.12g" % (z_m / max(D_m, EPS))
            row["z_over_penetration"] = "%.12g" % (z_m / max(penetration_m, EPS))
            row["dqb_deta"] = "%.12g" % dqb[i]
            row["dqs_deta"] = "%.12g" % dqs[i]
            row["qb_movmean_5"] = "%.12g" % qb_movmean[i]
            row["qs_movmean_5"] = "%.12g" % qs_movmean[i]
            row["qb_movstd_5"] = "%.12g" % qb_movstd[i]
            row["qs_movstd_5"] = "%.12g" % qs_movstd[i]
            row["qb_cummean"] = "%.12g" % qb_cmean[i]
            row["qs_cummean"] = "%.12g" % qs_cmean[i]
            row["energy_ratio"] = "%.12g" % energy_ratio[i]
            row["energy_ratio_cummax"] = "%.12g" % energy_ratio_cummax[i]
            row["oscillation_index_qb"] = "%.12g" % (qb_movstd[i] / max(abs(qb_movmean[i]), EPS))
            row["oscillation_index_qs"] = "%.12g" % (qs_movstd[i] / max(abs(qs_movmean[i]), EPS))
            row["eta"] = "%.12g" % eta[i]
            row["z_m"] = "%.12g" % z_m
            if not is_finite(L_m):
                row["L_m"] = ""
            out.append(row)
    return out


def resample_run(run_rows, metadata, eta_grid):
    D_m = as_float(metadata["D_m"])
    L_m = as_float(metadata["L_m"])
    numeric_columns = ["qb_MPa", "qs_kPa", "ALLKE", "ALLIE", "time_s", "walltime_h"]
    clean = []
    for row in run_rows:
        z_m = safe_float(row.get("z_m"))
        if not is_finite(z_m):
            continue
        row = row.copy()
        row["z_m"] = z_m
        row["eta"] = z_m / L_m
        for column in numeric_columns:
            row[column] = safe_float(row.get(column))
        clean.append(row)
    clean = sorted_numeric_rows(clean, "eta")
    if len(clean) < 2:
        raise ValueError("Need at least two depth points for %s" % metadata.get("run_id"))
    collapsed = collapse_by_x("eta", numeric_columns + ["z_m"], clean)
    x_src = [row["eta"] for row in collapsed]
    out = []
    interpolated = {}
    for column in numeric_columns:
        interpolated[column] = interpolate(x_src, [row[column] for row in collapsed], eta_grid)
    for i, eta in enumerate(eta_grid):
        row = {}
        for key, value in metadata.items():
            row[key] = value
        row["eta"] = "%.12g" % eta
        row["z_m"] = "%.12g" % (eta * L_m)
        row["D_m"] = "%.12g" % D_m
        row["L_m"] = "%.12g" % L_m
        for column in numeric_columns:
            row[column] = "%.12g" % interpolated[column][i] if is_finite(interpolated[column][i]) else ""
        out.append(row)
    return add_curve_features(out)


def build_paired_dataset(resampled_rows):
    by_scenario = {}
    for row in resampled_rows:
        by_scenario.setdefault(row.get("scenario_id"), []).append(row)
    dataset = []
    for scenario_id in sorted(by_scenario.keys()):
        scenario = by_scenario[scenario_id]
        refs = [row for row in scenario if as_int(row.get("S"), 0) == 1]
        if not refs:
            continue
        ref_by_eta = {}
        for row in refs:
            ref_by_eta["%.12g" % safe_float(row.get("eta"))] = row
        high_by_run = {}
        for row in scenario:
            if as_int(row.get("S"), 0) == 1:
                continue
            high_by_run.setdefault(row.get("run_id"), []).append(row)
        for run_id in sorted(high_by_run.keys()):
            for high in high_by_run[run_id]:
                key = "%.12g" % safe_float(high.get("eta"))
                ref = ref_by_eta.get(key)
                if not ref:
                    continue
                row = high.copy()
                row["qb_fast_MPa"] = high.get("qb_MPa", "")
                row["qs_fast_kPa"] = high.get("qs_kPa", "")
                row["qb_S1_MPa"] = ref.get("qb_MPa", "")
                row["qs_S1_kPa"] = ref.get("qs_kPa", "")
                row["target_residual_qb_MPa"] = "%.12g" % (
                    safe_float(row["qb_S1_MPa"]) - safe_float(row["qb_fast_MPa"])
                )
                row["target_residual_qs_kPa"] = "%.12g" % (
                    safe_float(row["qs_S1_kPa"]) - safe_float(row["qs_fast_kPa"])
                )
                dataset.append(row)
    if not dataset:
        raise ValueError("No paired high-S/S=1 rows could be built")
    return dataset


def clean_ml_rows(rows):
    clean = []
    needed = FEATURE_COLUMNS + TARGET_COLUMNS
    for row in rows:
        ok = True
        for column in needed:
            if not is_finite(safe_float(row.get(column))):
                ok = False
                break
        if ok:
            clean.append(row)
    return clean


def feature_vector(row, feature_columns):
    return [safe_float(row.get(column), 0.0) for column in feature_columns]


def fit_scaler(rows, feature_columns):
    columns = []
    for column in feature_columns:
        values = [safe_float(row.get(column), 0.0) for row in rows]
        mu = mean(values)
        sigma = std(values)
        if not is_finite(sigma) or sigma < EPS:
            sigma = 1.0
        columns.append({"name": column, "mean": mu, "std": sigma})
    return columns


def scaled_vector(row, scaler):
    return [(safe_float(row.get(item["name"]), 0.0) - item["mean"]) / item["std"] for item in scaler]


def solve_linear_system(a, b):
    n = len(b)
    aug = [list(a[i]) + [b[i]] for i in range(n)]
    for col in range(n):
        pivot = col
        pivot_abs = abs(aug[pivot][col])
        for row in range(col + 1, n):
            value = abs(aug[row][col])
            if value > pivot_abs:
                pivot = row
                pivot_abs = value
        if pivot_abs < EPS:
            aug[col][col] += EPS
            pivot_abs = abs(aug[col][col])
        if pivot != col:
            aug[col], aug[pivot] = aug[pivot], aug[col]
        scale = aug[col][col]
        for j in range(col, n + 1):
            aug[col][j] /= scale
        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col]
            if abs(factor) < EPS:
                continue
            for j in range(col, n + 1):
                aug[row][j] -= factor * aug[col][j]
    return [aug[i][n] for i in range(n)]


def fit_ridge_model(rows, feature_columns, target_column, l2):
    scaler = fit_scaler(rows, feature_columns)
    p = len(feature_columns) + 1
    xtx = [[0.0 for _ in range(p)] for _ in range(p)]
    xty = [0.0 for _ in range(p)]
    for row in rows:
        x = [1.0] + scaled_vector(row, scaler)
        y = safe_float(row.get(target_column), 0.0)
        for i in range(p):
            xty[i] += x[i] * y
            for j in range(p):
                xtx[i][j] += x[i] * x[j]
    for i in range(1, p):
        xtx[i][i] += float(l2)
    coefficients = solve_linear_system(xtx, xty)
    return {
        "model_type": "standard_library_ridge",
        "feature_columns": list(feature_columns),
        "target_column": target_column,
        "l2": float(l2),
        "scaler": scaler,
        "coefficients": coefficients,
    }


def predict_ridge(model, row):
    x = [1.0] + scaled_vector(row, model["scaler"])
    return sum(model["coefficients"][i] * x[i] for i in range(len(x)))


def group_kfold_splits(rows, group_column, n_splits, seed):
    groups = sorted(list(set(row.get(group_column, "") for row in rows)))
    rng = random.Random(seed)
    rng.shuffle(groups)
    n_splits = max(2, min(int(n_splits), len(groups)))
    folds = [[] for _ in range(n_splits)]
    for i, group in enumerate(groups):
        folds[i % n_splits].append(group)
    splits = []
    for fold_groups in folds:
        test_groups = set(fold_groups)
        train_rows = [row for row in rows if row.get(group_column, "") not in test_groups]
        test_rows = [row for row in rows if row.get(group_column, "") in test_groups]
        if train_rows and test_rows:
            splits.append((train_rows, test_rows))
    return splits


def regression_metrics(y_true, y_pred, eta_values):
    pairs = []
    for i in range(len(y_true)):
        yt = safe_float(y_true[i])
        yp = safe_float(y_pred[i])
        et = safe_float(eta_values[i]) if eta_values else float(i)
        if is_finite(yt) and is_finite(yp):
            pairs.append((et, yt, yp))
    if not pairs:
        return {"MAE": "", "RMSE": "", "NRMSE": "", "Bias": "", "MaxAE": "", "IAE": ""}
    errors = [yp - yt for _, yt, yp in pairs]
    abs_errors = [abs(e) for e in errors]
    rmse = math.sqrt(mean([e * e for e in errors]))
    y_values = [yt for _, yt, _ in pairs]
    denom = max(max(y_values) - min(y_values), max(mean([abs(y) for y in y_values]), EPS))
    ordered = sorted(pairs)
    iae = 0.0
    for i in range(1, len(ordered)):
        x0, y0, p0 = ordered[i - 1]
        x1, y1, p1 = ordered[i]
        iae += 0.5 * (abs(p0 - y0) + abs(p1 - y1)) * abs(x1 - x0)
    return {
        "MAE": mean(abs_errors),
        "RMSE": rmse,
        "NRMSE": rmse / denom,
        "Bias": mean(errors),
        "MaxAE": max(abs_errors),
        "IAE": iae,
    }


def classify_validity(max_energy_ratio):
    value = safe_float(max_energy_ratio)
    if not is_finite(value):
        return "unknown"
    if value <= 0.05:
        return "valid"
    if value <= 0.10:
        return "caution"
    return "invalid"


def evaluate_predictions(rows, zones):
    by_run = {}
    for row in rows:
        by_run.setdefault(row.get("run_id", ""), []).append(row)
    out = []
    for run_id in sorted(by_run.keys()):
        group = by_run[run_id]
        base = group[0]
        max_energy = max([safe_float(row.get("energy_ratio_cummax"), safe_float(row.get("energy_ratio"))) for row in group])
        for zone_name in sorted(zones.keys()):
            lo, hi = zones[zone_name]
            zone_rows = [row for row in group if safe_float(row.get("eta")) >= lo and safe_float(row.get("eta")) <= hi]
            if not zone_rows:
                continue
            for quantity, unit in [("qb", "MPa"), ("qs", "kPa")]:
                ref = [safe_float(row.get("%s_S1_%s" % (quantity, unit))) for row in zone_rows]
                fast = [safe_float(row.get("%s_fast_%s" % (quantity, unit))) for row in zone_rows]
                pred_res = [safe_float(row.get("pred_residual_%s_%s" % (quantity, unit))) for row in zone_rows]
                corrected = [fast[i] + pred_res[i] for i in range(len(fast))]
                eta = [safe_float(row.get("eta")) for row in zone_rows]
                uncorrected = regression_metrics(ref, fast, eta)
                corrected_metrics = regression_metrics(ref, corrected, eta)
                row = {
                    "run_id": run_id,
                    "scenario_id": base.get("scenario_id", ""),
                    "S": base.get("S", ""),
                    "geometry_id": base.get("geometry_id", ""),
                    "density_id": base.get("density_id", ""),
                    "velocity_id": base.get("velocity_id", ""),
                    "zone": zone_name,
                    "quantity": quantity,
                    "max_energy_ratio": "%.12g" % max_energy if is_finite(max_energy) else "",
                    "validity_class": classify_validity(max_energy),
                }
                for key in uncorrected:
                    row["uncorrected_" + key] = "%.12g" % uncorrected[key] if uncorrected[key] != "" else ""
                    row["corrected_" + key] = "%.12g" % corrected_metrics[key] if corrected_metrics[key] != "" else ""
                if is_finite(uncorrected.get("RMSE")) and is_finite(corrected_metrics.get("RMSE")):
                    row["improvement_ratio"] = "%.12g" % (uncorrected["RMSE"] / max(corrected_metrics["RMSE"], EPS))
                else:
                    row["improvement_ratio"] = ""
                out.append(row)
    return out


def training_ranges(rows):
    checks = {
        "S": "S",
        "ID": "ID_percent",
        "D": "D_m",
        "L_over_D": "L_over_D",
        "velocity": "velocity_m_per_s",
    }
    ranges = {}
    for label, column in checks.items():
        values = [safe_float(row.get(column)) for row in rows if is_finite(safe_float(row.get(column)))]
        ranges[label] = {"min": min(values), "max": max(values)} if values else {"min": None, "max": None}
    return ranges


def extrapolation_warnings(rows, ranges):
    checks = {
        "S": "S",
        "ID": "ID_percent",
        "D": "D_m",
        "L_over_D": "L_over_D",
        "velocity": "velocity_m_per_s",
    }
    warnings = []
    for label, column in checks.items():
        values = [safe_float(row.get(column)) for row in rows if is_finite(safe_float(row.get(column)))]
        if not values or label not in ranges or ranges[label]["min"] is None:
            continue
        if min(values) < ranges[label]["min"] - EPS or max(values) > ranges[label]["max"] + EPS:
            warnings.append(
                "%s=%.12g..%.12g outside trained range %.12g..%.12g"
                % (label, min(values), max(values), ranges[label]["min"], ranges[label]["max"])
            )
    return warnings
