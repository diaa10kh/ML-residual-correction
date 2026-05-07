from __future__ import print_function

import csv
import json
import math
import os
import sys

PY2 = sys.version_info[0] < 3


def _arg_value(flag):
    if flag in sys.argv:
        idx = sys.argv.index(flag)
        if idx + 1 < len(sys.argv):
            return sys.argv[idx + 1]
    return None


def load_config():
    path = os.environ.get('MGS_POSTPROCESS_CONFIG') or _arg_value('--config')
    if not path:
        raise RuntimeError('Missing --config postprocess_config.json')
    with open(path, 'r') as handle:
        return json.load(handle)


def native_path(path):
    if PY2:
        try:
            if isinstance(path, unicode):
                return path.encode('utf-8')
        except NameError:
            pass
    return path


def open_odb(path):
    path = native_path(path)
    try:
        from odbAccess import openOdb
        return openOdb(path=path, readOnly=True)
    except Exception:
        from abaqus import session
        return session.openOdb(name=path)


def history_series(step, variable_name):
    best = []
    for region_name in step.historyRegions.keys():
        region = step.historyRegions[region_name]
        output_name = None
        if variable_name in region.historyOutputs.keys():
            output_name = variable_name
        else:
            prefix = variable_name + ' '
            for candidate in region.historyOutputs.keys():
                if candidate.startswith(prefix):
                    output_name = candidate
                    break
        if output_name:
            data = list(region.historyOutputs[output_name].data)
            if len(data) > len(best):
                best = data
    return [(float(item[0]), float(item[1])) for item in best]


def value_at(series, time_value):
    if not series:
        return ''
    best_value = series[0][1]
    best_dist = abs(series[0][0] - time_value)
    for t, value in series[1:]:
        dist = abs(t - time_value)
        if dist < best_dist:
            best_dist = dist
            best_value = value
    return best_value


def safe_float(value, default=0.0):
    try:
        if value == '':
            return default
        return float(value)
    except Exception:
        return default


def write_rows(path, rows):
    folder = os.path.dirname(path)
    if folder and not os.path.isdir(folder):
        os.makedirs(folder)
    fieldnames = [
        'run_id',
        'scenario_id',
        'S',
        'z_m',
        'qb_MPa',
        'qs_kPa',
        'ALLKE',
        'ALLIE',
        'time_s',
        'walltime_h',
    ]
    with open(path, 'wb') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    config = load_config()
    odb_path = config.get('odb_path') or config.get('odb_file')
    if not odb_path or not os.path.exists(odb_path):
        raise RuntimeError('ODB not found: %s' % odb_path)

    odb = open_odb(odb_path)
    step_name = config.get('step_name', 'Einpressen')
    if step_name not in odb.steps.keys():
        raise RuntimeError('Step not found in ODB: %s' % step_name)
    step = odb.steps[step_name]

    u3 = history_series(step, 'U3')
    cfn3 = history_series(step, 'CFN3')
    cfs3 = history_series(step, 'CFS3')
    allke = history_series(step, 'ALLKE')
    allie = history_series(step, 'ALLIE')
    if not u3:
        raise RuntimeError('No U3 history output found in %s' % odb_path)

    D_m = safe_float(config.get('D_m'))
    symmetry_factor = safe_float(config.get('symmetry_factor'), 4.0)
    force_unit_scale_to_kN = safe_float(config.get('force_unit_scale_to_kN'), 1.0)
    base_area = math.pi * D_m * D_m / 4.0

    rows = []
    for time_value, u3_value in u3:
        z_m = max(0.0, -float(u3_value))
        cfn = abs(safe_float(value_at(cfn3, time_value))) * force_unit_scale_to_kN * symmetry_factor
        cfs = abs(safe_float(value_at(cfs3, time_value))) * force_unit_scale_to_kN * symmetry_factor
        qb_MPa = cfn / max(base_area, 1.0e-12) / 1000.0
        shaft_area = math.pi * D_m * max(z_m, 1.0e-9)
        qs_kPa = cfs / max(shaft_area, 1.0e-12)
        rows.append(
            {
                'run_id': config.get('run_id', ''),
                'scenario_id': config.get('scenario_id', ''),
                'S': config.get('S', ''),
                'z_m': '%.12g' % z_m,
                'qb_MPa': '%.12g' % qb_MPa,
                'qs_kPa': '%.12g' % qs_kPa,
                'ALLKE': '%.12g' % safe_float(value_at(allke, time_value)),
                'ALLIE': '%.12g' % safe_float(value_at(allie, time_value)),
                'time_s': '%.12g' % time_value,
                'walltime_h': config.get('walltime_h', ''),
            }
        )

    output_csv = config.get('output_csv') or config.get('postprocess_csv')
    write_rows(output_csv, rows)
    try:
        odb.close()
    except Exception:
        pass
    print('Wrote %d rows to %s' % (len(rows), output_csv))


if __name__ == '__main__':
    main()
