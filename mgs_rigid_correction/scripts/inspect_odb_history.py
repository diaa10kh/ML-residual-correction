from __future__ import print_function

import os
import sys

PY2 = sys.version_info[0] < 3


def native_path(path):
    if PY2:
        try:
            if isinstance(path, unicode):
                return path.encode('utf-8')
        except NameError:
            pass
    return path


def arg_value(flag):
    if flag in sys.argv:
        idx = sys.argv.index(flag)
        if idx + 1 < len(sys.argv):
            return sys.argv[idx + 1]
    return ''


def open_odb(path):
    path = native_path(path)
    try:
        from odbAccess import openOdb
        return openOdb(path=path, readOnly=True)
    except Exception:
        from abaqus import session
        return session.openOdb(name=path)


def main():
    odb_path = arg_value('--odb')
    if not odb_path:
        raise RuntimeError('Missing --odb path')
    if not os.path.exists(odb_path):
        raise RuntimeError('ODB not found: %s' % odb_path)
    odb = open_odb(odb_path)
    for step_name in odb.steps.keys():
        step = odb.steps[step_name]
        print('STEP: %s' % step_name)
        for region_name in step.historyRegions.keys():
            region = step.historyRegions[region_name]
            names = list(region.historyOutputs.keys())
            print('REGION: %s' % region_name)
            print('OUTPUTS: %s' % ', '.join(names))
    try:
        odb.close()
    except Exception:
        pass


if __name__ == '__main__':
    main()
