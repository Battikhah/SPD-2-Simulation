import json
from pathlib import Path

import numpy as np
import pandas as pd


def test_notebook_unions_sessions_and_reviewed_duplicate_groups():
    notebook = json.loads(Path('notebooks/TreeVibes_Three_Track_Training.ipynb').read_text(encoding='utf-8'))
    source = next(''.join(cell['source']) for cell in notebook['cells'] if "parents = list(range(len(manifest)))" in ''.join(cell['source']))
    grouping = source[source.index('parents = list(range(len(manifest)))'):source.index('MANIFEST_HASH =')]
    manifest = pd.DataFrame([
        {'group_id': 'a', 'session_id': 's1', 'duplicate_group_id': None, 'pcm_sha256': '1'},
        {'group_id': 'b', 'session_id': 's1', 'duplicate_group_id': 'd1', 'pcm_sha256': '2'},
        {'group_id': 'c', 'session_id': 's2', 'duplicate_group_id': 'd1', 'pcm_sha256': '3'},
        {'group_id': 'd', 'session_id': 's3', 'duplicate_group_id': None, 'pcm_sha256': '4'},
    ])
    exec(grouping, {'manifest': manifest, 'pd': pd})
    assert manifest.component.nunique() == 2
    assert manifest.component.iloc[0] == manifest.component.iloc[2]
    assert manifest.component.iloc[3] != manifest.component.iloc[0]


def test_workbook_missing_site_is_not_treated_as_a_shared_site(tmp_path):
    import hashlib
    import re

    from openpyxl import Workbook

    workbook = Workbook()
    workbook.active.append(['FOLDER', 'IMEI', 'GPS_LAT', 'GPS_LONG', 'VISUAL', 'AUDIO'])
    workbook.active.append(['folder_10', None, 'NA', 'NA', 'X', 0])
    workbook.save(tmp_path / 'TreeVibes.xlsx')
    folder = tmp_path / 'field' / 'folder_10'
    folder.mkdir(parents=True)
    (folder / 'recording.wav').touch()
    notebook = json.loads(Path('notebooks/TreeVibes_Three_Track_Training.ipynb').read_text(encoding='utf-8'))
    source = next(''.join(cell['source']) for cell in notebook['cells'] if 'def workbook_labels(root):' in ''.join(cell['source']))
    source = source[:source.index('if LABELS_CSV is not None:')]
    namespace = {'pd': pd, 'np': np, 're': re, 'digest': lambda x: hashlib.sha256(str(x).encode()).hexdigest()}
    exec(source, namespace)
    labels, _ = namespace['workbook_labels'](tmp_path)
    assert labels.loc[0, 'site_id'] == ''
    assert labels.loc[0, 'device_id'] == ''



def test_workbook_out_of_range_coordinates_are_not_accepted_as_a_site(tmp_path):
    import hashlib
    import re

    from openpyxl import Workbook

    workbook = Workbook()
    workbook.active.append(['FOLDER', 'IMEI', 'GPS_LAT', 'GPS_LONG', 'VISUAL', 'AUDIO'])
    workbook.active.append(['folder_11', None, 95.0, 25.0, 'X', 0])
    workbook.save(tmp_path / 'TreeVibes.xlsx')
    folder = tmp_path / 'field' / 'folder_11'
    folder.mkdir(parents=True)
    (folder / 'recording.wav').touch()
    notebook = json.loads(Path('notebooks/TreeVibes_Three_Track_Training.ipynb').read_text(encoding='utf-8'))
    source = next(''.join(cell['source']) for cell in notebook['cells'] if 'def workbook_labels(root):' in ''.join(cell['source']))
    namespace = {'pd': pd, 'np': np, 're': re, 'digest': lambda x: hashlib.sha256(str(x).encode()).hexdigest()}
    exec(source[:source.index('if LABELS_CSV is not None:')], namespace)
    labels, _ = namespace['workbook_labels'](tmp_path)
    assert labels.loc[0, 'site_id'] == ''
