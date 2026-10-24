#!/usr/bin/env python3
"""Build a portable results + replay site from frozen results (CPU only).

python lerobot/utils/export_review_site.py
python lerobot/utils/export_review_site.py --tasks shunt_insertion --episodes 41 46

Existing videos are reused only when the raw input, validity mask and encoder
settings match. No inference, calibration, label edits or publication occurs.
The output uses local script assets rather than fetch, so index.html also works
from disk. Public data is selected explicitly; full local reports are not copied.
"""
from __future__ import annotations

import argparse
import csv
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
import yaml

from replay_inspector import Recordings, onset_views
from results_inspector import read_viewer_snapshot, resolve_repo_path
from unified_benchmark import curves
from public_results import STUDIES, STUDY_FIELDS, select_public_data

REPO = Path(__file__).resolve().parents[2]
ASSETS = Path(__file__).with_name('review_site')
RESULTS = REPO / 'fomofd/tasks/benchmark_results'
OUTPUT = REPO / 'fomofd/tasks/review_site'
SOURCE_STATE = REPO / 'logs/development/review_site/site_sources.json'
PRIVATE_TABLES = REPO / 'logs/benchmarks/result_tables'
TASKS = ('needle_pickup', 'ring_over_post', 'tissue_retraction', 'shunt_insertion')
METHODS = {'wm_nc': 'WM-NC', 'wm_pe': 'WM-PE', 'wm_pu': 'WM-PU',
           'logpzo': 'logpZO · VAE', 'rnd': 'RND · VAE',
           'logpzo_native': 'logpZO · Policy', 'rnd_native': 'RND · Policy',
           'ace': 'ACE', 'act_ensemble_pos': 'AEU', 'qwen_v9': 'Qwen'}
ROW_FIELDS = ('variant task method view calibration method_name variant_title family '
              'main_comparison supplementary_method group_type failure_mode failure_origin '
              'measurement population_n recorded_n alarm_n unknown_n FAR false_alarm_n '
              'success_test_n FDR early_alarm_n sustained_pre_onset_alarm_n latency_n '
              'latency_mean_s latency_sd_sample_s latency_median_s failure_category '
              'episode_AUROC matched_FAR matched_FDR matched_detected_n matched_false_alarm_n '
              'selected_n selected_detection_rate coverage_status replay_scope late_alarm_n '
              'on_time_alarm_n full_test_on_time_rate').split()
PAIR_FIELDS = ('reference comparator label family status on_time_only shared_n reference_mean_s '
               'reference_sd_s comparator_mean_s comparator_sd_s mean_difference_s').split()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, payload, prefix=''):
    text = json.dumps(payload, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    # Also safe when the script is embedded in a future HTML bundle.
    text = text.replace('<', '\\u003c').replace('\u2028', '\\u2028').replace('\u2029', '\\u2029')
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(prefix + text + (';\n' if prefix else '\n'))
    tmp.replace(path)


def value(v):
    if v == '':
        return None
    if v in ('True', 'False'):
        return v == 'True'
    try:
        n = float(v)
        return int(n) if n.is_integer() else n if math.isfinite(n) else None
    except ValueError:
        return v


def public_rows(path, fields):
    with path.open() as f:
        return [{k: value(row.get(k, '')) for k in fields} for row in csv.DictReader(f)]


def read_tables():
    manifest = json.loads((PRIVATE_TABLES / 'details/manifest.json').read_text())
    if sha(RESULTS / 'index.html') != manifest['source_viewer_sha256']:
        raise ValueError('Detailed tables do not match the viewer; run export_benchmark_tables.py first')
    datasets = {}
    for key, filename, fields in [('offline', 'offline_by_failure.csv', ROW_FIELDS),
                                   ('online', 'online_by_failure.csv', ROW_FIELDS),
                                   ('pairs', 'pairwise_latency_summary.csv', PAIR_FIELDS)]:
        source = PRIVATE_TABLES / 'details' / filename
        if sha(source) != manifest['files']['details/' + filename]['sha256']:
            raise ValueError(f'Detailed table checksum changed: {filename}')
        rows = public_rows(source, fields)
        datasets[key] = rows
    return datasets


def recent_summary_tables(summary):
    """Development-run summaries are local provenance, not public results."""
    return {}


def write_results_csv(output, data):
    """One rectangular CSV for all public result tables; keep exact saved values."""
    records = []

    def add(section, name, rows):
        for index, row in enumerate(rows, 1):
            records.append({'section': section, 'table': name, 'record_index': index,
                            'protocol_id': data['protocol_id'], **row})

    for key, section in [('offline', 'offline_results'), ('online', 'online_results'),
                         ('pairs', 'paired_latency')]:
        add(section, key, data[key])
    add('fusion', 'episode_and_endpoint_fusion', data['fusion'])
    for study in data['studies']:
        add('study_statistics', study['id'], study['rows'])
    for name, table in data['tables'].items():
        # These are formatted summary cells, separate from exact numeric fields.
        fields = ['display_' + re.sub(r'[^a-z0-9]+', '_', c.lower()).strip('_')
                  for c in table['columns']]
        if len(set(fields)) != len(fields):
            raise ValueError(f'Ambiguous summary columns: {name}')
        add('summary_table', name, [dict(zip(fields, row)) for row in table['rows']])

    fields = list(dict.fromkeys(['section', 'table', 'record_index', 'protocol_id',
                                *ROW_FIELDS, *PAIR_FIELDS,
                                *(key for row in records for key in row)]))
    folder = output / 'downloads'
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / 'all_results.csv'
    temporary = target.with_suffix('.csv.tmp')
    with temporary.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    temporary.replace(target)
    # Retire only this exporter's superseded split downloads, never source data.
    for name in ('offline_by_failure.csv', 'online_by_failure.csv', 'pairwise_latency_summary.csv'):
        (folder / name).unlink(missing_ok=True)


def read_scores(snapshot):
    """Use the current frozen reports, including validated ACE/AEU replacements."""
    result = {}
    source_for = {m: snapshot.get('summary_sources', {}).get(m, 'main') for m in METHODS if m != 'qwen_v9'}
    for run in snapshot['runs']:
        selected = [m for m, source in source_for.items() if source == run['id']]
        if not selected:
            continue
        report = json.loads((RESULTS / run['path'] / 'report.json').read_text())
        protocol = report['protocols']['paper_protocol']
        if protocol['window'] != 1 or protocol['wm_frame_policy'] != 'all_endpoints':
            raise ValueError('Replay export requires the frozen all-endpoint, unsmoothed protocol')
        for entry in report['score_caches']:
            method, task = entry['method'], entry['task']
            if method not in selected:
                continue
            path = resolve_repo_path(entry['path'])
            if sha(path) != entry['sha256']:
                raise ValueError(f'Changed score cache: {task}/{method}')
            with np.load(path, allow_pickle=False) as f:
                arrays = {k: f[k] for k in ('episode_index', 'frame_index', 'raw_frame_index', 'wall_timestamp', 'scores')}
            settings = entry['scoring_settings']
            job = SimpleNamespace(task=task, mode=method.startswith('wm_'),
                                  merged={'data': {'history_len': settings['L'], 'pred_horizon': settings['K']}})
            for vi, view in enumerate(entry['views']):
                records = [r for r in report['calibration'] if r['task'] == task and r['method'] == method
                           and r['view'] == view and r['calibration'] == 'episode' and r['protocol'] == 'paper']
                if len(records) != 1 or not math.isfinite(records[0]['threshold']):
                    raise ValueError(f'Missing unique threshold: {task}/{method}/{view}')
                threshold = records[0]['threshold']
                for episode, series in curves(arrays, job, vi, protocol).items():
                    scores = series['score']
                    if not np.isfinite(scores).all():
                        raise ValueError('Nonfinite replay score')
                    result.setdefault((task, episode), []).append({
                        'method': method, 'view': view, 'label': METHODS[method],
                        'frames': series['raw'].astype(int).tolist(), 'scores': scores.tolist(),
                        'threshold': threshold, 'n_calibration': records[0]['n_calibration'],
                        'ode_steps': settings.get('ode_steps_per_solve')})
    return result


def encode_video(episode, stop, target):
    """One paired-view VFR video; PTS derives from recorded wall time, not FPS."""
    import av
    times = episode.times[:stop]
    if episode.timing_note or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError(f'Cannot faithfully export invalid/non-increasing timestamps: {episode.task}/{episode.episode_id}')
    # MP4 supports a microsecond time base; both cameras are in one frame and
    # cannot drift apart during browser playback.
    tmp = target.with_suffix('.partial.mp4')
    width = episode.fix.shape[2] + episode.wrist.shape[2]
    height = max(episode.fix.shape[1], episode.wrist.shape[1])
    with av.open(str(tmp), mode='w', options={'movflags': '+faststart'}) as container:
        stream = container.add_stream('libx264', rate=30)
        stream.width, stream.height, stream.pix_fmt = width, height, 'yuv420p'
        stream.time_base = Fraction(1, 1000000)
        stream.codec_context.time_base = Fraction(1, 1000000)
        stream.codec_context.thread_count = 1
        stream.options = {'crf': '24', 'preset': 'veryfast', 'bf': '0', 'g': '30'}
        for i, elapsed in enumerate(times):
            rgb = np.zeros((height, width, 3), dtype=np.uint8)
            rgb[:episode.fix.shape[1], :episode.fix.shape[2]] = episode.fix[i]
            rgb[:episode.wrist.shape[1], episode.fix.shape[2]:] = episode.wrist[i]
            frame = av.VideoFrame.from_ndarray(rgb, format='rgb24')
            frame.pts, frame.time_base = round(float(elapsed) * 1000000), Fraction(1, 1000000)
            for packet in stream.encode(frame):
                container.mux(packet)
            if i == 0:
                Image.fromarray(rgb).save(target.with_suffix('.jpg'), quality=88)
        for packet in stream.encode():
            container.mux(packet)
    with av.open(str(tmp)) as f:
        pts = [float(frame.time) for frame in f.decode(video=0)]
    if len(pts) != len(times) or not np.allclose(pts, times, atol=2e-6, rtol=0):
        raise ValueError('Encoded video timestamps do not match the recording')
    tmp.replace(target)


def scan_public(output):
    """Fail closed on known identity/path leaks in exported text and media tags.

    This is not a visual audit of faces/logos inside the recorded RGB frames.
    """
    private_terms = REPO / 'logs/benchmarks/private_release_terms.json'
    forbidden = ['/home/', '/storage/project/', 'github.io']
    if private_terms.is_file():
        forbidden += json.loads(private_terms.read_text())
    for path in output.rglob('*'):
        if path.is_file() and path.suffix in ('.html', '.js', '.css', '.json', '.csv', '.md', '.svg'):
            text = path.read_text().lower()
            for term in forbidden:
                if term in text:
                    raise ValueError(f'Identity/path leak in {path.relative_to(output)}: {term}')


def style_study_svg(text):
    """Restyle website copies, retaining scientific paths and embedded glyphs."""
    palette = {'#2477a8': '#d19a2a', '#d57826': '#647986',
               '#2563a6': '#d19a2a', '#d99018': '#a87312',
               '#15936d': '#758044', '#bf4781': '#647986'}
    text = re.sub(r'#[0-9a-fA-F]{6}\b', lambda m: palette.get(m[0].lower(), m[0]), text)
    if 'WM-NC' not in text:
        return text
    # Matplotlib saved these labels as glyph outlines, not SVG <text>. Replace
    # only those outlines; keep their defs, which other labels may also reuse.
    from matplotlib.font_manager import FontProperties
    from matplotlib.path import Path as MplPath
    from matplotlib.textpath import TextPath, TextToPath

    namespace = 'http://www.w3.org/2000/svg'
    tag = lambda name: f'{{{namespace}}}{name}'
    ET.register_namespace('', namespace)
    ET.register_namespace('xlink', 'http://www.w3.org/1999/xlink')
    root = ET.fromstring(text, parser=ET.XMLParser(target=ET.TreeBuilder(insert_comments=True)))
    parents = {child: parent for parent in root.iter() for child in parent}
    x0, y0, width, height = map(float, root.get('viewBox').split())
    left, right = x0, x0 + width
    normal = FontProperties(family='DejaVu Sans', size=100)
    bold = FontProperties(family='DejaVu Sans', weight='bold', size=100)
    metrics = TextToPath()
    commands = {MplPath.MOVETO: 'M', MplPath.LINETO: 'L', MplPath.CURVE3: 'Q',
                MplPath.CURVE4: 'C', MplPath.CLOSEPOLY: 'z'}
    for group in root.iter(tag('g')):
        comment = next((c for c in group if c.tag is ET.Comment and 'WM-NC' in (c.text or '')), None)
        if comment is None:
            continue
        drawing = group.find(tag('g'))
        if drawing is None or drawing.get('data-method-label'):
            continue
        transform = re.fullmatch(r'translate\(([-\d.e]+) ([-\d.e]+)\) scale\(([-\d.e]+) ([-\d.e]+)\)',
                                 drawing.get('transform', ''))
        if not transform:
            raise ValueError('Unexpected WM-NC vector label transform')
        tx, ty, sx, sy = map(float, transform.groups())
        label = comment.text.strip()
        old_width = metrics.get_text_width_height_descent(label, normal, ismath=False)[0]
        paths, advance = [], 0.
        parts = re.split(r'(WM-NC(?:\s*\(ours\))?)', label)
        for part in filter(None, parts):
            ours = part.startswith('WM-NC')
            part = 'WM-NC (ours)' if ours else part
            prop = bold if ours else normal
            path = TextPath((advance, 0), part, size=100, prop=prop)
            paths.append(path)
            advance += metrics.get_text_width_height_descent(part, prop, ismath=False)[0]
        # Method ticks are right aligned; axis labels and titles are centered.
        parent_id = parents[group].get('id', '')
        tx -= (advance - old_width) * sx * (1 if parent_id.startswith('ytick_') else .5)
        drawing.set('transform', f'translate({tx:.8f} {ty:.8f}) scale({sx} {sy})')
        updated = re.sub(r'WM-NC(?:\s*\(ours\))?', 'WM-NC (ours)', label)
        drawing.set('data-method-label', updated)
        drawing.set('aria-label', updated)
        comment.text = f' {updated} '
        for child in list(drawing):
            if child.tag != tag('defs'):
                drawing.remove(child)
        for path in paths:
            commands_svg = []
            for vertices, code in path.iter_segments(curves=True, simplify=False):
                coords = '' if code == MplPath.CLOSEPOLY else ' '.join(f'{v:.8f}' for v in vertices)
                commands_svg.append(commands[code] + coords)
            ET.SubElement(drawing, tag('path'), d=' '.join(commands_svg))
            bounds = path.get_extents()
            left = min(left, tx + sx * bounds.xmin - 7.2)
            right = max(right, tx + sx * bounds.xmax + 7.2)
    if left < x0 or right > x0 + width:
        # Expand only the canvas: every plotted coordinate stays unchanged.
        root.set('viewBox', f'{left:.8f} {y0} {right - left:.8f} {height}')
        root.set('width', f'{right - left:.8f}pt')
        root.insert(0, ET.Element(tag('rect'), x=str(left), y=str(y0),
                                 width=str(right - left), height=str(height), fill='white'))
    return ET.tostring(root, encoding='unicode')


def publish_site(output, data, snapshot):
    """Write current results without development records or obsolete assets."""
    select_public_data(data)
    public_studies = []
    matched = next((s for s in snapshot.get('reviewer_evidence', {}).get('sections', [])
                    if s['id'] == 'calibration_matched_far'), None)
    if matched:
        public_studies.append({'id': matched['id'], 'title': matched['title'],
                               'kind': 'matched_far_sensitivity',
                               'rows': [{k: v for k, v in r.items() if k in STUDY_FIELDS}
                                        for r in matched['records']]})
    for study in snapshot.get('studies', []):
        if study['id'] not in STUDIES:
            continue
        source_rows = study.get('summary', [])
        if study['id'] == 'main_calibration_sensitivity_500':
            # Reuse per-split pooled statistics; averaging task SDs is invalid.
            source_rows = [*source_rows, *study.get('overall', [])]
        if study['id'] == 'expert_policy_score_distributions':
            source_rows = [r for r in source_rows if r.get('level') == 'episode_maximum']
        rows = [{k: v for k, v in row.items() if k in STUDY_FIELDS} for row in source_rows]
        public_studies.append({'id': study['id'], 'title': study['title'], 'kind': study['kind'],
                               'rows': rows})
    data['studies'] = public_studies
    fusion = snapshot.get('fusion_level_comparison', {})
    data['fusion'] = [{k: r.get(k) for k in ('method', 'view', 'level', 'task', 'label', 'subset', 'failure_mode',
                                          'success_n', 'false_alarm_n', 'failure_n', 'detected_n', 'unknown_n',
                                          'FAR', 'FDR', 'episode_AUROC')}
                      for r in fusion.get('metrics', []) if r.get('annotator_id') == snapshot['pairwise_latency']['annotator_id']]
    write_results_csv(output, data)
    # These figures are no longer linked by the page. Remove only assets made
    # by earlier versions of this exporter, leaving unknown files for review.
    for path in (output / 'figures').glob('*'):
        if path.is_file() and path.name.startswith(('expert_policy_', 'calibration_', 'main_calibration_')):
            path.unlink()
    shutil.copyfile(ASSETS / 'protocol.md', output / 'downloads/protocol.md')
    for path in ASSETS.iterdir():
        if path.suffix in ('.html', '.css', '.js'):
            shutil.copyfile(path, output / path.name)
    # Packed rows avoid repeating column names and split files stay small enough
    # for common static/anonymized-file hosts. No external CDN is required.
    base = {k: v for k, v in data.items() if k not in ('offline', 'online', 'pairs', 'auc_rows')}
    base.update(offline=[], online=[], pairs=[], auc_rows=[], row_fields=ROW_FIELDS, pair_fields=PAIR_FIELDS)
    write_json(output / 'data/report.js', base, 'window.REVIEW_DATA=')
    scripts = []
    for old in (output / 'data').glob('chunk_*.js'):
        old.unlink()
    for key, fields in [('offline', ROW_FIELDS), ('online', ROW_FIELDS), ('pairs', PAIR_FIELDS), ('auc_rows', None)]:
        rows = [[r.get(k) for k in fields] for r in data[key]] if fields else data[key]
        for start in range(0, len(rows), 5000):
            filename = f'data/chunk_{key}_{start//5000:02d}.js'
            path = output / filename
            write_json(path, rows[start:start+5000], f'window.REVIEW_DATA.{key}.push(...')
            path.write_text(path.read_text().removesuffix(';\n') + ');\n')
            scripts.append(f'<script src="{filename}" defer></script>')
    index = output / 'index.html'
    index.write_text(index.read_text().replace('<!-- DATA_CHUNKS -->', '\n'.join(scripts)))
    state = json.loads(SOURCE_STATE.read_text()) if SOURCE_STATE.exists() else {}
    state[str(output.resolve())] = sha(RESULTS / 'index.html')
    write_json(SOURCE_STATE, state)


def refresh_presentation(output):
    output = output.resolve()
    state = json.loads(SOURCE_STATE.read_text())
    if state.get(str(output)) != sha(RESULTS / 'index.html'):
        raise ValueError('Source results changed; rebuild the export before refreshing its presentation')
    data = json.loads((output / 'data/report.js').read_text().removeprefix('window.REVIEW_DATA=').removesuffix(';\n'))
    for path in sorted((output / 'data').glob('chunk_*.js')):
        match = re.fullmatch(r'window\.REVIEW_DATA\.(\w+)\.push\(\.\.\.(.*)\);\n', path.read_text(), re.S)
        if not match:
            raise ValueError(f'Unexpected data chunk: {path.name}')
        data[match[1]].extend(json.loads(match[2]))
    for key, fields in [('offline', ROW_FIELDS), ('online', ROW_FIELDS), ('pairs', PAIR_FIELDS)]:
        data[key] = [dict(zip(fields, row)) for row in data[key]]
    snapshot = read_viewer_snapshot(RESULTS / 'index.html')
    publish_site(output, data, snapshot)
    scan_public(output)
    manifest = json.loads((output / 'manifest.json').read_text())
    manifest['bytes'] = sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
    write_json(output / 'manifest.json', manifest)
    print(f'Presentation refreshed; recording videos and score curves reused: {output}')


def refresh_results(output):
    """Refresh tables and onset markers using existing media, with score checks."""
    from benchmark_consensus_onsets import derive_labels
    output = output.resolve()
    snapshot = read_viewer_snapshot(RESULTS / 'index.html')
    config = yaml.safe_load((REPO / 'lerobot/configs/benchmark/results_catalog.yaml').read_text())
    spec = snapshot['evaluation_protocol']
    labels_path = REPO / spec['frozen_inputs']['onsets']['path']
    if sha(labels_path) != spec['frozen_inputs']['onsets']['sha256']:
        raise ValueError('Report onset snapshot changed')
    labels = derive_labels(yaml.safe_load(labels_path.read_text()), spec, resolve_repo_path)
    primary = spec['detection']['primary_annotator']
    old = json.loads((output / 'data/report.js').read_text().removeprefix('window.REVIEW_DATA=').removesuffix(';\n'))
    scores = read_scores(snapshot)
    count = 0
    for file in sorted((output / 'data/episodes').glob('*/*.js')):
        prefix, text = file.read_text().split('=', 1)
        payload = json.loads(text.removesuffix(';\n'))
        task, ep = payload['task'], payload['episode']
        # A report update may not silently change the replay score curves.
        if payload['curves'] != scores.get((task, ep), []):
            raise ValueError(f'Replay scores/thresholds changed: {task}/{ep}; use a full export')
        entries = labels['tasks'][task].get('failure_onsets', {}).get(ep, {})
        views = onset_views(entries.get(primary)) if primary in entries else {}
        payload['onsets'] = {v: a['frame_index'] for v, a in views.items()}
        payload['onset_times'] = {v: a.get('elapsed_seconds', payload['times'][a['frame_index']]) for v, a in views.items()}
        write_json(file, payload, prefix+'=')
        count += 1
    data = read_tables()
    keys = ('training_populations', 'policy_success_rates', 'main_calibration_sensitivity_500',
            'calibration_sensitivity_200', 'expert_policy_score_distributions')
    summary = yaml.safe_load((RESULTS / 'summary.yaml').read_text())
    columns = ['variant', 'task', 'method', 'view', 'calibration', 'episode_index', 'score_max']
    indices = [snapshot['episodes']['columns'].index(k) for k in columns]
    data.update(catalog=old['catalog'], methods=METHODS, training=old['training'],
                protocol_id=spec['protocol_id'], media_count=count, total_episodes=old['total_episodes'],
                tables={**{k: summary['tables'][k] for k in keys if k in summary['tables']},
                        **recent_summary_tables(summary)},
                auc_rows=[[row[i] for i in indices] for row in snapshot['episodes']['rows']])
    publish_site(output, data, snapshot)
    scan_public(output)
    manifest = json.loads((output / 'manifest.json').read_text())
    manifest.update(protocol=spec['protocol_id'], bytes=sum(p.stat().st_size for p in output.rglob('*') if p.is_file()))
    write_json(output / 'manifest.json', manifest)
    print(f'Results and {count} episode onset markers refreshed; all videos reused: {output}')


def build(output, tasks=TASKS, episodes=None):
    output = output.resolve()
    if output in (REPO, RESULTS) or output.is_relative_to(ASSETS):
        raise ValueError('Choose a separate generated-site directory')
    output.mkdir(parents=True, exist_ok=True)
    snapshot = read_viewer_snapshot(RESULTS / 'index.html')
    if snapshot.get('offline_protocol', {}).get('wm_frame_policy') != 'all_endpoints':
        raise ValueError('Expected the current frozen results snapshot')
    config = yaml.safe_load((REPO / 'lerobot/configs/benchmark/results_catalog.yaml').read_text())
    labels_path = REPO / 'lerobot/configs/benchmark/prl_episode_labels.yaml'
    labels = yaml.safe_load(labels_path.read_text())
    from benchmark_consensus_onsets import derive_labels
    labels = derive_labels(labels, snapshot.get('evaluation_protocol'), resolve_repo_path)
    # The viewer primary labels must describe this exact label file.
    recorded_hash = snapshot['pairwise_latency'].get('onset_labels_sha256')
    if recorded_hash and sha(labels_path) != recorded_hash:
        raise ValueError('Labels changed since the report; refresh the canonical reports first')
    recordings = Recordings(REPO, labels_path)
    scores = read_scores(snapshot)
    data = read_tables()
    catalog = []
    private_cache = REPO / 'logs/development/review_site/media_cache.json'
    cache = json.loads(private_cache.read_text()) if private_cache.exists() else {}
    correction = snapshot['offline_protocol']['correction']
    primary = snapshot['pairwise_latency']['annotator_id']
    total = sum(1 for task, episode in recordings.paths if task in tasks and (episodes is None or episode in episodes))
    n = 0
    for task_node in recordings.catalog['tasks']:
        task = task_node['id']
        public_task = {'id': task, 'title': task_node['title'], 'groups': []}
        for group in task_node['groups']:
            public_group = {k: group.get(k) for k in ('title', 'split', 'outcome', 'failure_mode', 'failure_origin')}
            public_group['episodes'] = []
            for episode_id in group['episode_ids']:
                available = task in tasks and (episodes is None or episode_id in episodes)
                public_group['episodes'].append({'id': episode_id, 'available': available})
                if not available:
                    continue
                ep = recordings.episode(task, episode_id)
                stop = len(ep.times)
                corrected = task == correction['task'] and episode_id == correction['episode_index']
                if corrected:
                    stop = min(stop, correction['raw_frame_stop'])
                key = f'{task}/{episode_id:06d}'
                target = output / 'media' / f'{key}.mp4'
                target.parent.mkdir(parents=True, exist_ok=True)
                source_hash = sha(recordings.paths[task, episode_id])
                identity = f'{source_hash}:{stop}:vfr-us-x264-crf24-v1'
                cached = cache.get(str(target), {})
                if (cached.get('source') != identity or not target.exists() or not target.with_suffix('.jpg').exists()
                        or sha(target) != cached.get('sha256')):
                    encode_video(ep, stop, target)
                    cache[str(target)] = {'source': identity, 'sha256': sha(target)}
                    write_json(private_cache, cache)
                annotation = labels['tasks'][task].get('failure_onsets', {}).get(episode_id, {}).get(primary)
                views = onset_views(annotation) if annotation else {}
                onsets = {v: entry['frame_index'] for v, entry in views.items()}
                if group['outcome'] == 'success' and onsets:
                    raise ValueError('A successful episode cannot have a failure onset')
                for curve in scores.get((task, episode_id), []):
                    if max(curve['frames']) >= stop:
                        raise ValueError('Replay curve exceeds the valid recording support')
                payload = {'task': task, 'episode': episode_id, 'times': ep.times[:stop].tolist(),
                           'video': f'media/{key}.mp4', 'poster': f'media/{key}.jpg',
                           'onsets': onsets, 'onset_times': {v: entry.get('elapsed_seconds', float(ep.times[entry['frame_index']])) for v, entry in views.items()}, 'corrected': corrected,
                           'curves': scores.get((task, episode_id), [])}
                write_json(output / 'data/episodes' / f'{key}.js', payload,
                           f'window.REVIEW_EPISODES[{json.dumps(key)}]=')
                n += 1
                print(f'{n}/{total} {key}: {stop} frames, {ep.times[stop-1]:.2f}s', flush=True)
            public_task['groups'].append(public_group)
        catalog.append(public_task)
    setup = config['training_setup']
    summary = yaml.safe_load((RESULTS / 'summary.yaml').read_text())
    # Explicitly chosen numerical tables retain detailed studies without the
    # local source paths, names, provenance files or machine identifiers.
    table_names = ('training_populations', 'policy_success_rates', 'main_calibration_sensitivity_500',
                   'calibration_sensitivity_200', 'expert_policy_score_distributions')
    auc_columns = ['variant', 'task', 'method', 'view', 'calibration', 'episode_index', 'score_max']
    indices = [snapshot['episodes']['columns'].index(k) for k in auc_columns]
    auc_rows = [[row[i] for i in indices] for row in snapshot['episodes']['rows']]
    data.update({'catalog': catalog, 'methods': METHODS, 'auc_rows': auc_rows,
                 'protocol_id': snapshot['evaluation_protocol']['protocol_id']
                 if 'protocol_id' in snapshot['evaluation_protocol'] else snapshot['offline_protocol']['id'],
                 'training': {'tasks': setup['tasks'], 'operator_count': 1,
                              'act': setup['act']['summary'], 'visual_vae': setup['visual_vae']['summary']},
                 'tables': {**{k: summary['tables'][k] for k in table_names if k in summary['tables']},
                            **recent_summary_tables(summary)},
                 'media_count': n, 'total_episodes': len(recordings.paths)})
    publish_site(output, data, snapshot)
    (output / '.nojekyll').write_text('')
    (output / 'robots.txt').write_text('User-agent: *\nDisallow: /\n')
    scan_public(output)
    manifest = {'schema_version': 1, 'protocol': data['protocol_id'], 'episodes': n,
                'text_identity_scan': 'passed', 'visual_identity_audit': 'not_performed',
                'video_timing': 'microsecond VFR PTS verified against recorded wall timestamps',
                'bytes': sum(p.stat().st_size for p in output.rglob('*') if p.is_file())}
    write_json(output / 'manifest.json', manifest)
    print(json.dumps({'output': str(output), **manifest}, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=OUTPUT)
    p.add_argument('--tasks', nargs='+', choices=TASKS, default=list(TASKS))
    p.add_argument('--episodes', nargs='+', type=int, help='Optional development-only media subset; results stay complete')
    p.add_argument('--refresh-presentation', action='store_true', help='Refresh the existing site without reading/encoding raw recordings')
    p.add_argument('--refresh-results', action='store_true', help='Refresh tables and onsets; verify unchanged score curves and reuse all media')
    args = p.parse_args()
    if args.refresh_results and args.refresh_presentation:
        p.error('Choose one refresh mode')
    if args.refresh_results:
        refresh_results(args.output)
    elif args.refresh_presentation:
        refresh_presentation(args.output)
    else:
        build(args.output, args.tasks, args.episodes)


if __name__ == '__main__':
    main()
