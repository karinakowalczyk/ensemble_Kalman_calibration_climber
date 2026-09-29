"""
Export the emulator+MCMC calibration setup from calibrate_do_paper.ipynb so the
Julia EKS driver can use exactly the same observation operator, target and
observation uncertainties.

Runs the notebook's own cells (selected by their header comment, executed
verbatim -- nothing re-implemented) inside the DO_emulation_calibration
directory, then writes data/python_calibration_setup.json in this repo:
  - global_x_grid                      (PDF grid)
  - PCA components_ / mean_            (projection = (pdf - mean) @ components.T)
  - obs_targets / obs_sigmas           (5 PCA scores + avg_waiting_time)
  - the compute_summary_stats settings used for ensemble members and the default run

Run locally (needs the PPE data and the notebook):
  ~/miniconda3/envs/emulator/bin/python python/export_python_calibration_setup.py
"""
import json
import os
import subprocess
import sys
from datetime import datetime

os.environ.setdefault('MPLBACKEND', 'Agg')

PY_REPO = '/home/karinako/projects/DO_emulation_calibration'
NOTEBOOK = 'calibrate_do_paper.ipynb'
OUT_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'data', 'python_calibration_setup.json')

# Cells to run, identified by the start of their source (robust to cell reordering).
CELL_HEADERS = [
    '# ── Top-level constants',
    'import os',
    '# ── File list',
    '# ── Default run ──',
    '# ── Pass 1: filter incomplete wide-PPE runs',
    '# ── Pass 2: summary stats — reuse global_x_grid',
    '# ── Default run stats',
    '# ── PCA on DO-PPE PDFs',
    '# ── Observational uncertainty: window analysis',
    '# ── Calibration targets',
]


def git_info(path, *files):
    def run(*args):
        return subprocess.run(['git', '-C', PY_REPO, *args], capture_output=True, text=True).stdout.strip()
    return {
        'commit': run('log', '-1', '--format=%h %ad', '--date=short', '--', *files),
        'uncommitted_changes': bool(run('status', '--short', '--', *files)),
    }


def main():
    os.chdir(PY_REPO)
    sys.path.insert(0, PY_REPO)
    nb = json.load(open(NOTEBOOK))
    sources = [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']

    ns = {}
    for header in CELL_HEADERS:
        matches = [s for s in sources if s.lstrip().startswith(header)]
        if len(matches) != 1:
            raise RuntimeError(f'Expected exactly one cell starting with {header!r}, found {len(matches)}')
        src = matches[0].replace('from tqdm.notebook import tqdm', 'from tqdm import tqdm')
        print(f'\n>>> running cell: {header}')
        exec(compile(src, f'<{header}>', 'exec'), ns)

    pca = ns['pca_model']
    if getattr(pca, 'whiten', False):
        raise RuntimeError('PCA uses whitening -- the plain (pdf - mean) @ components.T projection would be wrong')

    # Settings exactly as passed in the notebook's "Pass 2" (ensemble) and
    # "Default run stats" cells; everything not listed uses summary_stats defaults.
    member_kwargs = dict(
        spinup_fraction=ns['DO_SPINUP_FRACTION'],
        loess_span=ns['DO_LOESS_SPAN'],
        do_min_spacing=ns['DO_MIN_SPACING'],
        do_crossing_value=ns['DO_CROSSING_VALUE'],
        do_peak_threshold=ns['DO_PEAK_THRESHOLD'],
        pre_smooth_win=ns['PRE_SMOOTH_WIN'],
    )
    default_kwargs = dict(
        spinup_fraction=0.02,
        do_crossing_value=5.0,
        detection_mode='peak_walkback',
        do_peak_threshold=ns['DO_PEAK_THRESHOLD'],
    )

    obs_keys = [f'pca_{i}' for i in range(ns['N_PCA'])] + list(ns['MCMC_STAT_TARGETS'])
    setup = {
        'description': 'Calibration setup exported from calibrate_do_paper.ipynb '
                       '(observation operator, target and sigmas of the emulator+MCMC approach)',
        'exported': datetime.now().isoformat(timespec='seconds'),
        'source_repo': PY_REPO,
        'notebook': {'file': NOTEBOOK, **git_info(PY_REPO, NOTEBOOK)},
        'summary_stats_py': git_info(PY_REPO, 'summary_stats.py'),
        'do_ppe_dirs': ns['do_ppe_dirs'],
        'default_run_file': ns['DEFAULT_RUN_FILE'],
        'x_grid': ns['global_x_grid'].tolist(),
        'pca': {
            'n_components': int(pca.n_components_),
            'components': pca.components_.tolist(),       # (n_components, n_grid)
            'mean': pca.mean_.tolist(),                   # (n_grid,)
            'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
            'n_training_pdfs': int(pca.n_samples_),
        },
        'obs_keys': obs_keys,
        'obs_targets': [float(ns['obs_targets'][k]) for k in obs_keys],
        'obs_sigmas': [float(ns['obs_sigmas'][k]) for k in obs_keys],
        'window_analysis': {
            'window_length': ns['WINDOW_LENGTH_OBS'],
            'stride': ns['STRIDE_OBS'],
            'min_do_events': ns['MIN_DO_EVENTS_OBS'],
            'loess_background_yr': ns.get('WINDOW_LOESS_BACKGROUND_YR'),
            'n_valid_windows': len(ns['window_results_obs'][ns['WINDOW_LENGTH_OBS']]),
        },
        'member_stats_kwargs': member_kwargs,
        'default_stats_kwargs': default_kwargs,
    }

    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    with open(OUT_FILE, 'w') as f:
        json.dump(setup, f, indent=1)

    print(f'\nWrote {OUT_FILE}')
    for k, t, s in zip(obs_keys, setup['obs_targets'], setup['obs_sigmas']):
        print(f'  {k:<18} {t:12.4f} ± {s:.4f}')


if __name__ == '__main__':
    main()
