"""Select current, reproducible results for the paper companion release."""
from __future__ import annotations

PUBLIC_PROJECT_URL = 'https://janusmaple.github.io/FoMoFD/'

MAIN_METHODS = {'wm_nc', 'wm_pe', 'wm_pu', 'logpzo', 'rnd', 'logpzo_native', 'rnd_native'}
SEEDS = ('main', 'wm_k4_l4_seed1', 'wm_k4_l4_seed2_local_20261018',
         'wm_k4_l4_seed3_local_20261018')
WM_VARIANTS = set(SEEDS) | {
    'wm_k4_l4_action_none', 'fused_view_joint_k4_l4',
    'wm_k4_l4_data33', 'wm_k4_l4_data66',
    'wm_nc_k1_l4', 'wm_nc_k2_l4', 'wm_nc_k3_l4', 'wm_nc_k6_l4', 'wm_nc_k8_l4',
    'wm_nc_k4_l1', 'wm_nc_k4_l2', 'wm_nc_k4_l6', 'wm_nc_k4_l8',
}
STUDIES = {'main_calibration_sensitivity_500', 'calibration_matched_far',
           'expert_policy_score_distributions'}
SUMMARY_TABLES = {'training_populations', 'policy_success_rates'}
STUDY_FIELDS = {'task', 'view', 'method', 'calibration', 'cohort', 'level', 'n_splits',
                'n_episodes', 'n_windows', 'q25', 'median', 'q75', 'iqr',
                'FAR_mean', 'FAR_sd', 'FDR_mean', 'FDR_sd',
                'episode_AUROC_mean', 'episode_AUROC_sd'}
TITLES = {SEEDS[1]: 'K4 L4 · seed 1', SEEDS[2]: 'K4 L4 · seed 2',
          SEEDS[3]: 'K4 L4 · seed 3', 'ace_normal_bins': 'ACE',
          'act_ensemble': 'AEU · 3 ACT policies'}


def current_result(variant, method):
    if variant == 'main':
        return method in MAIN_METHODS
    return ((variant in WM_VARIANTS and method == 'wm_nc') or
            (variant == 'ace_normal_bins' and method == 'ace') or
            (variant == 'act_ensemble' and method == 'act_ensemble_pos') or
            (variant == 'vlm_qwen_v9' and method == 'qwen_v9'))


def current_rows(rows, *, online=False):
    result = []
    for row in rows:
        if not current_result(row.get('variant'), row.get('method')):
            continue
        if online and not int(row.get('recorded_n') or 0):
            continue
        row = dict(row)
        if row['variant'] in TITLES:
            row['variant_title'] = TITLES[row['variant']]
        result.append(row)
    return result


def select_public_data(data):
    """Apply the same boundary during full builds and both refresh modes."""
    data['offline'] = current_rows(data['offline'])
    data['online'] = current_rows(data['online'], online=True)
    data['pairs'] = [r for r in data['pairs'] if r.get('on_time_only') is True
                     and r.get('status') == 'available'
                     and r.get('family') in {'baseline', 'K', 'L'}
                     and current_result(*r['comparator'].split('|')[:2])]
    data['auc_rows'] = [r for r in data['auc_rows'] if current_result(r[0], r[2])]
    data['tables'] = {k: v for k, v in data.get('tables', {}).items() if k in SUMMARY_TABLES}
    return data
