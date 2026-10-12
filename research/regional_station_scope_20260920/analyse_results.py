"""Descriptive, same-row cohort comparisons; never feeds model/rule selection."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import geopandas as gpd
import numpy as np
import pandas as pd
from verify_regional import read, dump, calculated_metrics, stage_rows, MODELS, LOCAL, REFS

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / 'thesis_completion_20260912'
OUT = HERE.parents[1] / 'output/research/regional_station_review_20260929'
COHORTS = ['membership_all', 'documented_role']
COLORS = {'membership_all': '#267b85', 'documented_role': '#ac4164'}
LABELS = {'membership_all': 'Official membership, all sites', 'documented_role': 'Documented-role subset'}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for direction in ['CHN_to_AUS', 'AUS_to_CHN']:
        for seed in [5993, 5994, 5995]:
            predictions = {c: read(HERE / f'results/{c}/{direction}/{seed}/predictions.csv.gz') for c in COHORTS}
            for stage in ['temporal_audit', 'new_city_audit', 'new_grid_audit']:
                parts = {c: stage_rows(p, stage) for c, p in predictions.items()}
                common = set.intersection(*[set(zip(p.eFUA_ID, p.t)) for p in parts.values()])
                for cohort, p in parts.items():
                    z = p[[key in common for key in zip(p.eFUA_ID, p.t)]]
                    values = {m: calculated_metrics(z, m) for m in MODELS}
                    local = min(LOCAL, key=lambda m: values[m]['rmse'])
                    reference = min(REFS, key=lambda m: values[m]['rmse'])
                    for model in MODELS:
                        rows.append({'cohort': cohort, 'direction': direction, 'seed': seed, 'stage': stage,
                            'model': model, 'best_local': local, 'best_control': reference,
                            'gain_vs_local': 1 - values[model]['rmse'] / values[local]['rmse'],
                            'gain_vs_control': 1 - values[model]['rmse'] / values[reference]['rmse'], **values[model]})
    paired = pd.DataFrame(rows)
    paired.to_csv(HERE / 'audit/paired_common_row_metrics.csv', index=False)
    paired.to_csv(OUT / 'paired_common_row_metrics.csv', index=False, encoding='utf-8-sig')
    changes = read(HERE / 'audit/target_changes.csv.gz')
    change_rows = []
    for country, part in changes.groupby('country_iso'):
        common = part[part._merge.eq('both')]
        change_rows.append({'country_iso': country, 'common_months': len(common),
            'removed_months': int(part._merge.eq('left_only').sum()),
            'added_months': int(part._merge.eq('right_only').sum()),
            'mean_delta': common.target_delta.mean(), 'mean_absolute_delta': common.target_delta.abs().mean(),
            'max_absolute_delta': common.target_delta.abs().max(),
            'changed_months': int(common.target_delta.abs().gt(1e-10).sum())})
    pd.DataFrame(change_rows).to_csv(HERE / 'audit/target_change_summary.csv', index=False)
    gates = read(HERE / 'audit/all_gates.csv')
    ledger = read(HERE / 'audit/all_48_settings.csv')
    summary = ledger.groupby(['cohort', 'direction']).agg(settings=('model', 'size'),
        selection_passes=('discovery_pass', 'sum'), joint_passes=('joint_screen', 'sum')).reset_index()
    new_city = gates[gates.stage.eq('new_city_audit')].copy()
    new_city['gain_vs_local'] = 1 - new_city.rmse / new_city.best_local_rmse
    summaries = new_city.groupby(['cohort', 'direction']).agg(best_control_gain=('gain', 'max'),
        local_positive_settings=('gain_vs_local', lambda s: (s > 0).sum()),
        control_positive_settings=('gain', lambda s: (s > 0).sum())).reset_index()
    summary = summary.merge(summaries, on=['cohort', 'direction'], validate='one_to_one')
    summary.to_csv(OUT / 'screen_summary.csv', index=False, encoding='utf-8-sig')
    new_city.to_csv(OUT / 'held_out_city_results.csv', index=False, encoding='utf-8-sig')
    distinct = ledger.drop_duplicates(['cohort', 'direction', 'seed', 'model', 'rule'])
    dump(HERE / 'audit/descriptive_summary.json', {'settings': len(ledger),
        'distinct_rule_model_split_combinations': len(distinct),
        'selection_passes': int(ledger.discovery_pass.sum()),
        'joint_passes': int(ledger.joint_screen.sum()), 'target_changes': change_rows,
        'same_row_comparison_is_postfit_descriptive': True,
        'comparison_method_spotlight': 'regional_kernel (fixed, not selected using audit outcomes)'})
    make_figures(gates, paired)
    print(summary.to_string(index=False), flush=True)
    print(paired[(paired.stage == 'new_city_audit') & (paired.model == 'regional_kernel')][
        ['cohort', 'direction', 'seed', 'n', 'rmse', 'r2', 'gain_vs_local', 'gain_vs_control']].to_string(index=False), flush=True)


def make_figures(gates, paired):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True)
    for ax, direction, title in zip(axes, ['CHN_to_AUS', 'AUS_to_CHN'], ['China to Australia', 'Australia to China']):
        for cohort in COHORTS:
            z = gates[(gates.direction == direction) & (gates.cohort == cohort) & (gates.stage == 'new_city_audit')]
            z = z.drop_duplicates(['seed', 'model', 'rule'])
            ax.scatter(z.coverage * 100, z.gain * 100, color=COLORS[cohort],
                       marker='o' if cohort == COHORTS[0] else '^', s=45, alpha=.8, label=LABELS[cohort])
        ax.axhline(0, color='#333333', lw=.8)
        ax.axhline(5, color='#666666', ls='--', lw=.8)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('Accepted held-out city-month rows (%)')
        ax.set_xlim(0, 100)
        ax.grid(alpha=.18)
    axes[0].set_ylabel('RMSE reduction vs strongest control (%)')
    axes[1].legend(fontsize=8, loc='lower right')
    fig.suptitle('Locked conditions, 2021-2022 new-city audit: all gains below zero', fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / 'locked_coverage_skill.png', dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for ax, direction, title in zip(axes, ['CHN_to_AUS', 'AUS_to_CHN'], ['China to Australia', 'Australia to China']):
        z = paired[(paired.direction == direction) & (paired.stage == 'new_city_audit') & (paired.model == 'regional_kernel')]
        for i, seed in enumerate([5993, 5994, 5995]):
            pair = z[z.seed.eq(seed)].set_index('cohort').reindex(COHORTS)
            ax.plot([0, 1], pair.gain_vs_local * 100, marker=['o', 's', '^'][i],
                    color=['#267b85', '#ac4164', '#6a7027'][i], label=f'Split {seed}')
        ax.axhline(0, color='#333333', lw=.8)
        ax.set_xticks([0, 1], ['All sites', 'Documented-role subset'])
        ax.set_xlim(-.15, 1.15)
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=.18)
        ax.set_ylabel('RMSE reduction vs best local model (%)')
        ax.legend(fontsize=8)
    fig.suptitle('Fixed regional kernel on common FUA-month rows; targets differ by site subset', fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / 'paired_common_rows.png', dpi=170)
    plt.close(fig)

    stations = read(HERE / 'data/station_month_with_roles.csv.gz')
    z = stations[stations.selected_all].drop_duplicates(['site_key', 'longitude', 'latitude'])
    world = gpd.read_file(OLD / 'data/ne_50m_admin_0_countries.zip')
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 6.6))
    for ax, country, code, extent, title in zip(axes, ['CHN', 'AUS'], ['CHN', 'AUS'],
            [(116.4, 123.2, 27.7, 34.4), (142.5, 155.3, -39.5, -18.0)],
            ['YRD 2016 city-membership cohort', 'Observed eastern Australian networks']):
        world[world.ADM0_A3.eq(code)].plot(ax=ax, color='#f1f3f3', edgecolor='#949b9b', linewidth=.6)
        a = z[z.country_iso.eq(country)]
        for include, color, name, marker in [(False, '#ac4164', 'Excluded / unknown in role subset', 'x'),
                                             (True, '#267b85', 'Retained in role subset', 'o')]:
            b = a[a.documented_role.eq(include)]
            ax.scatter(b.longitude, b.latitude, color=color, label=name, marker=marker, s=16, alpha=.8)
        if country == 'CHN':
            ax.add_patch(Rectangle((117, 28), 6, 6, fill=False, ls='--', ec='#8e8e8e', lw=1, label='Earlier rectangular window'))
        ax.set(xlim=extent[:2], ylim=extent[2:], xlabel='Longitude', ylabel='Latitude', title=title)
        ax.legend(fontsize=7, loc='upper left', bbox_to_anchor=(0, -.13))
        ax.grid(alpha=.15)
    fig.suptitle('Monitoring locations, not an administrative boundary or PM2.5 concentration map', fontsize=11)
    fig.tight_layout(rect=[0, .06, 1, .96])
    fig.savefig(OUT / 'station_scope_map.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
