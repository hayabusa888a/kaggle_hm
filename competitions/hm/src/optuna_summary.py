"""Optuna のログから最良試行を復元する。

exp_16 を途中で打ち切ると study オブジェクトが失われるため、
ログ行（trial番号 / MAP@12 / lr / leaves / rounds）から上位を並べ直す。
lr と leaves 以外のパラメータは study に無いと分からないので、
探索を止めた場合は上位の lr/leaves/rounds を使って再現する。
"""
import re
import sys
from pathlib import Path

path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    '/workspace/outputs/experiments/exp16_optuna.log')
pat = re.compile(
    r'trial\s+(\d+)\s+MAP@12=([\d.]+)\s+lr=([\d.]+)\s+leaves=(\d+)\s+rounds=(\d+)\s+\((\d+)s\)')

rows = []
for line in path.read_text(encoding='utf-8').splitlines():
    m = pat.search(line)
    if m:
        rows.append({'trial': int(m.group(1)), 'map': float(m.group(2)),
                     'lr': float(m.group(3)), 'leaves': int(m.group(4)),
                     'rounds': int(m.group(5)), 'sec': int(m.group(6))})

rows.sort(key=lambda r: -r['map'])
base = 0.04104   # 現行構成と同条件の基準（trial 0）
print('{} 試行  基準(trial 0)={:.5f}\n'.format(len(rows), base))
print('{:>6}{:>10}{:>10}{:>9}{:>8}{:>8}{:>10}'.format(
    'trial', 'MAP@12', '基準比', 'lr', 'leaves', 'rounds', 'sec'))
for r in rows:
    print('{:>6}{:>10.5f}{:>+10.5f}{:>9.4f}{:>8}{:>8}{:>10}'.format(
        r['trial'], r['map'], r['map'] - base, r['lr'], r['leaves'], r['rounds'], r['sec']))

if rows:
    b = rows[0]
    print('\n最良: trial {}  MAP@12={:.5f}'.format(b['trial'], b['map']))
    print('  learning_rate   = {}'.format(b['lr']))
    print('  num_leaves      = {}'.format(b['leaves']))
    print('  num_boost_round = {}'.format(b['rounds']))
    print('\n※ min_data_in_leaf / feature_fraction / bagging_fraction / lambda 等は')
    print('   ログに出していないため、完走した場合のみ study から取得できる。')
