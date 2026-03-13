#!/usr/bin/env python
"""Quick summary of training results"""
import json
from pathlib import Path

# Load results
with open('reports/training_results.json') as f:
    results = json.load(f)

# Summary stats
total_tickers = len(results)
successful = sum(1 for r in results.values() if r)
total_models = sum(len(v) for ticker_res in results.values() for v in ticker_res.values())

print(f'\n📊 TRAINING SUMMARY')
print(f'================')
print(f'Total Tickers:     {total_tickers}')
print(f'Successful:        {successful}/{total_tickers}')
print(f'Total Models:      {total_models}')

# ROC-AUC stats
all_aucs = []
for ticker, horizons in results.items():
    for h_key, algos in horizons.items():
        for algo, metrics in algos.items():
            if 'roc_auc' in metrics:
                all_aucs.append(metrics['roc_auc'])

if all_aucs:
    avg = sum(all_aucs) / len(all_aucs)
    print(f'\n📈 ROC-AUC Statistics')
    print(f'====================')
    print(f'Average ROC-AUC:   {avg:.4f}')
    print(f'Min:               {min(all_aucs):.4f}')
    print(f'Max:               {max(all_aucs):.4f}')
    print(f'Median:            {sorted(all_aucs)[len(all_aucs)//2]:.4f}')
    std_dev = (sum((x-avg)**2 for x in all_aucs)/len(all_aucs))**0.5
    print(f'Std Dev:           {std_dev:.4f}')

    # Count by AUC bins
    bins = {'0.45-0.50': 0, '0.50-0.55': 0, '0.55-0.60': 0, '0.60+': 0}
    for auc in all_aucs:
        if auc < 0.50:
            bins['0.45-0.50'] += 1
        elif auc < 0.55:
            bins['0.50-0.55'] += 1
        elif auc < 0.60:
            bins['0.55-0.60'] += 1
        else:
            bins['0.60+'] += 1
    
    print(f'\nROC-AUC Distribution:')
    for bin_name, count in bins.items():
        pct = 100 * count / len(all_aucs)
        print(f'  {bin_name}: {count:3d} ({pct:5.1f}%)')

# Top 10 models
print(f'\n🏆 Top 10 Models (by ROC-AUC)')
print(f'=============================')
top_models = []
for ticker, horizons in results.items():
    for h_key, algos in horizons.items():
        for algo, metrics in algos.items():
            if 'roc_auc' in metrics:
                top_models.append((ticker, h_key, algo, metrics['roc_auc'], metrics.get('accuracy', 0)))

top_models.sort(key=lambda x: x[3], reverse=True)
for i, (t, h, a, auc, acc) in enumerate(top_models[:10], 1):
    print(f'{i:2d}. {t:8} {h:3} {a:6} → ROC-AUC: {auc:.4f} (Acc: {acc:.3f})')

print(f'\n✓ Models saved to: {Path("models").resolve()}')
print(f'✓ Report saved to: {Path("reports/training_results.json").resolve()}')
