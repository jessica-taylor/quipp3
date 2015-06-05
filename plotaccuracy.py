from pylab import *
from matplotlib import pyplot as plt
import sys
import sys
import json
import subprocess

program_name = sys.argv[1]

nsamps = 1
if program_name == 'nd_clustering':
  nsamps = 100
elif program_name == 'factor_analysis':
  nsamps = 50
elif program_name == 'hmm':
  nsamps = 40
elif program_name == 'lda':
  nsamps = 40



run_results = []
for line in open('plots/' + program_name + '.data'):
  run_results.append(json.loads(line))

print 'len', len(run_results)


def concat(lst):
  res = []
  for l in lst:
    res.extend(l)
  return res

# ymax = 0
# ymin = 0
# for run in run_results:
#   origLp = run['origLp']
#   infLps = run['infLps'][1:]
#   regrets = [origLp - i for i in infLps]
#   regrets = [min(r, 1000) for r in regrets]
#   regrets = [max(r, -100) for r in regrets]
#   print 'regrets', regrets
#   ymax = max(ymax, max(regrets))
#   ymin = min(ymin, min(regrets))
#   plot(regrets)
#   # plot([0, len(infLps) - 1], [origLp, origLp])
# axis(ymin=ymin, ymax=ymax)

niters = len(run_results[0]['infLps'])

if program_name == 'factor_analysis':
  niters = min(12, niters)

def mean(xs):
  return sum(xs) / len(xs)

def quantile(xs, q):
  xs = sorted(xs)
  avg_index = (len(xs) - 1) * q
  lo = xs[int(avg_index)]
  hi = xs[1+int(avg_index)]
  portion = avg_index - int(avg_index)
  return hi * portion + lo * (1 - portion)

burn = 1

regret_distrs = [[r['origLp'] - r['infLps'][i] for r in run_results] for i in range(burn, niters)]
print 'rd', len(regret_distrs), niters

xlabel('iteration')
ylabel('regret per sample')

quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
quantiles = [0.1, 0.3, 0.5, 0.7, 0.9]
for q in quantiles:
  quants = []
  for i in range(burn, niters):
    regrets = regret_distrs[i-burn]
    quants.append(max(0, quantile(regrets, q))/nsamps)
  plot(range(burn+1, niters+1), quants, label=str(q))
  

legend()
axis(xmin=2)


plot_name = 'plots/accuracy_' + program_name + '.png'

savefig(plot_name)

subprocess.call(['eog', plot_name])

