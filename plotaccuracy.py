from pylab import *
import sys
import sys
import json
import subprocess


niters = 30

program_name = sys.argv[1]

run_results = []
for line in open('plots/' + program_name + '.data'):
  run_results.append(json.loads(line))


def concat(lst):
  res = []
  for l in lst:
    res.extend(l)
  return res

ymax = 0
ymin = 0
for run in run_results:
  origLp = run['origLp']
  infLps = run['infLps'][1:]
  regrets = [origLp - i for i in infLps]
  regrets = [min(r, 1000) for r in regrets]
  regrets = [max(r, -100) for r in regrets]
  print 'regrets', regrets
  ymax = max(ymax, max(regrets))
  ymin = min(ymin, min(regrets))
  plot(regrets)
  # plot([0, len(infLps) - 1], [origLp, origLp])

axis(ymin=ymin, ymax=ymax)

plot_name = 'plots/accuracy_' + program_name + '.png'

savefig(plot_name)

subprocess.call(['eog', plot_name])

