import sys
import subprocess
import sys
import json


nparallel = 4

program_name = sys.argv[1]

niters = 30
if program_name == 'factor_analysis':
  niters = 12
if program_name == 'hmm':
  niters = 10
if program_name == 'lda':
  niters = 10



def append_result(line):
  with open('plots/' + program_name + '.data', 'a+') as f:
    f.write(line + '\n')

# for run in range(nruns):
#   output = subprocess.check_output(['./runquipp', 'examples/' + sys.argv[1] + '.wppl', '--iters', str(niters)])
#   last_line = output.split('\n')[-2]
#   print last_line
#   run_results.append(json.loads(last_line))

while True:
  popens = []
  for run in range(nparallel):
    p = subprocess.Popen(['./runquipp', 'examples/' + program_name + '.wppl', '--iters', str(niters)],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    popens.append(p)

  for p in popens:
    p.wait()
    output = p.stdout.read()
    error = p.stderr.read()
    last_line = output.split('\n')[-2]
    if 'results!' in last_line: 
      append_result(last_line)
    else:
      print '-----------------'
      print output
      print '*****************'
      print error
      print '+++++++++++++++++'
    print last_line
