from pylab import *
import math
from matplotlib.patches import Ellipse

dat = eval(open('irisdata.txt').read())
(xs, ys) = zip(*dat)

# def iparams(p):
#     (_, ((n1, n2), ((n3,),))) = p
#     variance = -1 / (2 * n2)
#     mean0 = n1 * variance
#     mean1 = (n1 + n3) * variance
#     return (variance, mean0, mean1)

def boolcolor(x):
    return 'red' if x else 'blue'

params1 = [[{"stdev":0.5372731549542051,"baseMean":6.317781274586794,"coeffs":[-1.3016404423965577]},{"stdev":0.6880720208986841,"baseMean":4.984290584841217,"coeffs":[-3.3624798852527316]}]]

params2 = [[{"stdev":0.6595119407561928,"baseMean":6.262000000000001,"coeffs":[]},{"stdev":0.8214401986754728,"baseMean":4.905999999999999,"coeffs":[]}],[{"stdev":0.3489469873777391,"baseMean":5.005999999999999,"coeffs":[]},{"stdev":0.17176728442867112,"baseMean":1.4640000000000004,"coeffs":[]}]]

params3 = [{"stdev":0.6595119407561928,"baseMean":6.262000000000001,"coeffs":[]},{"stdev":0.3489469873777391,"baseMean":5.005999999999999,"coeffs":[]},{"stdev":0.460023458046562,"baseMean":-1.5557132215677991,"coeffs":[1.0318928811190988]},{"stdev":0.16567935943532425,"baseMean":0.8137676160441453,"coeffs":[0.12989060806149724]}]

params3 = [{"stdev":0.7287275058887802,"baseMean":5.368444981174984,"coeffs":[]},{"stdev":0.6619425046955395,"baseMean":6.2631664887270455,"coeffs":[]},{"stdev":0.702275237872738,"baseMean":-6.875651795265539,"coeffs":[1.7159693631199096]},{"stdev":0.41252571793466597,"baseMean":-1.4608529214942643,"coeffs":[1.0341202813737411]}]

params3 =  [{"stdev":0.6332047780028703,"baseMean":5.269253362188897,"coeffs":[]},{"stdev":0.6739160276200951,"baseMean":6.278654319540614,"coeffs":[]},{"stdev":0.665264843960244,"baseMean":-6.503556209183356,"coeffs":[1.633997767969041]},{"stdev":0.4217729803083605,"baseMean":-1.3905625955310108,"coeffs":[1.019667224931946]}]

params3 =  [{"stdev":0.502494966603192,"baseMean":5.154049074616882,"coeffs":[]},{"stdev":0.6653231807208158,"baseMean":6.299102940362089,"coeffs":[]},{"stdev":0.6490476937643579,"baseMean":-5.567432995928555,"coeffs":[1.4442755632815718]},{"stdev":0.4362833364201138,"baseMean":-1.1943998457641816,"coeffs":[0.9838923552237534]}]

params3 = [{"stdev":0.3561792194462958,"baseMean":5.018956419777222,"coeffs":[]},{"stdev":0.6588212013216088,"baseMean":6.270160882323193,"coeffs":[]},{"stdev":0.3933041696762019,"baseMean":-0.3595969363212091,"coeffs":[0.37489516093589376]},{"stdev":0.4618204234566134,"baseMean":-1.5356667335697272,"coeffs":[1.0290632960216497]}]

params3 = [{"stdev":0.3489469873777391,"baseMean":5.005999999999999,"coeffs":[]},{"stdev":0.6595119407561928,"baseMean":6.262000000000001,"coeffs":[]},{"stdev":0.16567935943532425,"baseMean":0.8137676160441453,"coeffs":[0.12989060806149724]},{"stdev":0.460023458046562,"baseMean":-1.5557132215677991,"coeffs":[1.0318928811190988]}]

params3 = [{"stdev":0.3489469873777391,"baseMean":5.005999999999999,"coeffs":[]},{"stdev":0.6595119407561928,"baseMean":6.262000000000001,"coeffs":[]},{"stdev":0.16567935943532425,"baseMean":0.8137676160441453,"coeffs":[0.12989060806149724]},{"stdev":0.460023458046562,"baseMean":-1.5557132215677991,"coeffs":[1.0318928811190988]}]

params3 = [{"stdev":0.3489469873777391,"baseMean":5.005999999999999,"coeffs":[]},{"stdev":0.6595119407561928,"baseMean":6.262000000000001,"coeffs":[]},{"stdev":0.16567935943532425,"baseMean":0.8137676160441453,"coeffs":[0.12989060806149724]},{"stdev":0.460023458046562,"baseMean":-1.5557132215677991,"coeffs":[1.0318928811190988]}]

score = -269.1233394103485

def iparams1(p):
  stdev = p['stdev']
  x0 = p['baseMean']
  x1 = x0 + p['coeffs'][0]
  return stdev, x0, x1

def iparams2(p):
  stdev = p['stdev']
  mean = p['baseMean']
  return stdev, mean

def norm(v):
  return math.sqrt(sum([x**2 for x in v]))

def iparams3(px, py):
  xstdev = px['stdev']
  xmean = px['baseMean']

  yresid = py['stdev']
  ybase = py['baseMean']
  ycoeff = py['coeffs'][0]
  ymean = ybase + ycoeff * xmean

  angle = math.atan2(ycoeff, 1)
  covar = xstdev**2 * ycoeff

  ystdev = math.sqrt(yresid**2 + (ycoeff * xstdev)**2)
  print 'xstdev', xstdev, ystdev

  uvec = [1, ycoeff]
  uvec = [x/norm(uvec) for x in uvec]
  print 'uvec', uvec
  ustdev = math.sqrt(uvec[0]**2 * xstdev**2 + uvec[1] * ystdev**2 + 2 * uvec[0] * uvec[1] * covar)

  vstdev = math.sqrt(uvec[1]**2 * xstdev**2 + uvec[0] * ystdev**2 - 2 * uvec[0] * uvec[1] * covar)

  return xmean, ymean, angle, ustdev, vstdev






which_plot = 1

if which_plot == 0:

  xstdev0, xmean0, xmean1 = iparams1(params1[0][0])
  xstdev1 = xstdev0
  ystdev0, ymean0, ymean1 = iparams1(params1[0][1])
  ystdev1 = ystdev0

elif which_plot == 1:

  xstdev0, xmean0 = iparams2(params2[0][0])
  ystdev0, ymean0 = iparams2(params2[0][1])
  xstdev1, xmean1 = iparams2(params2[1][0])
  ystdev1, ymean1 = iparams2(params2[1][1])

else:

  xmean0, ymean0, angle0, ustdev0, vstdev0 = iparams3(params3[0], params3[2])
  xmean1, ymean1, angle1, ustdev1, vstdev1 = iparams3(params3[1], params3[3])

fig = figure()
ax = fig.add_subplot(111)
ax.scatter(xs, ys)

ellipses = []
if which_plot != 2:
  for x,y,xs,ys,c in [(xmean0, ymean0,xstdev0, ystdev0, 'blue'), (xmean1, ymean1, xstdev1, ystdev1, 'red')]:
    ellipses.append((Ellipse(xy=(x,y), width=xs*2, height=ys*2), c))
else:
  for x,y,a,us,vs,c in [(xmean0, ymean0, angle0, ustdev0, vstdev0, 'red'), (xmean1, ymean1, angle1, ustdev1, vstdev1, 'blue')]:
    print 'xy', x, y, a, us, vs, c
    ellipses.append((Ellipse(xy=(x,y), width=us*2, height=vs*2, angle=a*180/math.pi), c))

# ax.scatter(xs, ys, c=list(map(boolcolor, clusters)))

ylabel('Petal length (in)')
xlabel('Sepal length (in)')

for e,c in ellipses:
  ax.add_artist(e)
  e.set_clip_box(ax.bbox)
  e.set_alpha(0.1)
  e.set_facecolor(c)
savefig('irisclusters.png')
