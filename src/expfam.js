
var erp = require('webppl/src/erp');
var webpplUtil = require('webppl/src/util');


var opt = require('./optimization');
var ad = require('./autodiff'); 
var util = require('./util');

var mbind = util.mbind, mreturn = util.mreturn, mcurry = util.mcurry, fromMonad = util.fromMonad;


// ExpFam interface
//
// name: string
// sufStat(value: *): [double]
// g(natParam: [double]): double
// sample(natParam: [double]): *
// randNatParam(): [double]
// defaultNatParam: [double]
// featuresMask: [bool]


function dim(ef) {
  return ef.featuresMask.length;
}

function featuresDim(ef) {
  var d = 0;
  ef.featuresMask.forEach(function(f) {
    if (f) ++d;
  });
  return d;
}

function getFeatures(ef, value) {
  var ss = ef.sufStat(value);
  var feats = [];
  for (var i = 0; i < ss.length; ++i) {
    if (ef.featuresMask[i]) feats.push(ss[i]);
  }
  return feats;
}

function featuresToSufStat(t, ef, features) {
  var i = 0;
  return ef.featuresMask.map(function(mask) {
    return mask ? features[i++] : t.num(0);
  });
}

var Double = {
  name: 'Double',
  sufStat: function(value) { return [value, value*value]; },
  g: function(t, nps) {
    // return -Math.pow(nps[0], 2) / nps[1] - 0.5 * Math.log(-2 * nps[1]);
    return t.sub(t.num(0.5 * Math.log(2 * Math.PI)),
                 t.add(t.div(t.mul(nps[0], nps[0]), t.mul(t.num(4), nps[1])),
                       t.mul(t.num(0.5), t.log(t.mul(t.num(-2), nps[1])))));
  },
  sample: function(s, k, a, nps) {
    var variance = -1 / (2 * nps[1]);
    var mean = nps[0] * variance;
    return global.sample(s, k, a, erp.gaussianERP, [mean, Math.sqrt(variance)]);
  },
  defaultNatParam: [0.0, -0.001],
  randNatParam: fromMonad(function() {
    return mbind(global.sample, erp.gaussianERP, [0, 5], function(mean) {
      return mbind(global.sample, erp.gaussianERP, [0, 5], function(stdev) {
        var variance = stdev*stdev;
        return mreturn([mean / variance, -1 / (2 * variance)]);
      });
    });
  }),
  featuresMask: [true, false]
};

function Categorical(n) {
  return {
    name: 'Categorical(' + n + ')',
    sufStat: function(value) {
      return _.range(1, n).map(function(i) {
        return (value == i ? 1.0 : 0.0);
      });
    },
    g: function(t, nps) {
      return ad.numLogSumExp(t, [t.num(0)].concat(nps));
    },
    sample: function(s, k, a, nps) {
      var lse = webpplUtil.logsumexp([0].concat(nps));
      var probs = nps.map(function(np) { return Math.exp(np - lse); });
      return global.sample(s, k, a, erp.discreteERP, [probs]);
    },
    defaultNatParam: _.times(n-1, function() { return 0.0; }),
    randNatParam: fromMonad(function() {
      return mcurry(util.replicateM, n-1, mcurry(global.sample, erp.gaussianERP, [0, 5]));
    }),
    featuresMask: _.times(n-1, function() { return true; })
  };
}

function Tuple(types) {
  var cumulativeDim = [0];
  types.forEach(function(t) {
    cumulativeDim.push(cumulativeDim[cumulativeDim.length - 1] + dim(t));
  });
  function sliceNp(i, nps) {
    return nps.slice(cumulativeDim[i], cumulativeDim[i+1]);
  }
  return {
    name: 'Tuple(' + _.pluck(types, 'name').join(', ') + ')',
    sufStat: function(value) {
      return util.concat(types.map(function(type, i) {
        return type.sufStat(value[i]);
      }));
    },
    g: function(t, nps) {
      return ad.numSum(t, types.map(function(type, i) {
        return type.g(t, sliceNp(i, nps));
      }));
    },
    sample: fromMonad(function(nps) {
      return mcurry(util.mapM, types, fromMonad(function(type, i) {
        var typeNp = sliceNp(i, nps);
        return mcurry(type.sample, typeNp);
      }));
    }),
    defaultNatParam: util.concat(_.pluck(types, 'defaultNatParam')),
    randNatParam: mbind(util.mapM, _.pluck(types, 'randNatParam'), fromMonad(function(x) { return x; }), function(res) {
      return mreturn(util.concat(res));
    }),
    featuresMask: util.concat(_.pluck(types, 'featuresMask'))
  };
}

// A sample is of the form [weight, features, sufStat].

function groupSamplesByFeatures(samps) {
  var totWeightAndSufStats = {};
  samps.forEach(function(samp) {
    var features = JSON.stringify(samp[1]);
    if (!totWeightAndSufStats[features]) totWeightAndSufStats[features] = [0, opt.zeros(samp[2].length)];
    var tot = totWeightAndSufStats[features];
    tot[0] += samp[0];
    tot[1] = opt.vecAdd(tot[1], opt.vecScale(samp[0], samp[2]));
  });
  var samps2 = [];
  for (var stringFeatures in totWeightAndSufStats) {
    var tot = totWeightAndSufStats[stringFeatures];
    samps2.push([tot[0], JSON.parse(stringFeatures), opt.vecScale(1/tot[0], tot[1])]);
  }
  return samps2;
}

function paramsToVector(params) {
  return params.base.concat(opt.matElements(params.weights));
}

function vectorToParams(ef, vec) {
  return {
    base: vec.slice(0, dim(ef)),
    weights: opt.elementsToMat(featuresDim(ef), vec.slice(dim(ef)))
  };
}

function getNatParam(t, ef, params, argFeatures) {
  if (params.weights.length > 0) assert.equal(params.weights[0].length, argFeatures.length);
  var weightPart = ad.numMatMulByVector(t, params.weights, argFeatures.map(t.num));
  return ad.numVecAdd(t, params.base, featuresToSufStat(t, ef, weightPart));
}

function logProbability(t, ef, params, argFeatures, ss) {
  var np = getNatParam(t, ef, params, argFeatures);
  return t.sub(ad.numDotProduct(t, np, ss.map(t.num)), ef.g(t, np));
}

function paramsScoreFunction(ef, samples) {
  return function(t, eta) {
    var params = vectorToParams(ef, eta);
    var score = t.num(0.0);
    // TODO: normalize?
    samples.forEach(function(samp) {
      score = t.add(score, t.mul(t.num(samp[0]), logProbability(t, ef, params, samp[1], samp[2])));
    });
    return score;
  };
}

function gaussianMle(samples) {
  // let nxs = let (_, x, _) = head samples in length x
  //     xss :: Matrix Double = [1:x | (_, x, _) <- samples]
  //     ys :: [Double] = [y | (_, _, [y, _]) <- samples]
  //     beta :: [Double] = matInv (matMul (transpose xss) xss) `matMulByVector` matMulByVector (transpose xss) ys
  //     predYs :: [Double] = map (dotProduct beta) xss
  //     resid :: Double = mean [(y - p)^2 | (y, p) <- zip ys predYs]
  //     resid' = max 0.0001 resid
  // in repeat ([head beta / resid', -1 / (2 * resid')], [map (/ resid') (tail beta)])
  var nxs = samples[0][1].length;
  var weightsDiag = samples.map(function(s) { return s[0]; });
  var xss = samples.map(function(samp) {
    return [1].concat(samp[1]);
  });
  var ys = samples.map(function(samp) {
    return samp[2][0];
  });
  // TODO: avoid creating weights matrix
  var A = opt.matMul(opt.transpose(xss), opt.elemMatProduct(weightsDiag, xss));
  var b = opt.matMulByVector(opt.transpose(xss), opt.elemProduct(weightsDiag, ys));
  var beta = opt.linSolve(A, b);
  // var beta = opt.linSolve(
  //     opt.matMul(opt.transpose(xss), opt.matMul(weights, xss)),
  //     opt.matMulByVector(opt.transpose(xss), opt.matMulByVector(weights, ys)));
  var predYs = xss.map(function(row) { return opt.dotProduct(beta, row); });
  var totResid = 0.0;
  var totWeight = 0.0;
  ys.forEach(function(y, i) {
    totWeight += samples[i][0];
    totResid += samples[i][0] * Math.pow(y - predYs[i], 2);
  });
  var resid2 = Math.max(totResid / totWeight, 0.0001);
  var params = {
    base: [beta[0] / resid2, -1 / (2 * resid2)],
    weights: [beta.slice(1).map(function(b) { return b / resid2; })]
  };
  ys.forEach(function(y, i) {
    var lp = logProbability(ad.standardNumType, Double, params, samples[i][1], samples[i][2]);
    var predLp = - 0.5 * Math.log(2 * Math.PI * resid2) - Math.pow(y - predYs[i], 2) / (2 * resid2); 
    assert(Math.abs(lp - predLp) < 0.01);
  });
  return params;
}

function mle(ef, samples, params) {
  if (ef.name == 'Double' || ef.name == 'Tuple(Double)') {
    return gaussianMle(samples);
  }
  samples = groupSamplesByFeatures(samples);
  var score = paramsScoreFunction(ef, samples);
  return vectorToParams(ef, opt.gradientDescent(
      function(eta) {
        return score(ad.standardNumType, eta);
      },
      function(eta) {
        return ad.gradient(ad.standardNumType, score, eta).grad;
      },
      // function(eta) {
      //   var hessDual = ad.hessian(ad.standardNumType, score, eta);
      //   var hess = [];
      //   for (var i = 0; i < eta.length; ++i) {
      //     hess.push(hessDual[i].grad);
      //   }
      //   return hess;
      // },
      paramsToVector(params)));
}

module.exports = {
  dim: dim,
  featuresDim: featuresDim,
  getFeatures: getFeatures,
  Double: Double,
  Categorical: Categorical,
  Tuple: Tuple,
  logProbability: logProbability,
  getNatParam: getNatParam,
  mle: mle
};

