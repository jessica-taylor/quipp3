
var erp = require('webppl/src/erp');
var webpplUtil = require('webppl/src/util');


var optimization = require('./optimization');
var ad = require('./autodiff'); 
var util = require('./util');


// ExpFam interface
//
// name: string
// sufStat(value: *): [double]
// g(natParam: [double]): double
// sample(natParam: [double]): *
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
    return t.sub(t.num(0),
                 t.add(t.div(t.mul(nps[0], nps[0]), nps[1]),
                       t.mul(t.num(0.5), t.log(t.mul(t.num(-2), nps[1])))));
  },
  sample: function(s, k, a, nps) {
    var variance = -1 / (2 * nps[1]);
    var mean = nps[1] * variance;
    return global.sample(s, k, a, erp.gaussianERP, [mean, Math.sqrt(variance)]);
  },
  defaultNatParam: [0.0, -0.001],
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
    g: function(nps) {
      return webpplUtil.logsumexp([0].concat(nps));
    },
    sample: function(s, k, a, nps) {
      var lse = webpplUtil.logsumexp([0].concat(nps));
      var probs = nps.map(function(np) { return Math.exp(np - lse); });
      return global.sample(s, k, a, erp.discreteERP, [probs]);
    },
    defaultNatParam: _.times(n-1, function() { return 0.0; }),
    featuresMask: _.times(n-1, function() { return true; })
  };
}

function Tuple(types) {
  return {
    name: 'Tuple(' + _.pluck(types, 'name').join(', ') + ')',
    sufStat: function(value) {
      return util.concat(types.map(function(type, i) {
        return type.sufStat(value[i]);
      }));
    },
    g: function(nps) {
      return util.sum(types.map(function(type, i) { return type.g(nps[i]); }));
    },
    sample: function(s, k, a, nps) {
      util.wpplMap(s, k, a, function(s2, k2, a2, type, i) {
        type.sample(s2, k2, a2, nps[i]);
      });
    },
    defaultNatParam: concat(_.pluck(types, 'defaultNatParam')),
    featuresMask: concat(_.pluck(types, 'featuresMask'))
  };
}

// A sample is of the form [weight, features, sufStat].

function groupSamplesByFeatures(samps) {
  var totWeightAndSufStats = {};
  samps.forEach(function(samp) {
    var features = JSON.stringify(samp[1]);
    if (!totWeightAndSufStats[features]) totWeightAndSufStats[features] = [];
    var tot = totWeightAndSufStats[features];
    tot[0] += samp[0];
    tot[1] = vecAdd(tot[1], vecScale(samp[0], samp[2]));
  });
  var samps2 = [];
  for (var stringFeatures in totWeightAndSufStats) {
    var tot = totWeightAndSufStats[stringFeatures];
    samps2.push([tot[0], JSON.parse(stringFeatures), vecScale(1/tot[0], tot[2])]);
  }
  return samps2;
}

function paramsToVector(params) {
  return params.base.concat(optimization.matElements(params.weights));
}

function vectorToParams(ef, vec) {
  return {
    base: vec.splice(0, dim(ef)),
    weights: optimization.elementsToMat(featuresDim(ef), vec.splice(dim(ef)))
  };
}

function getNatParam(t, ef, params, argFeatures) {
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

function mle(ef, samples, params) {
  var score = paramsScoreFunction(ef, samples);
  return vectorToParams(ef, optimization.gradientDescent(
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
  getNatParam: getNatParam,
  mle: mle
};

