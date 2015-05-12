var _ = require('underscore');

var erp = require('webppl/src/erp');
var ad = require('./autodiff');
var expfam = require('./expfam');
var util = require('./util');

var mh = require('./mh')(webpplEnv);

var mbind = util.mbind, mreturn = util.mreturn, mbindMethod = util.mbindMethod, fromMonad = util.fromMonad, mcurry = util.mcurry, mcurryMethod = util.mcurryMethod;

function defaultParameters(signature) {
  var argTypes = signature[0];
  var retType = signature[1];
  var nfeatures = 0;
  argTypes.forEach(function(at) {
    nfeatures += expfam.featuresDim(at);
  });
  return {
    base: retType.defaultNatParam,
    weights: _.times(expfam.featuresDim(retType), function() {
      return _.times(nfeatures, function() { return Math.random() * 10 - 5; });
    })
  };
}

var randParameters = fromMonad('randParameters', function(signature) {
  var argTypes = signature[0];
  var retType = signature[1];
  var nfeatures = 0;
  argTypes.forEach(function(at) {
    nfeatures += expfam.featuresDim(at);
  });
  return mbind(retType.randNatParam, function(base) {
    assert(global.sample);
    return mbind(util.replicateM, expfam.featuresDim(retType),
                 mcurry(util.replicateM, nfeatures, mcurry(global.sample, erp.gaussianERP, [0, 5])),
                 function(weights) {
                   return mreturn({base: base, weights: weights});
                 });
  });
});

function getArgFeatures(argTypes, argVals) {
  return util.concat(argTypes.map(function(at, i) { return expfam.getFeatures(at, argVals[i]); }));
}

function natParamWithArguments(argTypes, retType, params, argVals) {
  return expfam.getNatParam(ad.standardNumType, retType, params, getArgFeatures(argTypes, argVals));
}

function trainParameters(signature, calls, params) {
  var argTypes = signature[0];
  var retType = signature[1];
  var samps = calls.map(function(call) {
    return [call[0], getArgFeatures(argTypes, call[1]), retType.sufStat(call[2])];
  });
  return expfam.mle(retType, samps, params);
}

function UnknownParametersModel() {
  this.signatures = [];
  this.sampler = null;
}

UnknownParametersModel.prototype.randFunction = function(s0, k0, a0) {
  var self = this;
  var args = [].slice.call(arguments, 3);
  assert(args.length >= 1);
  var argTypes = args.slice(0, args.length - 1);
  var retType = args[args.length - 1];
  var i = this.signatures.length;
  this.signatures.push([argTypes, retType]);
  var fn = function(s, k, a) {
    var args = [].slice.call(arguments, 3);
    var natParam = natParamWithArguments(self.signatures[i][0], self.signatures[i][1], s._quippParams[i], args);
    function k2(s2, result) {
      s2 = _.clone(s2);
      s2._quippCallLog = _.clone(s2._quippCallLog);
      s2._quippCallLog[i] = [[args, result], s2._quippCallLog[i]];
      return k(s2, result);
    }
    return self.signatures[i][1].sample(s, k2, a, natParam);
  };
  fn.factorScore = function(s, k, a) {
    var args = [].slice.call(arguments, 3, arguments.length - 1);
    var retVal = arguments[arguments.length - 1];
    var lp = expfam.logProbability(ad.standardNumType, self.signatures[i][1], s._quippParams[i],
                                   getArgFeatures(self.signatures[i][0], args),
                                   self.signatures[i][1].sufStat(retVal));
    // var lp0 = expfam.logProbability(ad.standardNumType, self.signatures[i][1], s._quippParams[i],
    //                                [[0]],
    //                                self.signatures[i][1].sufStat(retVal));
    // var lp1 = expfam.logProbability(ad.standardNumType, self.signatures[i][1], s._quippParams[i],
    //                                [[1]],
    //                                self.signatures[i][1].sufStat(retVal));
    s = _.clone(s);
    s._quippCallLog = _.clone(s._quippCallLog);
    s._quippCallLog[i] = [[args, retVal], s._quippCallLog[i]];
    return factor(s, k, a, lp);
  };
  return k0(s0, fn);
}

UnknownParametersModel.prototype.getSamplerWithParameters = function(params) {
  var self = this;
  assert(this.sampler);
  return function(s, k, a) {
    s = _.clone(s);
    s._quippCallLog = _.times(self.signatures.length, function() { return null; });
    s._quippParams = params;
    return self.sampler(s,
      function(s2, x) {
        return k(s2, [s2._quippCallLog.map(util.linkedListToArray), x]);
      }, a);
  };
};

UnknownParametersModel.prototype.randSample = function(s, k, a, params) {
  return this.getSamplerWithParameters(params)(s, k, a);
};

UnknownParametersModel.prototype.logPartition = fromMonad(function(params) {
  return mbind(global.ParticleFilterRejuv, this.getSamplerWithParameters(params), 50, 0, function(dist) {
    return mreturn(dist.normalizationConstant);
  });
});

UnknownParametersModel.prototype.stepParamsAndTrace = fromMonad(function(params, trace) {
  var self = this;
  var numSamps = 100;
  return mbind(mh.MH, self.getSamplerWithParameters(params), numSamps, trace, function(sampsDistrAndTrace) {
    var sampsDistr = sampsDistrAndTrace[0];
    var newTrace = sampsDistrAndTrace[1];
    var samps = sampsDistr.support();
    var combinedCallLog = _.times(self.signatures.length, function() { return []; });
    samps.forEach(function(samp) {
      var score = Math.exp(sampsDistr.score([], samp));
      var callLog = samp[0];
      for (var j = 0; j < self.signatures.length; ++j) {
        callLog[j].forEach(function(call) {
          combinedCallLog[j].push([score, call[0], call[1]]);
        });
      }
    });
    var newParams = [];
    for (var j = 0; j < self.signatures.length; ++j) {
      newParams.push(trainParameters(self.signatures[j], combinedCallLog[j], params[j]));
    }
    // for (var j = 0; j < 10; ++j) {
    //   var samp = sampsDistr.sample([]);
    //   console.log(samp[1]);
    // }
    console.log(JSON.stringify(newParams));
    return mreturn([newParams, newTrace]);
  });
});

UnknownParametersModel.prototype.inferParameters = fromMonad(function(reducer) {
  var self = this;
  var initParams = this.signatures.map(defaultParameters);
  var trainFrom = fromMonad(function(i, params, trace) {
    var rest = mbindMethod(self, 'stepParamsAndTrace', params, trace, function(paramsAndTrace) {
      return mcurry(trainFrom, i+1, paramsAndTrace[0], paramsAndTrace[1]);
    });
    return mcurry(reducer, params, trace, rest);
  });
  return mcurry(trainFrom, 0, initParams, null);
});

var unknownParametersModel = fromMonad(function(fun) {
  var model = new UnknownParametersModel();
  return mbind(fun, model.randFunction.bind(model), function(sampler) {
    model.sampler = sampler;
    return mreturn(model);
  });
});

var printReducer = fromMonad(function(params, trace, rest) {
  console.log('yay', params);
  return rest;
});

var inferParameters = fromMonad(function(fun) {
  return mbind(unknownParametersModel, fun, function(upm) {
    return mcurryMethod(model, 'inferParameters', printReducer);
  });
});

var generateRandData = fromMonad(function(fun, params) {
  var inner = fromMonad(function(rf) {
    return mbind(fun, rf, function(res) {
      return mreturn(res[0]);
    });
  });
  return mbind(unknownParametersModel, inner, function(upm) {
    return mbindMethod(upm, 'randSample', params, function(cld) {
      return mreturn(cld[1]);
    });
  });
});

var testParamInference = fromMonad(function(fun) {
  console.log('testParamInference');
  return mbind(unknownParametersModel, fun, function(upmWrong) {
    console.log('upmwrong', upmWrong.signatures);
    return mbind(util.mapM, upmWrong.signatures, randParameters, function(origParams) {
      // TODO!
      // origParams = [{"base":[20,-0.5],"weights":[[10]]}]
      console.log('orig params', JSON.stringify(origParams));
      return mbind(generateRandData, fun, origParams, function(trainingData) {
        console.log('training data', trainingData);
        return mbind(generateRandData, fun, origParams, function(testData) {

          function innerSamplerForData(data) {
            return fromMonad(function(randFunction) {
              return mbind(fun, randFunction, function(res) {
                return mreturn(mcurry(res[1], data));
              });
            });
          }

          return mbind(unknownParametersModel, innerSamplerForData(trainingData), function(upmTrain) {
            return mbind(unknownParametersModel, innerSamplerForData(trainingData), function(upmTest) {
              return mbindMethod(upmTest, 'logPartition', origParams, function(origLp) {
                console.log('orig lp', origLp);
                var reducer = fromMonad(function(infParams, trace, rest) {
                  console.log('params', JSON.stringify(infParams));
                  return mbindMethod(upmTest, 'logPartition', infParams, function(lp) {
                    console.log('inf lp', lp);
                    return rest;
                  });
                });
                return mcurryMethod(upmTrain, 'inferParameters', reducer);
              });
            });
          });
        });
      });
    });
  });
});



module.exports = {
  inferParameters: inferParameters,
  testParamInference: testParamInference
};
