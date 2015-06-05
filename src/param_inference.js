var _ = require('underscore');

var erp = require('webppl/src/erp');
var ad = require('./autodiff');
var expfam = require('./expfam');
var util = require('./util');

var mh = require('./mh')(webpplEnv);
var pfais = require('./pfais')(webpplEnv);

var mbind = util.mbind, mreturn = util.mreturn, mbindMethod = util.mbindMethod, fromMonad = util.fromMonad, mcurry = util.mcurry, mcurryMethod = util.mcurryMethod;

var argv = require('yargs').argv;
console.log('argv', argv);

function DataSampler() {
  this.index = 0;
  this.valueMap = {};
}

var nextObsIndex = function(s, k, a) {
  var res = s._obsIndex++;
  return k(s, res);
};

var resetObsIndex = function(s, k, a) {
  s = _.clone(s);
  s._obsIndex = 0;
  return k(s);
};

DataSampler.prototype.sampleOrCondition = fromMonad(function(addr, rf) {
  var self = this;
  var args = [].slice.call(arguments, 2, arguments.length);
  if (typeof addr == 'function' || typeof addr == 'object') {
    // return mbind(util.getAddress, function(a) {
    return mbind(nextObsIndex, function(a) {
      return mcurryMethod.apply(null, [self, 'sampleOrCondition', a, addr, rf].concat(args));
    });
  }

  if (rf.constructor == erp.ERP) {
    return mbind(global.sample, rf, args, function(samp) {
      self.valueMap[addr] = samp;
      return mreturn(samp);
    });
  }

  return mbind.apply(null, [rf].concat(args).concat([function(samp) {
    self.valueMap[addr] = samp;
    return mreturn(samp);
  }]));
});

var sampleData = fromMonad(function(s) {
  return mbind(resetObsIndex, function() {
    var sampler = new DataSampler();
    global.observe = sampler.sampleOrCondition.bind(sampler);
    return mbind(s, function(res) {
      return mreturn(sampler.valueMap);
    });
  });
});

function DataConditioner(valueMap) {
  this.index = 0;
  this.valueMap = valueMap;
}

DataConditioner.prototype.sampleOrCondition = fromMonad(function(addr, rf) {
  var self = this;
  var args = [].slice.call(arguments, 2, arguments.length);
  if (typeof addr == 'function' || typeof addr == 'object') {
    return mbind(nextObsIndex, function(a) {
    // return mbind(util.getAddress, function(a) {
      return mcurryMethod.apply(null, [self, 'sampleOrCondition', a, addr, rf].concat(args));
    });
  }

  assert(addr in self.valueMap);
  var ret = self.valueMap[addr];

  if (rf.constructor == erp.ERP) {
    return mbind(global.factor, rf.score(args, ret), function() {
      return mreturn(ret);
    });
  }

  return mcurry.apply(null, [rf.factorScore].concat(args).concat([ret]));
});

var conditionData = fromMonad(function(s, valueMap) {
  return mbind(resetObsIndex, function() {
    var conditioner = new DataConditioner(valueMap);
    global.observe = conditioner.sampleOrCondition.bind(conditioner);
    return mbind(s, function(res) {
      return mreturn(null);
    });
  });
});

var randParameters = fromMonad('randParameters', function(signature) {
  var argTypes = signature[0];
  var retType = signature[1];
  var nfeatures = 0;
  argTypes.forEach(function(at) {
    nfeatures += expfam.featuresDim(at);
  });
  if (retType.randParams) return mcurry(retType.randParams, nfeatures);
  return mbind(retType.randNatParam, function(base) {
    assert(global.sample);
    return mbind(util.replicateM, expfam.featuresDim(retType),
                 mcurry(util.replicateM, nfeatures, mcurry(global.sample, erp.gaussianERP, [0, 2])),
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
  // console.log('calls', JSON.stringify(calls));
  var argTypes = signature[0];
  var retType = signature[1];
  var samps = calls.map(function(call) {
    return [call[0], getArgFeatures(argTypes, call[1]), retType.sufStat(call[2])];
  });
  return expfam.mle(retType, samps, params);
}

function UnknownParametersModel(opts) {
  opts = opts || {};
  this.signatures = [];
  this.sampler = null;
  this.logPartitionParticles = opts.logPartitionParticles || 30;
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
  var self = this;
  var nparticles = self.logPartitionParticles;
  return mbind(global.ParticleFilterRejuv, self.getSamplerWithParameters(params), nparticles, 0, function(dist) {
  return mbind(global.ParticleFilterRejuv, self.getSamplerWithParameters(params), nparticles, 0, function(dist2) {
    // return mbind(pfais.ParticleFilter, self.getSamplerWithParameters(params), nparticles, 5, function(dist) {
      // console.log(dist.normalizationConstant, dist2.normalizationConstant);
      console.log(dist.normalizationConstant, dist2.normalizationConstant);
      return mreturn(dist.normalizationConstant);
    });
  });
});

UnknownParametersModel.prototype.formatParameters = function(paramsList) {
  var self = this;
  return paramsList.map(function(params, i) {
    return self.signatures[i][1].formatParams(params);
  });
};

UnknownParametersModel.prototype.stepParamsAndTrace = fromMonad(function(params, trace) {
  var self = this;
  // TODO: should this depend on something?
  // var numSamps = 2000;
  var numSamps = 200;
  return mbind(mh.MH, self.getSamplerWithParameters(params), numSamps, trace, function(sampsDistrAndTrace) {
    console.log('got samples');
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
    for (var j = 0; j < 1; ++j) {
      var samp = sampsDistr.sample([]);
      // console.log(JSON.stringify(samp[1]));
    }
    // console.log(JSON.stringify(newParams));
    return mreturn([newParams, newTrace]);
  });
});

UnknownParametersModel.prototype.inferParameters = fromMonad(function(reducer) {
  var self = this;

  return mbind(util.mapM, self.signatures, randParameters, function(initParams) {
    var trainFrom = fromMonad(function(i, params, trace) {
      var rest = mbindMethod(self, 'stepParamsAndTrace', params, trace, function(paramsAndTrace) {
        return mcurry(trainFrom, i+1, paramsAndTrace[0], paramsAndTrace[1]);
      });
      return mcurry(reducer, params, trace, rest);
    });
    return mcurry(trainFrom, 0, initParams, null);
  });
});

var unknownParametersModel = fromMonad(function(fun, opts) {
  var model = new UnknownParametersModel(opts);
  return mbind(fun, model.randFunction.bind(model), function(sampler) {
    model.sampler = sampler;
    return mreturn(model);
  });
});

var printReducer = fromMonad(function(params, trace, rest) {
  console.log('yay', params);
  return rest;
});

var inferParameters = fromMonad(function(fun, opts) {
  console.log('opts', opts);
  return mbind(unknownParametersModel, fun, opts, function(upm) {
    var reducer = fromMonad(function(params, trace, rest) {
      console.log('yay', JSON.stringify(upm.formatParameters(params)));
      console.log('also', JSON.stringify(params));
      return mbindMethod(upm, 'logPartition', params, function(lp) {
        console.log('score', lp);
        return rest;
      });
    });
    return mcurryMethod(upm, 'inferParameters', reducer);
  });
});

var generateRandData = fromMonad(function(fun, opts, params) {
  var inner = fromMonad(function(rf) {
    console.log('in inner');
    return mbind(fun, rf, function(res) {
      return mreturn(res[0]);
    });
  });
  return mbind(unknownParametersModel, inner, opts, function(upm) {
    return mbindMethod(upm, 'randSample', params, function(cld) {
      return mreturn(cld[1]);
    });
  });
});

var testParamInferenceSplit = fromMonad(function(fun, opts) {
  console.log('testParamInference');
  return mbind(unknownParametersModel, fun, opts, function(upmWrong) {
    console.log('upmwrong', upmWrong.signatures);
    return mbind(util.mapM, upmWrong.signatures, randParameters, function(origParams) {
      // origParams = [{"base":[-0.2525198567891542,-0.03591206849102666,-18.248036847588864,-2.1923064679508037,-0.11286868833712595,-0.008005218772590184],"weights":[[-4.024192574829005],[7.231811766517399],[2.829228707582246]]}]
      // TODO!
      console.log('orig params', JSON.stringify(upmWrong.formatParameters(origParams)));
      console.log('also', JSON.stringify(origParams));
      return mbind(generateRandData, fun, opts, origParams, function(trainingData) {
        // console.log('training data', trainingData);
        return mbind(generateRandData, fun, opts, origParams, function(testData) {
          // console.log('test data', testData);

          function innerSamplerForData(data) {
            return fromMonad(function(randFunction) {
              return mbind(fun, randFunction, function(res) {
                return mreturn(mcurry(res[1], data));
              });
            });
          }

          return mbind(unknownParametersModel, innerSamplerForData(trainingData), opts, function(upmTrain) {
            console.log('upmTrain');
            return mbind(unknownParametersModel, innerSamplerForData(testData), opts, function(upmTest) {
              console.log('upmTest');
              return mbindMethod(upmTest, 'logPartition', origParams, function(origLp) {
                console.log('@', origLp);
                var lps = [];
                var reducer = fromMonad(function(infParams, trace, rest) {
                  console.log('params', JSON.stringify(upmWrong.formatParameters(infParams)));
                  return mbindMethod(upmTest, 'logPartition', infParams, function(lp) {
                    console.log('#', lp);
                    lps.push(lp);
                    if (argv.iters !== undefined && lps.length == argv.iters) {
                      console.log(JSON.stringify({tag: 'results!', origLp: origLp, infLps: lps}));
                      process.exit();
                    }
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

var testParamInference = fromMonad(function(fun, opts) {
  var fun2 = fromMonad(function(randFunction) {
    return mbind(fun, randFunction, function(innerSampler) {
      var gendata = mcurry(sampleData, innerSampler);
      var condata = fromMonad(function(data) {
        return mcurry(conditionData, innerSampler, data);
      });
      return mreturn([gendata, condata]);
    });
  });
  return mcurry(testParamInferenceSplit, fun2, opts);
});


module.exports = {
  inferParameters: inferParameters,
  testParamInference: testParamInference,
  testParamInferenceSplit: testParamInferenceSplit
};
