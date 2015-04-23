var _ = require('underscore');

var ad = require('./autodiff');
var expfam = require('./expfam');
var util = require('./util');

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
      return _.times(nfeatures, function() { return 0; });
    })
  };
}

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
    lp *= 1000;
    var lp0 = expfam.logProbability(ad.standardNumType, self.signatures[i][1], s._quippParams[i],
                                   [0],
                                   self.signatures[i][1].sufStat(retVal));
    var lp1 = expfam.logProbability(ad.standardNumType, self.signatures[i][1], s._quippParams[i],
                                   [1],
                                   self.signatures[i][1].sufStat(retVal));
    // console.log(lp0 - lp1);
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

UnknownParametersModel.prototype.inferParameters = function(s, k, a) {
  var self = this;
  var numIters = 1000;
  var numSamps = 200;
  var burnIn = Math.floor(numSamps/4);
  var skip = 2;
  
  var initParams = this.signatures.map(defaultParameters);
  return trainFrom(s, k, a, 0, initParams);
  function trainFrom(st, kt, at, i, params) {
    if (i == numIters) {
      return kt(st, params);
    }
    // TODO start from a trace?
    return MH(st, mhK, at, self.getSamplerWithParameters(params), numSamps);
    function mhK(sm, sampsDistr) {
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
      for (var j = 0; j < 10; ++j) {
        var samp = sampsDistr.sample([]);
        console.log(samp[1]);
      }
      console.log(JSON.stringify(newParams));
      return trainFrom(sm, kt, at, i+1, newParams);
    }
  }
};

function unknownParametersModel(s, k, a, fun) {
  var model = new UnknownParametersModel();
  function k2(s2, sampler) {
    model.sampler = sampler;
    return k(s2, model);
  };
  return fun(s, k2, a, model.randFunction.bind(model));
}

function inferParameters(s, k, a, fun) {
  function k2(s2, model) {
    return model.inferParameters(s2, k, a);
  }
  return unknownParametersModel(s, k2, a, fun);
}

module.exports = {
  inferParameters: inferParameters
};
