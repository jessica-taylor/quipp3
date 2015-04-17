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
    return [1, getArgFeatures(argTypes, call[0]), retType.sufStat(call[1])];
  });
  return expfam.mle(retType, samps, params);
}

function UnknownParametersModel() {
  this.signatures = [];
  this.sampler = null;
}

UnknownParametersModel.prototype.randFunction = function(s0, k0, a0) {
  var self = this;
  var args = [].splice.call(arguments, 3);
  assert(args.length >= 1);
  var argTypes = args.splice(0, args.length - 1);
  var retType = args[args.length - 1];
  var i = this.signatures.length;
  this.signatures.push([argTypes, retType]);
  return k0(s0, function(s, k, a) {
    var args = [].splice.call(arguments, 3);
    var natParam = natParamWithArguments(self.signatures[i][0], self.signatures[i][1], s._quippParams[i], args);
    function k2(s2, result) {
      s._quippCallLog[i].push([args, result]);
      return k(s2, result);
    }
    return self.signatures[i][1].sample(s, k2, a, natParam);
  });
}

UnknownParametersModel.prototype.getSamplerWithParameters = function(params) {
  var self = this;
  assert(this.sampler);
  return function(s, k, a) {
    s._quippCallLog = _.times(self.signatures.length, function() { return []; });
    s._quippParams = params;
    return self.sampler(s,
      function(s2, x) {
        return k(s2, [s2._quippCallLog, x]);
      }, a);
  };
};

UnknownParametersModel.prototype.inferParameters = function(s, k, a) {
  var self = this;
  var numIters = 10;
  var numSamps = 10;
  var burnIn = Math.floor(numSamps/4);
  var skip = 2;
  
  var initParams = this.signatures.map(defaultParameters);
  console.log('initParams', initParams);
  return trainFrom(s, k, a, 0, initParams);
  function trainFrom(st, kt, at, i, params) {
    console.log(arguments);
    if (i == numIters) {
      return kt(st, params);
    }
    // TODO start from a trace?
    return MH(st, mhK, at, self.getSamplerWithParameters(params), numSamps - skip + 1);
    function mhK(sm, sampsDistr) {
      // TODO this might be wrong
      var samps = sampsDistr.support();
      console.log('samps', samps);
      var combinedCallLog = _.times(self.signatures.length, function() { return []; });
      for (var j = burnIn; j < numSamps; j += skip) {
        var callLog = samps[j][0];
        console.log('callLog', callLog);
        for (var k = 0; k < self.signatures.length; ++k) {
          [].push.apply(combinedCallLog[k], callLog[k]);
        }
      }
      var newParams = [];
      console.log('signatures', self.signatures);
      for (var j = 0; j < self.signatures.length; ++j) {
        newParams.push(trainParameters(self.signatures[j], combinedCallLog[j], params[j]));
      }
      return trainFrom(sm, kt, at, i+1, newParams);
    }
  }
};

function unknownParametersModel(s, k, a, fun) {
  var model = new UnknownParametersModel();
  function k2(s2, sampler) {
    console.log("in unknownparametersmodel continuation!");
    model.sampler = sampler;
    return k(s2, model);
  };
  console.log('fun', fun)
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
