var _ = require('underscore');

var expfam = require('./expfam');
var ad = require('./autodiff');

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
    });
  };
}

function getArgFeatures(argTypes, argVals) {
  return concat(argTypes.map(function(at, i) { return expfam.getFeatures(at, argVals[i]); }));
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

UnknownParametersModel.prototype.randFunction = function() {
  var self = this;
  var args = [].splice.call(arguments);
  var argTypes = args.splice(0, args.length - 1);
  var retType = args[args.length - 1];
  var i = this.signatures.length;
  this.signatures.push([argTypes, retType]);
  return function(s, k, a) {
    var args = [].splice.call(arguments, 3);
    var natParam = natParamWithArguments(self.signatures[i][0], s._quippParams[i], args);
    var result = self.signatures[i][1].sample(natParam);
    s._quippCallLog[i].push([args, result]);
    return result;
  };
}

UnknownParametersModel.prototype.getSamplerWithParameters = function(params) {
  var self = this;
  assert(this.sampler);
  return function(s, k, a) {
    s._quippCallLog = _.times(self.signatures.length, function() { return []; });
    s._quippParams = params;
    return self.sampler(s,
      function(s2, x) { k([s2._quippCallLog, x]); },
      a);
  };
};

UnknownParametersModel.prototype.inferParameters = function() {
  var params = this.signatures.map(defaultParameters);
  for (int i = 0; i < 10; ++i) {
    var samps = MH({}, function() { }, '', this.getSamplerWithParameters(params), 1000);
  }
};

function unknownParametersModel(fun) {
  var model = new UnknownParametersModel();
  var sampler = fun({}, k, '', model);
  function k(sampler) {
    model.sampler = sampler;
  };
  return model;
}

