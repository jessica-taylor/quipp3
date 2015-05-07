var _ = require('underscore');
var param_inference = require('./param_inference');
var expfam = require('./expfam');
var util = require('./util');

function factorScore(s, k, a, fn) {
  var args = [].slice.call(arguments, 4);
  return fn.factorScore.apply(this, [s, k, a].concat(args));
}

module.exports = {
  factorScore: factorScore,
  inferParameters: param_inference.inferParameters,
  testParamInference: param_inferenc.testParamInference,
  Double: expfam.Double,
  Categorical: util.makeWpplFunction(expfam.Categorical)
};
