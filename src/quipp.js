var _ = require('underscore');
var param_inference = require('./param_inference');
var expfam = require('./expfam');
var util = require('./util');

var fromMonad = util.fromMonad, mreturn = util.mreturn;

function factorScore(s, k, a, fn) {
  var args = [].slice.call(arguments, 4);
  return fn.factorScore.apply(this, [s, k, a].concat(args));
}

var Vector = fromMonad(function(n, t) {
  return mreturn(expfam.Tuple(_.times(n, function() { return t; })));
});


module.exports = {
  factorScore: factorScore,
  inferParameters: param_inference.inferParameters,
  testParamInference: param_inference.testParamInference,
  testParamInferenceSplit: param_inference.testParamInferenceSplit,
  Double: expfam.Double,
  Tuple: util.makeWpplFunction(expfam.Tuple),
  Vector: Vector,
  Categorical: util.makeWpplFunction(expfam.Categorical),
  randomValue: fromMonad(function(t) { return t.randomDefault; })
};
