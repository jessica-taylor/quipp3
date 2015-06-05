var _ = require('underscore');
var param_inference = require('./param_inference');
var expfam = require('./expfam');
var util = require('./util');

var fromMonad = util.fromMonad, mreturn = util.mreturn, mcurry = util.mcurry;

function factorScore(s, k, a, fn) {
  var args = [].slice.call(arguments, 4);
  return fn.factorScore.apply(this, [s, k, a].concat(args));
}

var Vector = fromMonad(function(n, t) {
  return mreturn(expfam.Tuple(_.times(n, function() { return t; })));
});

var Tuple2 = fromMonad(function(args) {
  return mreturn(expfam.Tuple([].slice.call(args)));
});

var Bool = expfam.Categorical(2);


module.exports = {
  factorScore: factorScore,
  inferParameters: param_inference.inferParameters,
  testParamInference: param_inference.testParamInference,
  testParamInferenceSplit: param_inference.testParamInferenceSplit,
  Double: expfam.Double,
  Bool: Bool,
  Tuple: Tuple2,
  Vector: Vector,
  Categorical: util.makeWpplFunction(expfam.Categorical),
  randomValue: fromMonad(function(t) { return t.randomDefault; })
};
