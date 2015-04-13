var _ = require('underscore');
var param_inference = require('./param_inference');
var expfam = require('./expfam');

module.exports = {
  inferParameters: param_inference.inferParameters
};

_.extend(module.exports, expfam);
