
var assert = require('assert');
var numeric = require('numeric');
var _ = require('underscore');

var expfam = require('./expfam');

function vecAdd(x, y) {
  var res = [];
  for (var i = 0; i < x.length; ++i) {
    res.push(x[i] + y[i]);
  }
  return res;
}

function vecScale(s, x) {
  var res = [];
  for (var i = 0; i < x.length; ++i) {
    res.push(s*x[i]);
  }
  return res;
}

function dotProduct(x, y) {
  var res = 0.0;
  for (var i = 0; i < x.length; ++i) {
    res += x[i]*y[i];
  }
  return res;
}

function matRow(m, i) {
  return m[i];
}

function matCol(m, i) {
  return mat.map(function(row) { return row[i]; });
}

function matElements(m) {
  var res = [];
  m.forEach(function(row) {
    res.push.apply(res, row);
  });
  return res;
}

function elementsToMat(nrows, elems) {
  var ncols = elems.length / nrows;
  var mat = [];
  for (var i = 0; i < nrows; ++i) {
    var row = [];
    for (var j = 0; j < ncols; ++j) {
      row.push(elems[i*ncols + j]);
    }
    mat.push(row);
  }
  return mat;
}

function transpose(mat) {
  var res = [];
  for (var i = 0; i < mat[0].length; ++i) {
    res.push(matCol(mat, i));
  }
  return res;
}

function matMulByVector(m, x) {
  var res = [];
  for (var i = 0; i < m.length; ++i) {
    res.push(dotProduct(m[i], x));
  }
  return res;
}

function lineSearch(f, x, dir) {
  var t = 0;
  var fx = f(x);
  for (var i = 0; i < 100; ++i) {
    var x2 = vecAdd(x, vecScale(Math.pow(2, -t), dir));
    if (f(x2) >= fx) {
      return x2;
    }
  }
  throw new Exception('Found nothing better.  f(x) = ' + fx);
}

function linSolve(A, b) {
  return numeric.solve(A, b);
}

function newtonMethod(f, gradf, hessf, x) {
  for (var i = 0; i < 10; ++i) {
    x = lineSearch(f, x, vecScale(-1, linSolve(hessf(x), gradf(x))));
  }
  return x;
}


module.exports = {
  matElements: matElements,
  elementsToMat: elementsToMat,
  newtonMethod: newtonMethod
};

