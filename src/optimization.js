
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


function PositiveSearcher() {
  this.min = 0;
  this.max = null;
}

PositiveSearcher.prototype.getT = function() {
  if (this.max === null) {
    if (this.min === 0) {
      return 1;
    } else {
      return this.min * 2;
    }
  } else {
    return (this.min + this.max) / 2;
  }
}

PositiveSearcher.prototype.isAbove = function(min) {
  this.min = Math.max(this.min, min);
};

PositiveSearcher.prototype.isBelow = function(max) {
  if (this.max === null) {
    this.max = max;
  } else {
    this.max = Math.min(this.max, max);
  }
};

function LogSearcher() {
  this.sign = null;
  this.posSearcher = new PositiveSearcher();
}

LogSearcher.prototype.getT = function() {
  if (this.sign === null) {
    return 1;
  } else {
    return Math.exp(this.sign * this.posSearcher.getT());
  }
};

LogSearcher.prototype.isAbove = function(min) {
  if (this.sign === null) {
    assert(min === 1);
    this.sign = 1;
  } else if (this.sign === 1) {
    this.posSearcher.isAbove(Math.log(min));
  } else {
    this.posSearcher.isBelow(-Math.log(min));
  }
};

LogSearcher.prototype.isBelow = function(max) {
  if (this.sign === null) {
    assert(max === 1);
    this.sign = -1;
  } else if (this.sign === 1) {
    this.posSearcher.isBelow(Math.log(max));
  } else {
    this.posSearcher.isAbove(-Math.log(max));
  }
};



function gradLineSearch(f, gradf, x, dir) {
  function gradfdir(t) {
    return dotProduct(gradf(vecAdd(x, vecScale(t, dir))), dir);
  }
  var searcher = new LogSearcher();
  for (var iter = 0; iter < 20; iter++) {
    var t = searcher.getT();
    // console.log('t', t);

    if (Number.isNaN(f(vecAdd(x, vecScale(t, dir)))) || gradfdir(t) <= 0) {
      searcher.isBelow(t);
    } else {
      searcher.isAbove(t);
    }
  }
  var tfinal = searcher.getT();
  return vecAdd(x, vecScale(tfinal, dir));
}


function gradientDescent(f, gradf, x) {
  for (var iter = 0; iter < 10; iter++) {
    var grad = gradf(x);
    var xNew = gradLineSearch(f, gradf, x, grad);
    x = xNew;
  }
  return x;
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
  newtonMethod: newtonMethod,
  gradientDescent: gradientDescent
};

