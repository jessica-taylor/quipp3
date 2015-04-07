
// NumType interface:
// .num(n)
// .add(x, y)
// .sub(x, y)
// .mul(x, y)
// .div(x, y)


var standardNumType = {
  num: function(x) { return x; },
  add: function(x, y) { return x+y; },
  sub: function(x, y) { return x-y; },
  mul: function(x, y) { return x*y; },
  div: function(x, y) { return x/y; }
};

function numVecAdd(t, x, y) {
  var res = [];
  for (int i = 0; i < x.length; ++i) {
    res.push(t.add(x[i], y[i]));
  }
  return res;
}

function numVecSub(t, x, y) {
  var res = [];
  for (int i = 0; i < x.length; ++i) {
    res.push(t.sub(x[i], y[i]));
  }
  return res;
}

function numVecScale(t, s, x) {
  var res = [];
  for (int i = 0; i < x.length; ++i) {
    res.push(t.mul(s, x[i]));
  }
  return res;
}

function Dual(value, grad) {
  this.value = value;
  this.grad = grad;
}

function dualNumType(t, varVals) {
  var n = varVals.length;
  return {
    num: function(v) {
      return new Dual(t.num(v), _.times(n, function() { return t.num(0); }));
    },
    constant: function(v) {
      return new Dual(v, _.times(n, function() { return t.num(0); }));
    },
    variable: function(i) {
      return new Dual(varVals[i], _.times(n, function(j) { return t.num(Number(i == j)); }));
    },
    add: function(x, y) {
      return new Dual(
          basetype.add(x.value, y.value),
          numVecAdd(t, x.grad, y.grad)
      );
    },
    sub: function(x, y) {
      return new Dual(
          t.sub(x.value, y.value),
          numVecAdd(t, x.grad, numVecScale(t.num(-1), y.grad))
      );
    },
    mul: function(x, y) {
      return new Dual(
          t.mul(x.value, y.value),
          numVecAdd(t,
                    numVecScale(t, y.value, x.grad),
                    numVecScale(t, x.value, y.grad))
      );
    },
    div: function(x, y) {
      return new Dual(
          t.div(x.value, y.value),
          numVecScale(t,
                      t.div(t.num(1), t.mul(y.value, y.value)),
                      numVecSub(t,
                                numVecScale(t, y.value, x.grad),
                                numVecScale(t, x.value, y.grad)))
      );
    }
  };
}


function gradient(t, f, x) {
  var dt = dualNumType(t, x);
  var dx = _.times(x.length, function(i) { return dt.variable(i); });
  var res = f(dt, dx);
  return res;
}

function hessian(t, f, x) {
  var dt = dualNumType(t, x);
  var dx = _.times(x.length, function(i) { return dt.variable(i); });
  var ddt = dualNumType(dt, dx);
  var ddx = _.times(x.length, function(i) { return ddt.variable(i); });
  var res = f(ddt, ddx);
  return res;
}

function hessian2(t, f, x) {
  function df(t2, x2) {
    return gradient(t2, f, x2);
  }
  var dt = dualNumType(t, x);
  var dx = _.times(x.length, function(i) { return dt.variable(i); });
  return gradient(dt, df, dx);
}

module.exports = {
  standardNumType: standardNumType,
  numVecAdd: numVecAdd,
  numVecSub: numVecSub,
  numVecScale: numVecScale,
  Dual: Dual,
  dualNumType: dualNumType,
  gradient: gradient,
  hessian: hessian
};
