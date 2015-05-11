
Error.stackTraceLimit = Infinity;
// mbind(f, arg1, arg2, cb)
//
// is equivalent to
//
// function(s, k, a) {
//   return f(s, k2, a, arg1, arg2);
//   function k2(s2, result) {
//     return cb(result)(s2, k, a);
//   }
// }
//


var labelStack = [];

function withStackMessage(msg, x) {
  return function(s, k, a) {
    labelStack.push(msg);
    return x(s, k2, a);
    function k2(s2, res) {
      labelStack.pop();
      return k(s2, res);
    }
  };
}

function catchStack(x) {
  try {
    return x();
  } catch (ex) {
    console.log(ex.stack);
    console.log(labelStack);
    console.log('!!');
    process.exit();
  }
}


function mbind(x) {
  assert(typeof x == 'function', x);
  var args = [].slice.call(arguments);
  var extraArgs = args.slice(1, args.length - 1);
  var f = args[args.length - 1];
  assert(typeof f == 'function', f);
  return function(s, k, a) {
    var self = this;
    return catchStack(function() { return x.apply(self, [s, k2, a].concat(extraArgs)); });
    function k2(s2, res) {
      return catchStack(function() {
        var nextf = f(res);
        assert (typeof nextf == 'function', nextf);
        return nextf(s2, k, a);
      });
    }
  }
}

function mbindMethod(x, method) {
  assert(typeof x == 'object', x);
  assert(typeof method == 'string', method);
  var args = [].slice.call(arguments);
  var extraArgs = args.slice(2, args.length - 1);
  var f = args[args.length - 1];
  return function(s, k, a) {
    return x[method].apply(x, [s, k2, a].concat(extraArgs));
    function k2(s2, res) {
      return f(res)(s2, k, a);
    }
  }
}

function mreturn(x) {
  return function(s, k, a) {
    return k(s, x);
  };
}

function fromMonad(msg, f) {
  if (f == null) {
    f = msg;
    msg = null;
  }
  assert(typeof f == 'function');
  return function(s, k, a) {
    var args = [].slice.call(arguments, 3);
    if (msg == null) {
      return f.apply(this, args)(s, k, a);
    } else {
      return withStackMessage(msg, f.apply(this, args))(s, k, a);
    }
  };
}

function mcurry(f) {
  assert(typeof f == 'function');
  var args = [].slice.call(arguments, 1);
  return function(s, k, a) {
    var extraArgs = [].slice.call(arguments, 3);
    return f.apply(this, [s, k, a].concat(args).concat(extraArgs));
  };
}

var replicateMlist = fromMonad(function(n, f) {
  if (n == 0) {
    return mreturn(null);
  }
  return mbind(f, function(first) {
    return mbind(replicateMlist, n-1, f, function(rest) {
      return mreturn([first, rest]);
    });
  });
});

var replicateM = fromMonad('replicateM', function(n, f) {
  assert(typeof n == 'number');
  assert(typeof f == 'function');
  return mbind(replicateMlist, n, f, function(res) {
    return mreturn(linkedListToArray(res));
  });
});

var mapMlist = fromMonad('mapMlist', function(xs, f) {
  assert(f != null);
  if (xs == null) {
    return mreturn(null);
  }
  return mbind(f, xs[0], function(first) {
    return mbind(mapMlist, xs[1], f, function(rest) {
      return mreturn([first, rest]);
    });
  });
});

var mapM = fromMonad('mapM', function(xs, f) {
  assert(typeof f == 'function');
  return mbind(mapMlist, arrayToLinkedList(xs), f, function(res) {
    return mreturn(linkedListToArray(res));
  });
});

function makeWpplFunction(fn) {
  return fromMonad(function() {
    return mreturn(fn.apply(this, arguments));
  });
}

function makeWpplFunction(fn) {
  return function(e, k, a) {
    var args = [].slice.call(arguments, 3);
    var result = fn.apply(this, args);
    return k(e, result);
  };
}

function concat(lsts) {
  var c = [];
  lsts.forEach(function(lst) {
    c.push.apply(c, lst);
  });
  return c;
}

function sum(lst) {
  var tot = 0.0;
  lst.forEach(function(l) { tot += l; });
  return tot;
}

function arrayToLinkedList(xs) {
  function from(i) {
    if (i == xs.length) {
      return null;
    } else {
      return [xs[i], from(i+1)];
    }
  }
  return from(0);
}

function linkedListToArray(xs) {
  var result = [];
  while (xs != null) {
    result.push(xs[0]);
    xs = xs[1];
  }
  return result;
}

function wpplMapToLinkedList(s, k, a, f, i, xs) {
  // TODO change a?
  if (i == xs.length) k(s, null);
  function k2(s2, y) {
    function k3(s3, ys) {
      return k(s3, [y, ys]);
    }
    return wpplMapLinkedList(s2, k3, a, i+1, xs);
  }
  return f(s, k2, a, xs[i], i);
}

function wpplMap(s, k, a, f, xs) {
  function k2(s2, ys) {
    return k(s2, linkedListToArray(ys));
  }
  return wpplMapLinkedList(s, k2, a, f, 0, xs);
}

module.exports = {
  mbind: mbind,
  mbindMethod: mbindMethod,
  mreturn: mreturn,
  mcurry: mcurry,
  fromMonad: fromMonad,
  replicateM: replicateM,
  mapM: mapM,
  linkedListToArray: linkedListToArray,
  arrayToLinkedList: arrayToLinkedList,
  makeWpplFunction: makeWpplFunction,
  concat: concat,
  wpplMap: wpplMap,
  sum: sum
};
