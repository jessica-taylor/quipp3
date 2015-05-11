
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

function mbind(x) {
  assert(x != null);
  console.log('mbind', x);
  var args = [].slice.call(arguments);
  var extraArgs = args.slice(1, args.length - 1);
  var f = args[args.length - 1];
  return function(s, k, a) {
    return x.apply(this, [s, k2, a].concat(extraArgs));
    function k2(s2, res) {
      return f(res)(s2, k, a);
    }
  }
}

function mbindMethod(x, method) {
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

function fmap(x) {
  var args = [].slice.call(arguments);
  var extraArgs = args.slice(1, args.length - 1);
  var f = args[args.length - 1];
  return mbind.apply(null, [x].concat(extraArgs).concat([function(res) {
    return mreturn(f(res));
  }]));
}

function fromMonad(f) {
  assert(f != null);
  return function(s, k, a) {
    var args = [].slice.call(arguments, 3);
    console.log('f', f, f.apply(this, args));
    return f.apply(this, args)(s, k, a);
  };
}

function mcurry(f) {
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

var replicateM = fromMonad(function(n, f) {
  return fmap(n, f, linkedListToArray);
});

var mapMlist = fromMonad(function(xs, f) {
  assert(f != null);
  if (xs == null) {
    return mreturn(null);
  }
  return mbind(f, xs[0], function(first) {
    return mbind(mapMlist, xs[1], f, function(rest) {
      return [first, rest];
    });
  });
});

var mapM = fromMonad(function(xs, f) {
  return fmap(mapMlist, arrayToLinkedList(xs), f, linkedListToArray);
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
