
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
  linkedListToArray: linkedListToArray,
  arrayToLinkedList: arrayToLinkedList,
  makeWpplFunction: makeWpplFunction,
  concat: concat,
  wpplMap: wpplMap
};
