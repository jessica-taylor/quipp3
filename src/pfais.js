////////////////////////////////////////////////////////////////////
// Particle filtering
//
// Sequential importance re-sampling, which treats 'factor' calls as
// the synchronization / intermediate distribution points.

'use strict';

var _ = require('underscore');
var ad = require('./autodiff');
var webpplUtil = require('webppl/src/util');
var erp = require('webppl/src/erp');
var expfam = require('./expfam');
var util = require('./util');
var optimization = require('./optimization'); 
var mbind = util.mbind, mreturn = util.mreturn, mcurry = util.mcurry, fromMonad = util.fromMonad;

function objective(t, ef, samps, np) {
  var unnormalizedLps = samps.map(function(samp) {
    return ad.numDotProduct(t, np, samp[1].map(t.num));
  });
  var estG = ad.numLogSumExp(t, unnormalizedLps);
  estG = t.sub(estG, t.num(Math.log(samps.length)));
  var actualG = ef.g(t, np);
  var totLp = t.num(0);
  unnormalizedLps.forEach(function(lp, i) {
    var thisLp = t.sub(t.sub(lp, estG), t.mul(t.num(0.01), actualG));
    totLp = t.add(totLp, t.mul(t.num(samps[i][0]), thisLp));
  });
  return totLp;
}

function optNp(ef, samps, origNp) {
  function scoreFn(t, np) {
    return objective(t, ef, samps, np);
  }
  return optimization.gradientDescent(
      function(np) {
        return scoreFn(ad.standardNumType, np);
      },
      function(np) {
        return ad.gradient(ad.standardNumType, scoreFn, np).grad;
      },
      origNp);
}


module.exports = function(env) {

  function copyParticle(particle) {
    return {
      continuation: particle.continuation,
      weight: particle.weight,
      value: particle.value,
      store: _.clone(particle.store)
    };
  }

  function erpRetType(erpp, params) {
    if (erpp == erp.gaussianERP) {
      var mean = params[0];
      var variance = params[1];
      return [expfam.Double, [mean / variance, 1 / (2 * variance)]];
    } else if (erpp == erp.bernoulliERP) {
      return [expfam.Categorical(2), null];
    } else if (erpp == erp.randomIntegerERP) {
      return [expfam.Categorical(params[0]), _.times(params[0] - 1, function() { return 0; })];
    } else if (erpp == erp.discreteERP) {
      return [expfam.Categorical(params[0].length), null];
    }
    console.log('argh :/', erpp)
    return null;
  }

  function ParticleFilter(s, k, a, wpplFn, numParticles, aisIters, strict) {

    this.aisIters = aisIters;
    this.aisRetTypes = [];
    this.aisTrainingSamples = [];
    this.aisParams = [];
    this.aisIter = 0;
    this.sampleIndex = 0;
    this.particles = [];
    this.particleIndex = 0;  // marks the active particle

    // Create initial particles
    var exitK = function(s) {
      return wpplFn(s, env.exit, a);
    };
    for (var i = 0; i < numParticles; i++) {
      var particle = {
        continuation: exitK,
        weight: 0,
        value: undefined,
        store: _.clone(s)
      };
      this.particles.push(particle);
    }
    // TODO clone store?
    this.particlesAtLastFactor = this.particles.map(copyParticle);

    this.strict = strict;
    // Move old coroutine out of the way and install this as the current
    // handler.
    this.k = k;
    this.oldCoroutine = env.coroutine;
    env.coroutine = this;

    this.oldStore = _.clone(s); // will be reinstated at the end
  }

  ParticleFilter.prototype.run = function() {
    // Run first particle
    return this.activeParticle().continuation(this.activeParticle().store);
  };

  ParticleFilter.prototype.sample = fromMonad(function(erp, params) {
    var self = this;
    if (this.sampleIndex >= this.aisParams.length) {
      if (this.sampleIndex >= this.aisRetTypes.length) {
        this.aisRetTypes.push(erpRetType(erp, params)[0]);
        this.aisTrainingSamples.push([]);
      }
      assert.equal(this.aisRetTypes[this.sampleIndex].name, erpRetType(erp, params)[0].name);
      // TODO make sure type consistent?
      var samp = erp.sample(params);
      this.aisTrainingSamples[this.sampleIndex].push([this.particleIndex, samp]);
      ++this.sampleIndex;
      return mreturn(samp);
    } else {
      var retType = this.aisRetTypes[this.sampleIndex];
      assert.equal(retType.name, erpRetType(erp, params)[0].name);
      var origParams = erpRetType(erp, params)[1];
      var qParams = this.aisParams[this.sampleIndex];
      qParams = {
        base: qParams.base.map(function(q, i) { return 0.5 * q + 0.5 * origParams[i]; }),
        weights: qParams.weights
      }
      return mbind(expfam.simpleSample, retType, qParams.base, function(samp) {
        self.aisTrainingSamples[self.sampleIndex].push([self.particleIndex, samp]);
        ++self.sampleIndex;
        // Importance weight
        var q = expfam.logProbability(ad.standardNumType, retType, qParams, [], retType.sufStat(samp));
        var p = erp.score(params, samp);
        self.activeParticle().weight += p - q;
        return mreturn(samp);
      });
    }
  });

  ParticleFilter.prototype.factor = function(s, cc, a, score) {
    // Update particle weight
    this.activeParticle().weight += score;
    this.activeParticle().continuation = cc;
    this.activeParticle().store = s;
    this.sampleIndex = 0;

    if (this.allParticlesAdvanced()) {
      this.particleIndex = 0;
      if (this.aisIter == this.aisIters) {
        // Resample in proportion to weights
        this.resampleParticles();
        this.particlesAtLastFactor = this.particles.map(copyParticle);
      } else {
        this.retrainAis();
        this.aisIter++;
        this.aisTrainingSamples = this.aisTrainingSamples.map(function() { return []; });
        this.particles = this.particlesAtLastFactor.map(copyParticle);
      }
    } else {
      // Advance to the next particle
      this.particleIndex += 1;
    }

    return this.activeParticle().continuation(this.activeParticle().store);
  };

  ParticleFilter.prototype.activeParticle = function() {
    return this.particles[this.particleIndex];
  };

  ParticleFilter.prototype.allParticlesAdvanced = function() {
    return ((this.particleIndex + 1) === this.particles.length);
  };

  ParticleFilter.prototype.retrainAis = function() {
    var self = this;
    var W = webpplUtil.logsumexp(_.map(this.particles, function(p) {
      return p.weight;
    }));
    for (var i = 0; i < this.aisRetTypes.length; ++i) {
      var retType = this.aisRetTypes[i];
      var oldParams = this.aisParams[i] || expfam.defaultParameters([[], retType]);
      var samps = this.aisTrainingSamples[i].map(function(s) {
        var whichParticle = s[0];
        var val = s[1];
        var weight = self.particles.length * Math.exp(self.particles[whichParticle].weight - W);
        return [weight, retType.sufStat(val)];
      });
      // var newParams = expfam.mle(retType, samps, oldParams);
      var newParams = {
        base: optNp(retType, samps, retType.defaultNatParam),
        weights: _.times(expfam.featuresDim(retType), function() { return []; })
      };
      console.log('newParams', newParams);
      this.aisParams[i] = newParams;
    }
  };

  ParticleFilter.prototype.resampleParticles = function() {
    // Residual resampling following Liu 2008; p. 72, section 3.4.4
    var m = this.particles.length;
    var W = webpplUtil.logsumexp(_.map(this.particles, function(p) {
      return p.weight;
    }));
    var avgW = W - Math.log(m);

    if (avgW == -Infinity) {      // debugging: check if NaN
      if (this.strict) {
        throw 'Error! All particles -Infinity';
      }
    } else {
      // Compute list of retained particles
      var retainedParticles = [];
      var newExpWeights = [];
      _.each(
          this.particles,
          function(particle) {
            var w = Math.exp(particle.weight - avgW);
            var nRetained = Math.floor(w);
            newExpWeights.push(w - nRetained);
            for (var i = 0; i < nRetained; i++) {
              retainedParticles.push(copyParticle(particle));
            }
          });
      // Compute new particles
      var numNewParticles = m - retainedParticles.length;
      var newParticles = [];
      var j;
      for (var i = 0; i < numNewParticles; i++) {
        j = erp.multinomialSample(newExpWeights);
        newParticles.push(copyParticle(this.particles[j]));
      }

      // Particles after update: Retained + new particles
      this.particles = newParticles.concat(retainedParticles);
    }

    // Reset all weights
    _.each(this.particles, function(particle) {
      particle.weight = avgW;
    });
  };

  ParticleFilter.prototype.exit = function(s, retval) {

    this.activeParticle().value = retval;

    // Wait for all particles to reach exit before computing
    // marginal distribution from particles
    if (!this.allParticlesAdvanced()) {
      this.particleIndex += 1;
      return this.activeParticle().continuation(this.activeParticle().store);
    }

    // Compute marginal distribution from (unweighted) particles
    var hist = {};
    _.each(
        this.particles,
        function(particle) {
          var k = JSON.stringify(particle.value);
          if (hist[k] === undefined) {
            hist[k] = {prob: 0, val: particle.value};
          }
          hist[k].prob += 1;
        });
    var dist = erp.makeMarginalERP(hist);

    // Save estimated normalization constant in erp (average particle weight)
    dist.normalizationConstant = this.particles[0].weight;

    // Reinstate previous coroutine:
    env.coroutine = this.oldCoroutine;

    // Return from particle filter by calling original continuation:
    return this.k(this.oldStore, dist);
  };

  function pf(s, cc, a, wpplFn, numParticles, strict) {
    return new ParticleFilter(s, cc, a, wpplFn, numParticles, strict === undefined ? true : strict).run();
  }

  return {
    ParticleFilter: pf
  };

};
