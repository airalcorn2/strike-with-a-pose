$(document).ready(main);

function main(jQuery) {
  var app = new OBJLoader2Example(document.getElementById('example'));

  $(window).resize(function() {app.resizeDisplayGL();});

  console.log('Starting initialisation phase...');
  app.initGL();
  app.resizeDisplayGL();
  app.initContent();

  render_and_predict(app);
}

async function render_and_predict(app) {
  var model = await mobilenet.load();

  var guess = function() {
    var canvas = document.getElementById('example');
    var img = tf.fromPixels(canvas);
    return model.classify(img);
  };

  function MakeQuerablePromise(promise) {
    // Don't create a wrapper for promises that can already be queried.
    if (promise.isResolved) return promise;

    var isResolved = false;
    var isRejected = false;

    // Observe the promise, saving the fulfillment in a closure scope.
    var result = promise.then(
      function(v) {isResolved = true; return v;},
      function(e) {isRejected = true; throw e;});
    result.isFulfilled = function() {return isResolved || isRejected;};
    result.isResolved = function() {return isResolved;};
    result.isRejected = function() {return isRejected;};
    return result;
  }

  var t0 = performance.now();
  var job = null;
  var render = function(t1) {
    requestAnimationFrame(render);
    var d = app.render();
    if (t1 - t0 > 200 && d > 2e-3 && (!job || job.isResolved)) {
      t0 = t1;
      var canvas = document.getElementById('example');
      var img = tf.fromPixels(canvas);
      job = model.classify(img);
      job.then(ybar => {
        var $ybar = $('#predictions').empty();
        var i, text = '';
        for (i = 0; i < ybar.length; ++i) {
          text = ybar[i]['probability'] + ' ' + ybar[i]['className'];
          $ybar.append($('<p></p>').text(text));
        }
      });
      job = MakeQuerablePromise(job);
    };
  };

  render();
}

var OBJLoader2Example = function(elementToBindTo) {
  this.renderer = null;
  this.canvas = elementToBindTo;
  this.aspectRatio = 1;
  this.recalcAspectRatio();

  this.scene = null;
  this.cameraDefaults = {
    posCamera: new THREE.Vector3(0.0, 4.0, 15.0),
    posCameraTarget: new THREE.Vector3(0, 0, 0),
    near: 0.1,
    far: 10000,
    fov: 45,
  };
  this.camera = null;
  this.cameraTarget = this.cameraDefaults.posCameraTarget;

  this.controls = null;
};

OBJLoader2Example.prototype = {

  constructor: OBJLoader2Example,

  initGL: function() {
    this.renderer = new THREE.WebGLRenderer({
      canvas: this.canvas,
      antialias: true,
      autoClear: true,
    });
    this.renderer.setClearColor(0xFFFFFF, 0);

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0xFFFFFF);

    this.camera = new THREE.PerspectiveCamera(this.cameraDefaults.fov,
                                              this.aspectRatio,
                                              this.cameraDefaults.near,
                                              this.cameraDefaults.far);
    this.resetCamera();
    this.controls = new THREE.TrackballControls(this.camera,
                                                this.renderer.domElement);

    var ambientLight = new THREE.AmbientLight(0xFFFFFF);
    var directionalLight1 = new THREE.DirectionalLight(0xFFFFFF);
    var directionalLight2 = new THREE.DirectionalLight(0xFFFFFF);

    directionalLight1.position.set(-100, 100, 500);
    directionalLight2.position.set(100, 50, -500);

    this.scene.add(directionalLight1);
    this.scene.add(directionalLight2);
    this.scene.add(ambientLight);
  },

  initContent: function() {
    var modelName = 'Jeep';
    this._reportProgress({detail: {text: 'Loading: ' + modelName}});

    var scope = this;
    var objLoader = new THREE.OBJLoader2();
    var callbackOnLoad = function (event) {
      scope.scene.add(event.detail.loaderRootNode);
      console.log('Loading complete: ' + event.detail.modelName);
      scope._reportProgress({detail: {text: ''}});
    };

    var onLoadMtl = function (materials) {
      objLoader.setModelName(modelName);
      objLoader.setMaterials(materials);
      objLoader.setLogging(true, true);
      objLoader.load('models/Jeep.obj', callbackOnLoad, null, null,
                     null, false);
    };
    objLoader.loadMtl('models/Jeep.mtl', null, onLoadMtl);
  },

  _reportProgress: function(event) {
    var output = THREE.LoaderSupport.Validator.verifyInput(
      event.detail.text, '');
    console.log('Progress: ' + output);
    document.getElementById('feedback').innerHTML = output;
  },

  resizeDisplayGL: function() {
    this.controls.handleResize();

    const canvas = this.renderer.domElement;
    // look up the size the canvas is being displayed
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;

    var size = width < height ? width : height;

    // adjust displayBuffer size to match
    if (canvas.width !== size || canvas.height !== size) {
      canvas.width = size;
      canvas.height = size;
      // you must pass false here or three.js sadly fights the browser
      this.renderer.setSize(size, size, false);
      this.aspectRatio = 1.0;
      this.updateCamera();
    }
  },

  recalcAspectRatio: function() {
    this.aspectRatio = (this.canvas.offsetHeight === 0)
      ? 1
      : this.canvas.offsetWidth / this.canvas.offsetHeight;
  },

  resetCamera: function() {
    this.camera.position.copy(this.cameraDefaults.posCamera);
    this.cameraTarget.copy(this.cameraDefaults.posCameraTarget);
    this.updateCamera();
  },

  updateCamera: function() {
    this.camera.aspect = this.aspectRatio;
    this.camera.lookAt(this.cameraTarget);
    this.camera.updateProjectionMatrix();
  },

  render: function() {
    if (!this.renderer.autoClear) this.renderer.clear();
    var d = this.controls.update();
    this.renderer.render(this.scene, this.camera);
    return d;
  }
};
