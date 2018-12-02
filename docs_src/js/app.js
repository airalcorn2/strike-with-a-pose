const model = mobilenet.load();

function main(jQuery) {
  console.log('hello');

  var app = new OBJLoader2Example(document.getElementById('example'));

  var render = function() {
    requestAnimationFrame(render);
    app.render();
  };

  $(window).resize(function() {app.resizeDisplayGL();});

  console.log( 'Starting initialisation phase...' );
  app.initGL();
  app.resizeDisplayGL();
  app.initContent();

  render();

  $(document).keyup(function(evt) {
    if (evt.which != 13) return;
    console.log('hello, world');
    var canvas = document.getElementById('example');
    var img = tf.fromPixels(canvas);
    // console.log(img);
    mobilenet.load().then(model => {
      // Classify the image.
      model.classify(img).then(predictions => {
        console.log('Predictions: ');
        console.log(predictions);
        console.log(predictions);
        $ybar = $('#ybar').empty();
        var i, text = '';
        for (i = 0; i < predictions.length; ++i) {
          text = predictions[i]['probability'] + ' ' + predictions[i]['className'];
          $ybar.append($('<p></p>').text(text));
        }
      });
    });
  });
}

$(document).ready(main);

var OBJLoader2Example = function ( elementToBindTo ) {
  this.renderer = null;
  this.canvas = elementToBindTo;
  this.aspectRatio = 1;
  this.recalcAspectRatio();

  this.scene = null;
  this.cameraDefaults = {
    posCamera: new THREE.Vector3( 0.0, 400.0, 1000.0 ),
    posCameraTarget: new THREE.Vector3( 0, 0, 0 ),
    near: 0.1,
    far: 10000,
    fov: 45
  };
  this.camera = null;
  this.cameraTarget = this.cameraDefaults.posCameraTarget;

  this.controls = null;
};

OBJLoader2Example.prototype = {

  constructor: OBJLoader2Example,

  initGL: function () {
    this.renderer = new THREE.WebGLRenderer( {
      canvas: this.canvas,
      antialias: true,
      autoClear: true,
      preserveDrawingBuffer: true,
    } );
    // this.renderer.setClearColor( 0x050505 );
    // this.renderer.setClearColor( 0xfffff0, 0);

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color( 0xffffff );

    this.camera = new THREE.PerspectiveCamera( this.cameraDefaults.fov, this.aspectRatio, this.cameraDefaults.near, this.cameraDefaults.far );
    this.resetCamera();
    this.controls = new THREE.TrackballControls( this.camera, this.renderer.domElement );

    // var ambientLight = new THREE.AmbientLight( 0xA0A0A0 );
    // var directionalLight1 = new THREE.DirectionalLight( 0xC0C090 );
    // var directionalLight2 = new THREE.DirectionalLight( 0xC0C090 );
    var ambientLight = new THREE.AmbientLight( 0xFFFFFF );
    var directionalLight1 = new THREE.DirectionalLight( 0xFFFFFF );
    var directionalLight2 = new THREE.DirectionalLight( 0xFFFFFF );

    directionalLight1.position.set( -100, 100, 500 );
    directionalLight2.position.set( 100, 50, -500 );

    this.scene.add( directionalLight1 );
    this.scene.add( directionalLight2 );
    this.scene.add( ambientLight );

    /* var helper = new THREE.GridHelper( 1200, 60, 0xFF4444, 0x404040 );
     * this.scene.add( helper ); */
  },

  initContent: function () {
    var modelName = 'schoolbus01';
    this._reportProgress( { detail: { text: 'Loading: ' + modelName } } );

    var scope = this;
    var objLoader = new THREE.OBJLoader2();
    var callbackOnLoad = function ( event ) {
      scope.scene.add( event.detail.loaderRootNode );
      console.log( 'Loading complete: ' + event.detail.modelName );
      scope._reportProgress( { detail: { text: '' } } );
    };

    var onLoadMtl = function ( materials ) {
      objLoader.setModelName( modelName );
      objLoader.setMaterials( materials );
      objLoader.setLogging( true, true );
      objLoader.load( 'models/schoolbus01_49.obj', callbackOnLoad, null, null, null, false );
    };
    objLoader.loadMtl( 'models/schoolbus01_49.mtl', null, onLoadMtl );
  },

  _reportProgress: function( event ) {
    var output = THREE.LoaderSupport.Validator.verifyInput( event.detail.text, '' );
    console.log( 'Progress: ' + output );
    document.getElementById( 'feedback' ).innerHTML = output;
  },

  resizeDisplayGL: function () {
    this.controls.handleResize();

    this.recalcAspectRatio();
    this.renderer.setSize( this.canvas.offsetWidth, this.canvas.offsetHeight, false );

    this.updateCamera();
  },

  recalcAspectRatio: function () {
    this.aspectRatio = ( this.canvas.offsetHeight === 0 ) ? 1 : this.canvas.offsetWidth / this.canvas.offsetHeight;
  },

  resetCamera: function () {
    this.camera.position.copy( this.cameraDefaults.posCamera );
    this.cameraTarget.copy( this.cameraDefaults.posCameraTarget );

    this.updateCamera();
  },

  updateCamera: function () {
    this.camera.aspect = this.aspectRatio;
    this.camera.lookAt( this.cameraTarget );
    this.camera.updateProjectionMatrix();
  },

  render: function () {
    if ( ! this.renderer.autoClear ) this.renderer.clear();
    this.controls.update();
    this.renderer.render( this.scene, this.camera );
  }

};
