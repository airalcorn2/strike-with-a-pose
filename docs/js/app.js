$(document).ready(main);

function main(jQuery) {
  var app = new OBJLoader2Example(document.getElementById('example'));

  $(window).resize(() => {app.resizeDisplayGL();});

  console.log('Starting initialisation phase...');
  app.initGL();
  app.resizeDisplayGL();
  app.initContent();

  render_and_predict(app);
}

function mobilecheck() {
  var check = false;
  (function(a){if(/(android|bb\d+|meego).+mobile|avantgo|bada\/|blackberry|blazer|compal|elaine|fennec|hiptop|iemobile|ip(hone|od)|iris|kindle|lge |maemo|midp|mmp|mobile.+firefox|netfront|opera m(ob|in)i|palm( os)?|phone|p(ixi|re)\/|plucker|pocket|psp|series(4|6)0|symbian|treo|up\.(browser|link)|vodafone|wap|windows ce|xda|xiino/i.test(a)||/1207|6310|6590|3gso|4thp|50[1-6]i|770s|802s|a wa|abac|ac(er|oo|s\-)|ai(ko|rn)|al(av|ca|co)|amoi|an(ex|ny|yw)|aptu|ar(ch|go)|as(te|us)|attw|au(di|\-m|r |s )|avan|be(ck|ll|nq)|bi(lb|rd)|bl(ac|az)|br(e|v)w|bumb|bw\-(n|u)|c55\/|capi|ccwa|cdm\-|cell|chtm|cldc|cmd\-|co(mp|nd)|craw|da(it|ll|ng)|dbte|dc\-s|devi|dica|dmob|do(c|p)o|ds(12|\-d)|el(49|ai)|em(l2|ul)|er(ic|k0)|esl8|ez([4-7]0|os|wa|ze)|fetc|fly(\-|_)|g1 u|g560|gene|gf\-5|g\-mo|go(\.w|od)|gr(ad|un)|haie|hcit|hd\-(m|p|t)|hei\-|hi(pt|ta)|hp( i|ip)|hs\-c|ht(c(\-| |_|a|g|p|s|t)|tp)|hu(aw|tc)|i\-(20|go|ma)|i230|iac( |\-|\/)|ibro|idea|ig01|ikom|im1k|inno|ipaq|iris|ja(t|v)a|jbro|jemu|jigs|kddi|keji|kgt( |\/)|klon|kpt |kwc\-|kyo(c|k)|le(no|xi)|lg( g|\/(k|l|u)|50|54|\-[a-w])|libw|lynx|m1\-w|m3ga|m50\/|ma(te|ui|xo)|mc(01|21|ca)|m\-cr|me(rc|ri)|mi(o8|oa|ts)|mmef|mo(01|02|bi|de|do|t(\-| |o|v)|zz)|mt(50|p1|v )|mwbp|mywa|n10[0-2]|n20[2-3]|n30(0|2)|n50(0|2|5)|n7(0(0|1)|10)|ne((c|m)\-|on|tf|wf|wg|wt)|nok(6|i)|nzph|o2im|op(ti|wv)|oran|owg1|p800|pan(a|d|t)|pdxg|pg(13|\-([1-8]|c))|phil|pire|pl(ay|uc)|pn\-2|po(ck|rt|se)|prox|psio|pt\-g|qa\-a|qc(07|12|21|32|60|\-[2-7]|i\-)|qtek|r380|r600|raks|rim9|ro(ve|zo)|s55\/|sa(ge|ma|mm|ms|ny|va)|sc(01|h\-|oo|p\-)|sdk\/|se(c(\-|0|1)|47|mc|nd|ri)|sgh\-|shar|sie(\-|m)|sk\-0|sl(45|id)|sm(al|ar|b3|it|t5)|so(ft|ny)|sp(01|h\-|v\-|v )|sy(01|mb)|t2(18|50)|t6(00|10|18)|ta(gt|lk)|tcl\-|tdg\-|tel(i|m)|tim\-|t\-mo|to(pl|sh)|ts(70|m\-|m3|m5)|tx\-9|up(\.b|g1|si)|utst|v400|v750|veri|vi(rg|te)|vk(40|5[0-3]|\-v)|vm40|voda|vulc|vx(52|53|60|61|70|80|81|83|85|98)|w3c(\-| )|webc|whit|wi(g |nc|nw)|wmlb|wonu|x700|yas\-|your|zeto|zte\-/i.test(a.substr(0,4))) check = true;})(navigator.userAgent||navigator.vendor||window.opera);
  return check;
};

async function render_and_predict(app) {
  const canvas = document.getElementById('example');
  const model = await mobilenet.load();

  function WrapPromise(p) {
    if (p.resolved) return p;
    var resolved = false;
    var result = [];
    let then = p.then(ybar => {resolved = true; return res = ybar;});
    return then;
  }

  function show_result(res) {
    const $ybar = $('#predictions').empty();
    for (let i = 0; i < res.length; ++i) {
      let t = res[i]['probability'].toFixed(8) + ' ' + res[i]['className'];
      $ybar.append($('<p></p>').text(t));
    }
  }

  const pred_fps = mobilecheck() ? 2 : 5;
  var t0 = performance.now();
  var moving = false;
  var res = [], jobs = [];
  var render = function(t1) {
    setTimeout(() => {requestAnimationFrame(render);}, 1000 / 60);
    let moved = app.render() > 1e-4;
    if (!moved && moving) {
      Promise.all(jobs).then(() => {
        const img = tf.fromPixels(canvas);
        model.classify(img).then(show_result);
      });
    } else if (moved && t1 - t0 > 1000 / pred_fps) {
      t0 = t1;
      const img = tf.fromPixels(canvas);
      const job = model.classify(img).then(ybar => {return res = ybar;});
      jobs.push(job);
      show_result(res);
    };
    moving = moved;
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

    let vw = Math.max(document.documentElement.clientWidth,
                      window.innerWidth || 0);
    let vh = Math.max(document.documentElement.clientHeight,
                      window.innerHeight || 0);

    if (vw >= 800) vw /= 2;
    var size = (vw < vh ? vw : vh) * 0.8;

    // adjust displayBuffer size to match
    if (canvas.width !== size || canvas.height !== size) {
      canvas.width = size;
      canvas.height = size;
      // you must pass false here or three.js sadly fights the browser
      this.renderer.setSize(size, size, false);
      this.aspectRatio = 1.0;
      this.updateCamera();
    }

    return size;
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
    let d = this.controls.update();
    this.renderer.render(this.scene, this.camera);
    return d;
  }
};
