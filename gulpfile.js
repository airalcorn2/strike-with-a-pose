const $ = require('gulp');
const $bro = require('gulp-bro');
const $changed = require('gulp-changed');
const $concat = require('gulp-concat');
const $htmlmin = require('gulp-htmlmin');
const $plumber = require('gulp-plumber');
const $postcss = require('gulp-postcss');
const $sourcemap = require('gulp-sourcemaps');
const $terser = require('gulp-terser');

// const _source = require('vinyl-source-stream');
// const _buffer = require('vinyl-buffer');

const browserify = require('browserify');
const babelify = require('babelify');
const del = require('del');
const server = require('browser-sync').create();

$.task('build', $.series(clean, $.parallel(pages, assets, scripts)));
$.task('default', $.series('build', $.parallel(serve, watch)));
$.task('publish', publish);

function clean() {
  return del(['build']);
}

function reload(done) {
  server.reload();
  done();
}

function watch() {
  $.watch('docs_src/index.html', $.series(pages, reload));
  // $.watch('src/css/*.css', $.series(styles, reload));
  $.watch('docs_src/models/*', $.series(assets, reload));
  $.watch('docs_src/js/*', $.series(scripts, reload));
}

function serve(done) {
  server.init({server: 'build'});
  done();
}

function pages() {
  return $.src(['docs_src/index.html'])
    .pipe($changed('build'))
    .pipe($plumber())
    .pipe($htmlmin({
      removeComments: true,
      collapseWhitespace: true,
      removeEmptyAttributes: true,
      minifyJS: true,
      minifyCSS: true}))
    .pipe($.dest('build'));
}

// function styles() {
//   return $.src('src/css/*.css')
//     .pipe($changed('build'))
//     .pipe($plumber())
//     .pipe($postcss([
//       require('precss'),
//       require('cssnano')({
//         autoprefixer: {browsers: ['last 2 version'], add: true},
//         discardComments: {removeAll: true}})]))
//     .pipe($.dest('build'));
// }

function assets() {

  return $.src('docs_src/models/*')
    .pipe($changed('build'))
    .pipe($.dest('build/models'));
}

function scripts() {
  return $.src(['docs_src/js/jquery.min.js',
                'docs_src/js/three.min.js',
                'docs_src/js/TrackballControls.js',
                'docs_src/js/MTLLoader.js',
                'docs_src/js/dat.gui.min.js',
                'docs_src/js/LoaderSupport.js',
                'docs_src/js/OBJLoader2.js',
                'docs_src/js/tf.min.js',
                'docs_src/js/mobilenet.js',
                'docs_src/js/app.js',])
    .pipe($.dest('./build/js'));
}

function publish() {
  return $.src('./build/**/*').pipe($.dest('./'));
}
