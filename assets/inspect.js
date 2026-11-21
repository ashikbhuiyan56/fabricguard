// In-browser defect inspection with ONNX Runtime Web.
// Mirrors app/app.py: input checks -> grayscale -> resize -> ImageNet normalisation -> model -> CAM -> decision.
(function () {
  var D = window.FG;
  var ORT_VERSION = "1.20.1";
  var TAU = 0.8;            // same default threshold as the Streamlit app
  var MAX_MB = 15, MIN_SIDE = 64;
  var MEAN = [0.485, 0.456, 0.406], STD = [0.229, 0.224, 0.225];

  var $ = function (id) { return document.getElementById(id); };
  var view = $("view"), ctx = view.getContext("2d");
  var session = null, meta = null, last = null;

  // ---------- model ----------
  ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@" + ORT_VERSION + "/dist/";
  if (!self.crossOriginIsolated) ort.env.wasm.numThreads = 1;

  fetch("model/meta.json")
    .then(function (r) { if (!r.ok) throw new Error("no model"); return r.json(); })
    .then(function (m) { meta = m; return ort.InferenceSession.create("model/fabricguard.onnx"); })
    .then(function (s) {
      session = s;
      var m = meta.metrics || {}, f = function (s) { return m[s] ? Math.round(m[s].f1_macro * 100) + "%" : "n/a"; };
      $("modelStatus").textContent = "Model ready. Choose a photo to start.";
      $("modelInfo").textContent = "Demo model: EfficientNet-B0, macro-F1 " + f("val") + " on validation and " +
        f("test") + " on test images it never saw in training.";
      if (pending) { var p = pending; pending = null; run(p.src, p.truth); }
    })
    .catch(function (e) {
      console.error(e);
      $("modelStatus").textContent = "The model could not be loaded. Please refresh the page.";
    });

  // ---------- inputs ----------
  var pending = null;
  function handleFile(file) {
    if (!file) return;
    if (!/^image\//.test(file.type)) return showError("This file is not an image.");
    if (file.size > MAX_MB * 1024 * 1024) return showError("The file is larger than " + MAX_MB + " MB.");
    createImageBitmap(file, { imageOrientation: "from-image" })
      .then(function (bmp) { start(bmp, null); })
      .catch(function () { showError("The image could not be read. It may be corrupted."); });
  }
  function start(src, truth) {
    if (!session) { pending = { src: src, truth: truth }; drawOnly(src); $("modelStatus").textContent = "Loading the model…"; return; }
    run(src, truth);
  }
  $("fileInput").addEventListener("change", function (e) { handleFile(e.target.files[0]); e.target.value = ""; });
  $("camInput").addEventListener("change", function (e) { handleFile(e.target.files[0]); e.target.value = ""; });
  var drop = $("drop");
  ["dragenter", "dragover"].forEach(function (t) { drop.addEventListener(t, function (e) { e.preventDefault(); drop.classList.add("over"); }); });
  ["dragleave", "drop"].forEach(function (t) { drop.addEventListener(t, function (e) { e.preventDefault(); drop.classList.remove("over"); }); });
  drop.addEventListener("drop", function (e) { handleFile(e.dataTransfer.files[0]); });
  $("heatToggle").addEventListener("change", function () { if (last) paint(last); });

  var row = $("sampleRow");
  D.classes.forEach(function (c) {
    var b = document.createElement("button");
    b.type = "button"; b.className = "sample"; b.title = c.name;
    b.innerHTML = '<img src="samples/' + c.key + '.jpg" alt="' + c.name + ' test image">';
    b.addEventListener("click", function () {
      var img = new Image();
      img.onload = function () { start(img, c); };
      img.src = "samples/" + c.key + ".jpg";
    });
    row.appendChild(b);
  });

  // ---------- helpers ----------
  function gray(src, w, h) {
    var c = document.createElement("canvas"); c.width = w; c.height = h;
    var g = c.getContext("2d", { willReadFrequently: true });
    g.imageSmoothingEnabled = true; g.imageSmoothingQuality = "high";
    g.drawImage(src, 0, 0, w, h);
    var d = g.getImageData(0, 0, w, h).data, out = new Float32Array(w * h);
    for (var i = 0; i < w * h; i++) out[i] = 0.299 * d[i * 4] + 0.587 * d[i * 4 + 1] + 0.114 * d[i * 4 + 2];
    return out;
  }

  // same checks and thresholds as validate() in app/app.py
  function qualityWarnings(src) {
    var n = 512, g = gray(src, n, n), sum = 0, sq = 0, i;
    for (i = 0; i < g.length; i++) { var v = Math.round(g[i]); g[i] = v; sum += v; sq += v * v; }
    var mean = sum / g.length, std = Math.sqrt(sq / g.length - mean * mean);
    var ls = 0, ls2 = 0, cnt = 0;
    for (var y = 1; y < n - 1; y++) for (var x = 1; x < n - 1; x++) {
      var k = y * n + x, lap = g[k - 1] + g[k + 1] + g[k - n] + g[k + n] - 4 * g[k];
      ls += lap; ls2 += lap * lap; cnt++;
    }
    var lapVar = ls2 / cnt - Math.pow(ls / cnt, 2), w = [];
    if (mean < 35) w.push("The image is very dark. Improve the lighting and retake it.");
    if (mean > 235) w.push("The image is over-exposed. Reduce light or glare and retake it.");
    if (lapVar < 15) w.push("The image looks blurred. Hold the camera still and refocus.");
    if (std < 5) w.push("The image has almost no texture. Is the garment in the frame?");
    return w;
  }

  function toTensor(src, size) {
    if (meta.pre_resize) {  // Colab models saw every image as 256x256 grayscale first
      var c = document.createElement("canvas"); c.width = c.height = meta.pre_resize;
      var cx = c.getContext("2d"); cx.imageSmoothingQuality = "high";
      cx.drawImage(src, 0, 0, c.width, c.height); src = c;
    }
    var g = gray(src, size, size), n = size * size, t = new Float32Array(3 * n);
    for (var i = 0; i < n; i++) {
      var v = Math.round(g[i]) / 255;
      for (var ch = 0; ch < 3; ch++) t[ch * n + i] = (v - MEAN[ch]) / STD[ch];
    }
    return new ort.Tensor("float32", t, [1, 3, size, size]);
  }

  function jet(v) {
    var r = Math.min(1, Math.max(0, 1.5 - Math.abs(4 * v - 3)));
    var g = Math.min(1, Math.max(0, 1.5 - Math.abs(4 * v - 2)));
    var b = Math.min(1, Math.max(0, 1.5 - Math.abs(4 * v - 1)));
    return [r * 255, g * 255, b * 255];
  }

  function drawOnly(src) {
    var w = src.width, h = src.height, s = 448 / Math.max(w, h);
    view.width = Math.round(w * s); view.height = Math.round(h * s);
    ctx.drawImage(src, 0, 0, view.width, view.height);
    view.hidden = false; $("dropEmpty").hidden = true;
  }

  function paint(r) {
    drawOnly(r.src);
    if (!$("heatToggle").checked || !r.cam) return;
    var W = view.width, H = view.height, img = ctx.getImageData(0, 0, W, H), d = img.data;
    var ch = r.camH, cw = r.camW, cam = r.cam;
    for (var y = 0; y < H; y++) {
      var fy = Math.min(ch - 1, Math.max(0, (y + 0.5) * ch / H - 0.5)), y0 = Math.floor(fy), y1 = Math.min(ch - 1, y0 + 1), dy = fy - y0;
      for (var x = 0; x < W; x++) {
        var fx = Math.min(cw - 1, Math.max(0, (x + 0.5) * cw / W - 0.5)), x0 = Math.floor(fx), x1 = Math.min(cw - 1, x0 + 1), dx = fx - x0;
        var v = (cam[y0 * cw + x0] * (1 - dx) + cam[y0 * cw + x1] * dx) * (1 - dy) + (cam[y1 * cw + x0] * (1 - dx) + cam[y1 * cw + x1] * dx) * dy;
        var c = jet(v), k = (y * W + x) * 4;
        // grayscale base + jet heat-map, alpha 0.45 as in src/gradcam.py
        var base = 0.299 * d[k] + 0.587 * d[k + 1] + 0.114 * d[k + 2];
        d[k] = 0.45 * c[0] + 0.55 * base; d[k + 1] = 0.45 * c[1] + 0.55 * base; d[k + 2] = 0.45 * c[2] + 0.55 * base;
      }
    }
    ctx.putImageData(img, 0, 0);
  }

  function showError(msg) {
    $("resultEmpty").hidden = true; $("resultBody").hidden = false;
    $("verdict").className = "verdict rejected";
    $("verdict").innerHTML = "<span>Rejected</span><b>" + msg + "</b>";
    $("warnings").innerHTML = ""; $("action").textContent = ""; $("probs").innerHTML = ""; $("timing").textContent = "";
  }

  // ---------- inference ----------
  function run(src, truth) {
    if (Math.min(src.width, src.height) < MIN_SIDE) {
      drawOnly(src);
      return showError("The image is too small (" + src.width + "×" + src.height + "). It needs to be at least " + MIN_SIDE + "×" + MIN_SIDE + ".");
    }
    var warns = qualityWarnings(src);
    var t0 = performance.now();
    var feeds = {}; feeds[session.inputNames[0]] = toTensor(src, meta.img_size);
    session.run(feeds).then(function (out) {
      var ms = performance.now() - t0;
      var probs = Array.from(out.probs.data);
      var top = probs.indexOf(Math.max.apply(null, probs)), conf = probs[top];
      var dims = out.cams.dims, camH = dims[2], camW = dims[3], n = camH * camW;
      var cam = Array.from(out.cams.data.slice(top * n, (top + 1) * n)).map(function (v) { return Math.max(0, v); });
      var mx = Math.max.apply(null, cam) || 1;
      cam = cam.map(function (v) { return v / mx; });
      last = { src: src, cam: cam, camH: camH, camW: camW };
      paint(last);
      render(probs, top, conf, warns, ms, truth);
    }).catch(function (e) { console.error(e); showError("Something went wrong while checking this image."); });
  }

  function render(probs, top, conf, warns, ms, truth) {
    var key = meta.classes[top];
    var cls = D.classes.filter(function (c) { return c.key === key; })[0];
    var auto = conf >= TAU && warns.length === 0;
    $("resultEmpty").hidden = true; $("resultBody").hidden = false;
    var v = $("verdict");
    v.className = "verdict " + (auto ? "auto" : "review");
    v.innerHTML = "<span>" + (auto ? "Auto decision" : "Manual review") + "</span>" +
      "<b>" + (auto ? "" : "Likely ") + cls.name + "</b><em>" + Math.round(conf * 100) + "% confident</em>" +
      (truth ? '<small>Correct answer: ' + truth.name + (truth.key === key ? " ✓" : " ✗") + "</small>" : "");
    var reason = !auto && !warns.length ? ["Confidence is below " + Math.round(TAU * 100) + "%, so an inspector should check this piece."] : [];
    $("warnings").innerHTML = warns.concat(reason).map(function (w) { return '<p class="warn">' + w + "</p>"; }).join("");
    $("action").innerHTML = "<b>Suggested action:</b> " + cls.action;
    var order = probs.map(function (p, i) { return i; }).sort(function (a, b) { return probs[b] - probs[a]; });
    $("probs").innerHTML = order.map(function (i) {
      var c = D.classes.filter(function (x) { return x.key === meta.classes[i]; })[0];
      return '<div class="prob' + (i === top ? " top" : "") + '"><span>' + c.name + "</span><div><i style=\"width:" +
        (probs[i] * 100).toFixed(1) + '%"></i></div><b>' + (probs[i] * 100).toFixed(1) + "%</b></div>";
    }).join("");
    $("timing").textContent = "Checked in " + Math.round(ms) + " ms on this device.";
  }
})();
