(function () {
  var D = window.FG;

  // mobile menu
  var btn = document.getElementById("menuBtn");
  var nav = document.getElementById("nav");
  btn.addEventListener("click", function () {
    var open = nav.classList.toggle("open");
    btn.setAttribute("aria-expanded", open);
  });
  nav.querySelectorAll("a").forEach(function (a) {
    a.addEventListener("click", function () {
      nav.classList.remove("open");
      btn.setAttribute("aria-expanded", "false");
    });
  });

  // defects
  var short = {
    broken_button: "Cracked or chipped button on a placket or cuff.",
    button_hike: "Button sewn loosely, tilted or out of position.",
    color_defect: "Dark stains and blotches on the fabric.",
    foreign_yarn: "A different yarn running across the fabric.",
    hole: "Holes and tears in the panel.",
    sewing_error: "Skipped, broken or uneven stitches."
  };
  var grid = document.getElementById("defectGrid");
  D.classes.forEach(function (c) {
    var d = document.createElement("div");
    d.className = "defect";
    d.innerHTML = '<img src="assets/img/cls_' + c.key + '.jpg" alt="' + c.name + '" loading="lazy">' +
      "<h3>" + c.name + "</h3><p>" + short[c.key] + "</p>";
    grid.appendChild(d);
  });

  // results bars (validation macro-F1)
  var chart = document.getElementById("metricChart");
  var best = Math.max.apply(null, D.models.map(function (m) { return m.f1; }));
  D.models.slice().reverse().forEach(function (m) {
    var row = document.createElement("div");
    row.innerHTML = '<div class="bar-label"><span>' + m.name.replace(" (baseline)", " baseline") + "</span><b>" +
      Math.round(m.f1 * 100) + '%</b></div><div class="bar' + (m.f1 === best ? " top" : "") +
      '"><i data-w="' + m.f1 * 100 + '"></i></div>';
    chart.appendChild(row);
  });
  var note = document.createElement("p");
  note.className = "results-note";
  note.textContent = "Macro-F1 on the validation set (372 images).";
  chart.appendChild(note);

  var bars = chart.querySelectorAll(".bar i");
  var fill = function () { bars.forEach(function (b) { b.style.width = b.dataset.w + "%"; }); };
  if ("IntersectionObserver" in window) {
    var io = new IntersectionObserver(function (e) {
      if (e[0].isIntersecting) { fill(); io.disconnect(); }
    }, { threshold: 0.4 });
    io.observe(chart);
  } else { fill(); }

  // team
  var team = document.getElementById("teamGrid");
  D.team.forEach(function (p) {
    var li = document.createElement("li");
    li.innerHTML = "<b>" + p.name + "</b><span>" + p.id + "</span>";
    team.appendChild(li);
  });
})();
