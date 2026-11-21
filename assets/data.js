// All numbers come from the project's own outputs (outputs/eda/*.csv and report/milestone4.md).
// Update this file when Milestone 5 results are ready.
window.FG = {
  classes: [
    { key: "broken_button", name: "Broken Button", source: "Button attaching machine, button supplier lot",
      action: "Reject piece → send to button re-attachment; check button supplier lot / pressing machine." },
    { key: "button_hike", name: "Button Hike", source: "Button-sewing thread tension, needle",
      action: "Send to rework (re-sew button); check button-sewing machine thread tension and needle." },
    { key: "color_defect", name: "Color Defect", source: "Dyeing / finishing, oil or dirt in handling",
      action: "Hold the fabric lot; check dyeing shade / batch mix-up before cutting more panels." },
    { key: "foreign_yarn", name: "Foreign Yarn", source: "Knitting / weaving, yarn contamination",
      action: "Mark and remove contaminated yarn if possible; report to fabric supplier / knitting section." },
    { key: "hole", name: "Hole", source: "Knitting needles, cutting, sharp handling points",
      action: "Reject or repair panel; inspect knitting needles, cutting table and handling for sharp points." },
    { key: "sewing_error", name: "Sewing Error", source: "Operator or machine settings",
      action: "Return to the sewing line operator for rework; check stitch settings and operator training." }
  ],

  // outputs/eda/split_counts.csv
  splits: {
    broken_button: { train: 381, val: 82, test: 82 },
    button_hike:   { train: 214, val: 47, test: 46 },
    color_defect:  { train: 268, val: 58, test: 58 },
    foreign_yarn:  { train: 233, val: 51, test: 50 },
    hole:          { train: 390, val: 84, test: 82 },
    sewing_error:  { train: 217, val: 50, test: 50 }
  },

  // report/milestone4.md, section 4 (validation set)
  models: [
    { id: "M1", name: "SVM (baseline)",   acc: 0.481, prec: 0.478, rec: 0.423, f1: 0.399, ms: 3.54, msNote: "†", params: null,  train: 3 },
    { id: "M2", name: "Small CNN",        acc: 0.608, prec: 0.597, rec: 0.566, f1: 0.558, ms: 1.45, params: 1.18, train: 387 },
    { id: "M3", name: "ResNet-18",        acc: 0.860, prec: 0.881, rec: 0.852, f1: 0.854, ms: 2.93, params: 11.18, train: 192 },
    { id: "M4", name: "EfficientNet-B0",  acc: 0.871, prec: 0.873, rec: 0.869, f1: 0.862, ms: 8.67, params: 4.02, train: 171 }
  ],

  // report/milestone4.md, section 5 (per-class recall, validation)
  recall: {
    broken_button: [0.77, 0.94, 1.00, 1.00],
    button_hike:   [0.09, 0.30, 1.00, 1.00],
    color_defect:  [0.59, 0.66, 0.86, 0.93],
    foreign_yarn:  [0.41, 0.73, 0.49, 0.51],
    hole:          [0.67, 0.62, 0.82, 0.77],
    sewing_error:  [0.02, 0.16, 0.94, 1.00]
  },

  // report/team.json
  team: [
    { name: "Abir Kamorkar",            id: "242310005101004" },
    { name: "Ashik Bhuiyan",            id: "242310005101103" },
    { name: "Tanjila Tasnim Punna",     id: "242310005101513" },
    { name: "Mehedi Islam Miazi Sami",  id: "242310005101582" },
    { name: "Sabbir Hossien Hamim",     id: "242310005101803" }
  ],

  // Illustrative probability vectors for the decision-rule demo (not model output).
  simSamples: [
    { label: "Clear sewing error",        probs: [0.01, 0.02, 0.02, 0.02, 0.03, 0.90] },
    { label: "Hole vs color defect",      probs: [0.02, 0.01, 0.31, 0.08, 0.55, 0.03] },
    { label: "Ambiguous foreign yarn",    probs: [0.02, 0.02, 0.18, 0.44, 0.31, 0.03] },
    { label: "Confident broken button",   probs: [0.96, 0.02, 0.00, 0.00, 0.01, 0.01] }
  ]
};
