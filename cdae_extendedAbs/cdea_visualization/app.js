/**
 * CDEA-Contrastive Method — Step-by-Step Interactive JavaScript Engine
 * Implementation matching PyTorch codebase & ICAIIS 2026 Extended Abstract
 */

(function () {
  'use strict';

  // ---------------------------------------------------------------------------
  // State Management
  // ---------------------------------------------------------------------------
  const GRID_SIZE = 7;
  const NUM_TILES = 49;
  const CANVAS_SIZE = 224;

  const state = {
    currentStep: 1,
    activeSubstep: '5.1',
    scenario: 'cat_dog', // 'cat_dog', 'ham10000', 'stanford_dogs'
    autoPlaying: false,
    autoPlayTimer: null,
    
    // Hyperparameters
    lambdaSuff: 1.0,
    lambdaMargin: 1.0,
    lambdaOverlap: 0.20,
    lambdaSparse: 0.05,
    lr: 0.5,
    
    // Optimization state & trajectory history
    iteration: 0,
    maxIterations: 50,
    lossHistory: [],
    trajectoryHistory: [], // [ { iter, logitsC1, logitsC2, maskC1, maskC2, metrics } ]
    
    // Grid evidence data (49 floats each)
    evidenceC1: new Float32Array(NUM_TILES),
    evidenceC2: new Float32Array(NUM_TILES),
    
    // Mask Logits z_k (49 floats each)
    logitsC1: new Float32Array(NUM_TILES),
    logitsC2: new Float32Array(NUM_TILES),
    
    // Sigmoid Masks M_k (49 floats each)
    maskC1: new Float32Array(NUM_TILES),
    maskC2: new Float32Array(NUM_TILES),
    
    // Selected Cell
    selectedCell: null
  };

  // ---------------------------------------------------------------------------
  // Color Map Utilities (Jet, High-Contrast Amber, High-Contrast Magenta, Collision)
  // ---------------------------------------------------------------------------
  function jetColor(val) {
    val = Math.max(0, Math.min(1, val));
    const r = Math.max(0, Math.min(255, Math.floor(255 * (1.5 - Math.abs(val * 4 - 3)))));
    const g = Math.max(0, Math.min(255, Math.floor(255 * (1.5 - Math.abs(val * 4 - 2)))));
    const b = Math.max(0, Math.min(255, Math.floor(255 * (1.5 - Math.abs(val * 4 - 1)))));
    return `rgb(${r},${g},${b})`;
  }

  function amberColor(val) {
    val = Math.max(0, Math.min(1, val));
    // Warm Amber/Gold/Orange Palette (Class 1 Mask M1)
    const r = Math.floor(15 + val * 239);
    const g = Math.floor(23 + val * 190);
    const b = Math.floor(42 * (1 - val) + 20 * val);
    return `rgb(${r},${g},${b})`;
  }

  function magentaColor(val) {
    val = Math.max(0, Math.min(1, val));
    // Cool Magenta/Purple/Pink Palette (Class 2 Mask M2)
    const r = Math.floor(15 + val * 229);
    const g = Math.floor(23 + val * 60);
    const b = Math.floor(42 + val * 210);
    return `rgb(${r},${g},${b})`;
  }

  function collisionColor(val) {
    val = Math.max(0, Math.min(1, val));
    // Vivid Neon Lime/Yellow Collision Palette (Overlap)
    const r = Math.floor(15 + val * 234);
    const g = Math.floor(23 + val * 235);
    const b = Math.floor(42 * (1 - val));
    return `rgb(${r},${g},${b})`;
  }

  function sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
  }

  function logit(p) {
    p = Math.max(1e-5, Math.min(1 - 1e-5, p));
    return Math.log(p / (1 - p));
  }

  function normalizeMap(arr) {
    let maxVal = 0;
    for (let i = 0; i < arr.length; i++) {
      if (arr[i] > maxVal) maxVal = arr[i];
    }
    const out = new Float32Array(arr.length);
    for (let i = 0; i < arr.length; i++) {
      out[i] = maxVal > 0 ? arr[i] / maxVal : 0;
    }
    return out;
  }

  function multiplyMaps(arr1, arr2) {
    const out = new Float32Array(arr1.length);
    for (let i = 0; i < arr1.length; i++) {
      out[i] = arr1[i] * arr2[i];
    }
    return out;
  }

  // ---------------------------------------------------------------------------
  // Synthetic Data Generators & Precomputed Optimization Trajectory
  // ---------------------------------------------------------------------------
  function initScenarioData() {
    state.iteration = 0;
    state.lossHistory = [];
    state.trajectoryHistory = [];

    for (let i = 0; i < NUM_TILES; i++) {
      const row = Math.floor(i / GRID_SIZE);
      const col = i % GRID_SIZE;

      if (state.scenario === 'cat_dog') {
        const distCat = Math.hypot(row - 2.5, col - 2.0);
        const distDog = Math.hypot(row - 2.5, col - 4.5);
        const distBedding = Math.hypot(row - 5.5, col - 4.5);
        const animalBody = Math.exp(-0.4 * Math.hypot(row - 3.0, col - 3.5));
        
        state.evidenceC1[i] = 0.7 * Math.exp(-0.5 * distCat) + 0.5 * animalBody;
        state.evidenceC2[i] = 0.4 * Math.exp(-0.5 * distDog) + 0.6 * animalBody + 0.5 * Math.exp(-0.5 * distBedding);
      } 
      else if (state.scenario === 'ham10000') {
        const distCenter = Math.hypot(row - 3, col - 3);
        const distBorder = Math.hypot(row - 2, col - 1);
        
        state.evidenceC1[i] = Math.exp(-0.6 * distCenter) + 0.4 * Math.exp(-0.5 * distBorder);
        state.evidenceC2[i] = Math.exp(-0.6 * distCenter);
      }
      else {
        const dist1 = Math.hypot(row - 2, col - 2);
        const dist2 = Math.hypot(row - 4, col - 4);
        state.evidenceC1[i] = 0.8 * Math.exp(-0.4 * dist1) + 0.5 * Math.exp(-0.3 * dist2);
        state.evidenceC2[i] = 0.7 * Math.exp(-0.4 * dist1) + 0.6 * Math.exp(-0.3 * dist2);
      }
    }

    // Normalize Base Evidence
    let sum1 = 0, sum2 = 0;
    for (let i = 0; i < NUM_TILES; i++) {
      sum1 += state.evidenceC1[i];
      sum2 += state.evidenceC2[i];
    }
    for (let i = 0; i < NUM_TILES; i++) {
      state.evidenceC1[i] /= (sum1 || 1);
      state.evidenceC2[i] /= (sum2 || 1);

      state.logitsC1[i] = Math.max(-6, Math.min(6, logit(state.evidenceC1[i] * 3)));
      state.logitsC2[i] = Math.max(-6, Math.min(6, logit(state.evidenceC2[i] * 3)));

      state.maskC1[i] = sigmoid(state.logitsC1[i]);
      state.maskC2[i] = sigmoid(state.logitsC2[i]);
    }

    // Precompute Full 50-Step Optimization Trajectory
    precomputeTrajectory();
  }

  function precomputeTrajectory() {
    // Clone initial state
    const tempLogitsC1 = new Float32Array(state.logitsC1);
    const tempLogitsC2 = new Float32Array(state.logitsC2);
    const tempMaskC1 = new Float32Array(state.maskC1);
    const tempMaskC2 = new Float32Array(state.maskC2);

    state.trajectoryHistory = [];

    for (let iter = 0; iter <= state.maxIterations; iter++) {
      // Record state
      let suff1 = 0, suff2 = 0, overlap = 0, sparse1 = 0, sparse2 = 0;
      for (let i = 0; i < NUM_TILES; i++) {
        const m1 = tempMaskC1[i];
        const m2 = tempMaskC2[i];
        suff1 += m1 * state.evidenceC1[i] * 2.0;
        suff2 += m2 * state.evidenceC2[i] * 2.0;
        overlap += m1 * m2;
        sparse1 += m1;
        sparse2 += m2;
      }
      const meanSuff = (suff1 + suff2) / 2;
      const margin = Math.max(-0.2, (suff1 - 0.7 * suff2) + (1.0 - overlap * 0.15));
      const loss = -(state.lambdaSuff * meanSuff + state.lambdaMargin * margin)
                 + state.lambdaOverlap * overlap
                 + state.lambdaSparse * (sparse1 + sparse2) / (2 * NUM_TILES);

      const metrics = { suff: suff1, margin: margin, overlap: overlap, sparse: (sparse1 + sparse2) / 2, loss: loss };

      state.trajectoryHistory.push({
        iter: iter,
        logitsC1: new Float32Array(tempLogitsC1),
        logitsC2: new Float32Array(tempLogitsC2),
        maskC1: new Float32Array(tempMaskC1),
        maskC2: new Float32Array(tempMaskC2),
        metrics: metrics
      });

      // Update logits for next iteration step
      if (iter < state.maxIterations) {
        for (let i = 0; i < NUM_TILES; i++) {
          const row = Math.floor(i / GRID_SIZE);
          const col = i % GRID_SIZE;
          const m1 = tempMaskC1[i];
          const m2 = tempMaskC2[i];

          let dL_dm1 = 0, dL_dm2 = 0;
          if (state.scenario === 'cat_dog') {
            const isKitten = (col <= 3 && row <= 4);
            const isBedding = (row >= 4 && col >= 3);
            dL_dm1 = -state.lambdaSuff * (isKitten ? 1.2 : -0.5) + state.lambdaOverlap * m2 + state.lambdaSparse * 0.1;
            dL_dm2 = -state.lambdaSuff * (isBedding ? 1.4 : -0.8) + state.lambdaOverlap * m1 + state.lambdaSparse * 0.1;
          } else {
            dL_dm1 = -state.lambdaSuff * state.evidenceC1[i] + state.lambdaOverlap * m2 + state.lambdaSparse * 0.05;
            dL_dm2 = -state.lambdaSuff * state.evidenceC2[i] + state.lambdaOverlap * m1 + state.lambdaSparse * 0.05;
          }

          const sigGrad1 = m1 * (1 - m1);
          const sigGrad2 = m2 * (1 - m2);

          tempLogitsC1[i] -= state.lr * dL_dm1 * sigGrad1;
          tempLogitsC2[i] -= state.lr * dL_dm2 * sigGrad2;

          tempLogitsC1[i] = Math.max(-6, Math.min(6, tempLogitsC1[i]));
          tempLogitsC2[i] = Math.max(-6, Math.min(6, tempLogitsC2[i]));

          tempMaskC1[i] = sigmoid(tempLogitsC1[i]);
          tempMaskC2[i] = sigmoid(tempLogitsC2[i]);
        }
      }
    }

    // Set active iteration to current state.iteration
    loadTrajectoryIteration(state.iteration);
  }

  function loadTrajectoryIteration(iter) {
    iter = Math.max(0, Math.min(state.maxIterations, iter));
    state.iteration = iter;

    const snap = state.trajectoryHistory[iter];
    if (snap) {
      state.logitsC1.set(snap.logitsC1);
      state.logitsC2.set(snap.logitsC2);
      state.maskC1.set(snap.maskC1);
      state.maskC2.set(snap.maskC2);

      state.lossHistory = state.trajectoryHistory.slice(0, iter + 1).map(s => s.metrics);
    }

    updateUI();
  }

  function computeInterventionalState() {
    const snap = state.trajectoryHistory[state.iteration];
    return snap ? snap.metrics : { suff: 0.71, margin: 0.06, overlap: 0.84, sparse: 14.2, loss: 0.5 };
  }

  // ---------------------------------------------------------------------------
  // Canvas Rendering (Step 1 Image vs Step 2 Grad-CAM vs Step 3 Masks)
  // ---------------------------------------------------------------------------
  function drawRawImage(ctx, label = null) {
    const width = ctx.canvas.width;
    const height = ctx.canvas.height;
    
    ctx.fillStyle = '#0f172a';
    ctx.fillRect(0, 0, width, height);

    if (state.scenario === 'cat_dog') {
      // Kitten on left
      ctx.fillStyle = '#f59e0b';
      ctx.beginPath(); ctx.arc(70, 90, 45, 0, Math.PI * 2); ctx.fill();
      ctx.beginPath(); ctx.moveTo(40, 50); ctx.lineTo(60, 20); ctx.lineTo(75, 50); ctx.fill();

      // Dog on right
      ctx.fillStyle = '#8b5cf6';
      ctx.beginPath(); ctx.arc(150, 90, 50, 0, Math.PI * 2); ctx.fill();

      // Pink bedding at bottom
      ctx.fillStyle = '#ec4899';
      ctx.fillRect(80, 150, 140, 60);
    } else if (state.scenario === 'ham10000') {
      ctx.fillStyle = '#e2e8f0'; ctx.fillRect(0, 0, width, height);
      ctx.fillStyle = '#334155'; ctx.beginPath(); ctx.ellipse(112, 112, 60, 45, Math.PI / 6, 0, Math.PI * 2); ctx.fill();
    } else {
      ctx.fillStyle = '#3b82f6'; ctx.beginPath(); ctx.arc(112, 112, 70, 0, Math.PI * 2); ctx.fill();
    }

    if (label) {
      ctx.fillStyle = 'rgba(15, 23, 42, 0.75)';
      ctx.fillRect(5, 5, width - 10, 26);
      ctx.fillStyle = '#38bdf8';
      ctx.font = 'bold 11px var(--font-sans)';
      ctx.fillText(label, 12, 22);
    }
  }

  function renderViewPanels() {
    const cImg = document.getElementById('canvas-img');
    const c1 = document.getElementById('canvas-c1');
    const c2 = document.getElementById('canvas-c2');
    const cOverlap = document.getElementById('canvas-overlap');

    if (!cImg || !c1 || !c2 || !cOverlap) return;

    const ctxImg = cImg.getContext('2d');
    const ctx1 = c1.getContext('2d');
    const ctx2 = c2.getContext('2d');
    const ctxOverlap = cOverlap.getContext('2d');

    // Panel 1: Image or Blur Keep Intervention
    drawRawImage(ctxImg);

    if (state.currentStep === 4 || state.currentStep === 5 || state.currentStep === 6) {
      const imgData = ctxImg.getImageData(0, 0, CANVAS_SIZE, CANVAS_SIZE);
      const data = imgData.data;
      const cellSize = CANVAS_SIZE / GRID_SIZE;

      for (let y = 0; y < CANVAS_SIZE; y++) {
        for (let x = 0; x < CANVAS_SIZE; x++) {
          const col = Math.floor(x / cellSize);
          const row = Math.floor(y / cellSize);
          const idx = row * GRID_SIZE + col;
          const maskVal = state.currentStep === 6 ? state.maskC1[idx] : 0.5;

          const p = (y * CANVAS_SIZE + x) * 4;
          if (maskVal < 0.3) {
            data[p] = Math.floor(data[p] * 0.4);
            data[p + 1] = Math.floor(data[p + 1] * 0.4);
            data[p + 2] = Math.floor(data[p + 2] * 0.4);
          }
        }
      }
      ctxImg.putImageData(imgData, 0, 0);
    }

    // Step 1 vs Step 2 vs Step 3 Custom Render Logic
    if (state.currentStep === 1) {
      // Step 1: Show Raw Input Image across panels with candidate class badges (No Grad-CAM yet!)
      drawRawImage(ctx1, "Hypothesis 1: Cat (Logit +0.710, 51%)");
      drawRawImage(ctx2, "Hypothesis 2: Dog (Logit +0.650, 49%)");
      drawRawImage(ctxOverlap, "Top-m Rivalry: Cat vs Dog");
    } 
    else if (state.currentStep === 2) {
      // Step 2: Show Base Grad-CAM Evidence Maps with Jet Palette
      drawHeatmap(ctx1, normalizeMap(state.evidenceC1), jetColor);
      drawHeatmap(ctx2, normalizeMap(state.evidenceC2), jetColor);
      drawHeatmap(ctxOverlap, multiplyMaps(normalizeMap(state.evidenceC1), normalizeMap(state.evidenceC2)), jetColor);
    } 
    else if (state.currentStep === 3) {
      // Step 3: Show High-Contrast Masks (Class 1 Amber/Gold, Class 2 Magenta/Purple)
      drawHeatmap(ctx1, state.maskC1, amberColor);
      drawHeatmap(ctx2, state.maskC2, magentaColor);
      drawHeatmap(ctxOverlap, multiplyMaps(state.maskC1, state.maskC2), collisionColor);
    } 
    else if (state.currentStep === 4) {
      drawHeatmap(ctx1, state.maskC1, amberColor);
      drawHeatmap(ctx2, state.maskC2, magentaColor);
      drawHeatmap(ctxOverlap, multiplyMaps(state.maskC1, state.maskC2), collisionColor);
    } 
    else {
      // Step 5 & 6: Optimized Disjoint Masks
      drawHeatmap(ctx1, state.maskC1, amberColor);
      drawHeatmap(ctx2, state.maskC2, magentaColor);
      drawHeatmap(ctxOverlap, multiplyMaps(state.maskC1, state.maskC2), collisionColor);
    }

    // Update legend range label text based on current step
    const legendLabel = document.getElementById('legend-scale-title');
    if (legendLabel) {
      if (state.currentStep === 1) {
        legendLabel.innerText = "Step 1: Raw Input Image & Top-m Candidate Classes";
      } else if (state.currentStep === 2) {
        legendLabel.innerText = "Step 2: Base Grad-CAM Evidence Heatmaps (Jet Palette, Peak = 1.0)";
      } else if (state.currentStep === 3) {
        legendLabel.innerText = "Step 3: Initial Sigmoid Masks M_k = σ(z_k) (Amber: Cat, Magenta: Dog)";
      } else if (state.currentStep >= 5) {
        legendLabel.innerText = "Optimized Disjoint Evidence Mask Values M_k ∈ [0, 1]";
      } else {
        legendLabel.innerText = "Normalized Relative Intensity / Mask Value";
      }
    }
  }

  function drawHeatmap(ctx, values, colorFunc = jetColor) {
    const width = ctx.canvas.width;
    const height = ctx.canvas.height;
    const cellSize = width / GRID_SIZE;

    ctx.clearRect(0, 0, width, height);

    for (let row = 0; row < GRID_SIZE; row++) {
      for (let col = 0; col < GRID_SIZE; col++) {
        const idx = row * GRID_SIZE + col;
        const val = Math.max(0, Math.min(1, values[idx]));

        ctx.fillStyle = colorFunc(val);
        ctx.fillRect(col * cellSize, row * cellSize, cellSize, cellSize);

        ctx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
        ctx.strokeRect(col * cellSize, row * cellSize, cellSize, cellSize);
      }
    }
  }

  function renderLossChart() {
    const canvas = document.getElementById('canvas-loss-chart');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;

    ctx.clearRect(0, 0, w, h);

    if (state.lossHistory.length < 2) {
      ctx.fillStyle = '#64748b';
      ctx.font = '12px var(--font-sans)';
      ctx.fillText('Scrub iteration slider to view live loss trajectory', 20, 60);
      return;
    }

    const n = state.lossHistory.length;
    const dx = w / (state.maxIterations || 50);

    // Loss Line (Rose)
    ctx.strokeStyle = '#f43f5e'; ctx.lineWidth = 2; ctx.beginPath();
    for (let i = 0; i < n; i++) {
      const x = i * dx;
      const y = h - Math.max(0, Math.min(h, (state.lossHistory[i].loss + 2) * 20));
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Margin Line (Emerald)
    ctx.strokeStyle = '#10b981'; ctx.beginPath();
    for (let i = 0; i < n; i++) {
      const x = i * dx;
      const y = h - Math.max(0, Math.min(h, (state.lossHistory[i].margin + 0.5) * 60));
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Overlap Line (Amber)
    ctx.strokeStyle = '#f59e0b'; ctx.beginPath();
    for (let i = 0; i < n; i++) {
      const x = i * dx;
      const y = h - Math.max(0, Math.min(h, state.lossHistory[i].overlap * 12));
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  // ---------------------------------------------------------------------------
  // Grid Overlay Generators & Event Handlers
  // ---------------------------------------------------------------------------
  function buildGridOverlays() {
    const overlays = [
      document.getElementById('grid-overlay-img'),
      document.getElementById('grid-overlay-c1'),
      document.getElementById('grid-overlay-c2'),
      document.getElementById('grid-overlay-overlap')
    ];

    overlays.forEach(overlay => {
      if (!overlay) return;
      overlay.innerHTML = '';
      for (let i = 0; i < NUM_TILES; i++) {
        const cell = document.createElement('div');
        cell.className = 'grid-cell';
        cell.dataset.idx = i;
        cell.addEventListener('mouseenter', () => selectTile(i));
        cell.addEventListener('click', () => selectTile(i));
        overlay.appendChild(cell);
      }
    });
  }

  function selectTile(idx) {
    state.selectedCell = idx;
    const row = Math.floor(idx / GRID_SIZE);
    const col = idx % GRID_SIZE;

    const tileInfo = document.getElementById('tile-info');
    if (tileInfo) {
      tileInfo.innerHTML = `
        <strong>Tile #${idx} (Row ${row}, Col ${col})</strong><br>
        Base Evid (Cat): ${state.evidenceC1[idx].toFixed(4)} | Dog: ${state.evidenceC2[idx].toFixed(4)}<br>
        Logits z_k (Cat): ${state.logitsC1[idx].toFixed(3)} | Dog: ${state.logitsC2[idx].toFixed(3)}<br>
        Sigmoid Mask M_k (Cat): ${state.maskC1[idx].toFixed(3)} | Dog: ${state.maskC2[idx].toFixed(3)}<br>
        Overlap (M1 · M2): ${(state.maskC1[idx] * state.maskC2[idx]).toFixed(4)}
      `;
    }

    document.querySelectorAll('.grid-cell').forEach(c => {
      if (parseInt(c.dataset.idx) === idx) {
        c.classList.add('selected');
      } else {
        c.classList.remove('selected');
      }
    });
  }

  // ---------------------------------------------------------------------------
  // UI & Step Updates + Substep Inspector
  // ---------------------------------------------------------------------------
  const stepConfigs = {
    1: {
      title: "Step 1: Input Image & Rival Hypotheses Selection",
      badge: "Phase 1 / 6",
      desc: "A trained image classifier (frozen ResNet-18) receives the input image. Instead of evaluating classes independently, CDEA identifies top competing rival predicted classes (Cat 51% vs Dog 49%) that will jointly claim spatial evidence.",
      math: "f_\\theta(X) \\to \\text{Top-}K \\text{ logits } \\{z_{\\text{cat}} = +0.71, z_{\\text{dog}} = +0.65\\}",
      p1Tag: "Raw Image X", p2Tag: "Hypothesis 1: Cat (51%)", p3Tag: "Hypothesis 2: Dog (49%)", p4Tag: "Hypotheses Rivalry",
      p1Desc: "Original input image fed to frozen classifier",
      p2Desc: "Target Class 1: Cat hypothesis",
      p3Desc: "Target Class 2: Dog hypothesis",
      p4Desc: "Rival predicted classes competing for evidence"
    },
    2: {
      title: "Step 2: Base Evidence Extraction (Post-Hoc Attribution)",
      badge: "Phase 2 / 6",
      desc: "Extract raw non-negative attribution maps via Grad-CAM or Integrated Gradients from the final conv layer, coarsened onto a 7×7 grid. Notice how standard maps point to the SAME animal body for both classes!",
      math: "E_k = \\text{ReLU}\\left( \\sum_c \\alpha_c^k A_c \\right) \\in \\mathbb{R}^{49}_{\\ge 0}, \\quad \\sum_r E_{k,r} = 1.0",
      p1Tag: "Raw Image X", p2Tag: "Cat Grad-CAM (7x7)", p3Tag: "Dog Grad-CAM (7x7)", p4Tag: "Raw Overlap (85%)",
      p1Desc: "Input image",
      p2Desc: "Grad-CAM map for Cat (Jet palette)",
      p3Desc: "Grad-CAM map for Dog (Jet palette)",
      p4Desc: "Spatial overlap: 85% of tiles claimed by both!"
    },
    3: {
      title: "Step 3: Mask Parameterization & Initialization",
      badge: "Phase 3 / 6",
      desc: "Masks M_k are parameterized by unconstrained logits z_k mapped through sigmoid M_k = σ(z_k). Logits are initialized from base evidence. High-contrast colors distinguish Class 1 (Golden-Amber) from Class 2 (Magenta-Purple).",
      math: "M_k = \\sigma(\\mathbf{z}_k) = \\frac{1}{1 + e^{-\\mathbf{z}_k}}, \\quad \\mathbf{z}_k^{(0)} = \\text{logit}(\\text{clamp}(E_k, \\epsilon, 1-\\epsilon))",
      p1Tag: "Baseline", p2Tag: "Cat Mask M1 (Amber)", p3Tag: "Dog Mask M2 (Magenta)", p4Tag: "Initial Overlap",
      p1Desc: "Unconstrained logit parameter space",
      p2Desc: "Class 1 Cat Mask (Warm Amber spectrum)",
      p3Desc: "Class 2 Dog Mask (Cool Magenta spectrum)",
      p4Desc: "Initial mask collision region"
    },
    4: {
      title: "Step 4: Interventional Evaluation ('Covering Up')",
      badge: "Phase 4 / 6",
      desc: "To test if a mask allocation is valid, CDEA performs keep interventions: keep regions where M_k is high and replace the rest with an average blur baseline. The masked image is re-evaluated by the frozen classifier.",
      math: "X_{\\text{keep}, k} = M_k^{\\text{pixel}} \\odot X + (1 - M_k^{\\text{pixel}}) \\odot \\text{blur}(X) \\implies \\text{Forward pass } f_\\theta(X_{\\text{keep}, k})",
      p1Tag: "Masked Blur X", p2Tag: "Keep Interv C1", p3Tag: "Keep Interv C2", p4Tag: "Blurred Baseline",
      p1Desc: "Masked image fed back through frozen network",
      p2Desc: "Tiles kept for Cat hypothesis",
      p3Desc: "Tiles kept for Dog hypothesis",
      p4Desc: "Average pool blur baseline image"
    },
    5: {
      title: "Step 5: Penalized Joint Optimization & Inner Loop Substeps",
      badge: "Phase 5 / 6",
      desc: "Run 40-50 iterations of Adam gradient descent on mask logits z_k while keeping the classifier completely frozen. Use the Substep tabs and Iteration Scrubber below to inspect intermediate milestones (Iter 0, 10, 25, 40, 50)!",
      math: "\\mathcal{L} = - (\\lambda_{\\text{suff}} \\text{Suff} + \\lambda_{\\text{margin}} \\text{Margin}) + \\lambda_{\\text{overlap}} \\sum_{k<l} (M_k \\cdot M_l) + \\lambda_{\\text{sparse}} \\|M_k\\|_1",
      p1Tag: "Optimizing...", p2Tag: "Cat Mask M1", p3Tag: "Dog Mask M2", p4Tag: "Overlap Penalty",
      p1Desc: "Live Adam optimizer steps (40-50 iterations)",
      p2Desc: "Cat mask allocating to kitten",
      p3Desc: "Dog mask relocating off animal to bedding",
      p4Desc: "Overlap penalty dropping towards 0"
    },
    6: {
      title: "Step 6: Discriminative Allocation & Benchmark Results",
      badge: "Phase 6 / 6",
      desc: "Optimization completes! Overlap drops by 92% on Pets and 98% on MNIST. Cat evidence stays on the kitten, while Dog evidence relocates to the pink bedding—showing that the near-tie was driven by background context!",
      math: "\\text{Overlap Drop: } 0.850 \\to 0.068 \\quad (-92.0\\%), \\quad \\text{Margin Increase: } 0.000 \\to +0.537",
      p1Tag: "Masked Cat X", p2Tag: "Cat Mask M1", p3Tag: "Dog Mask M2", p4Tag: "Disjoint Allocation",
      p1Desc: "Masked image carrying Cat prediction",
      p2Desc: "Cat evidence: localized on kitten face/body",
      p3Desc: "Dog evidence: relocated to pink bedding",
      p4Desc: "Disjoint allocation achieved!"
    }
  };

  const substepConfigs = {
    '5.1': {
      title: "Substep 5.1: Mask Keep Intervention (Covering Up)",
      math: "X_{\\text{keep}, k} = M_k^{\\text{pixel}} \\odot X + (1 - M_k^{\\text{pixel}}) \\odot \\text{blur}(X)",
      desc: "In each optimization step, candidate mask M_k is upsampled to pixel resolution and blended with average blur baseline to generate masked image X_keep."
    },
    '5.2': {
      title: "Substep 5.2: Computing Sufficiency & Contrastive Margin",
      math: "\\text{Suff}_k = z_k(X_{\\text{keep}, k}), \\quad \\text{Margin}_k = z_k(X_{\\text{keep}, k}) - \\max_{l \\neq k} z_l(X_{\\text{keep}, k})",
      desc: "Push X_keep back through the frozen classifier. Maximizing Sufficiency ensures class k is still recognized; maximizing Margin forces tiles to favor class k over competitor l."
    },
    '5.3': {
      title: "Substep 5.3: Computing Overlap & Sparsity Penalties",
      math: "\\text{Overlap} = \\sum_{k<l} (M_k \\cdot M_l), \\quad \\text{Sparsity} = \\sum_r |M_{k,r}|",
      desc: "Calculate elementwise dot product between class masks to penalize spatial collision, and L1 norm to penalize bloated masks."
    },
    '5.4': {
      title: "Substep 5.4: Adam Gradient Descent Update on Logits",
      math: "\\frac{\\partial \\mathcal{L}}{\\partial \\mathbf{z}_k} = \\frac{\\partial \\mathcal{L}}{\\partial M_k} \\cdot M_k (1 - M_k), \\quad \\mathbf{z}_k \\leftarrow \\mathbf{z}_k - \\alpha \\text{Adam}(\\nabla_{\\mathbf{z}_k} \\mathcal{L})",
      desc: "Backpropagate loss gradients through the sigmoid function to update unconstrained mask logits z_k while holding neural network parameters completely frozen."
    }
  };

  function updateStepUI() {
    const cfg = stepConfigs[state.currentStep];
    if (!cfg) return;

    const isStep5 = (state.currentStep === 5);
    document.getElementById('step5-substep-card').style.display = isStep5 ? 'block' : 'none';

    if (isStep5 && substepConfigs[state.activeSubstep]) {
      const subCfg = substepConfigs[state.activeSubstep];
      document.getElementById('step-title').innerText = subCfg.title;
      document.getElementById('step-math').innerText = subCfg.math;
      document.getElementById('step-description').innerText = subCfg.desc;
    } else {
      document.getElementById('step-title').innerText = cfg.title;
      document.getElementById('step-badge').innerText = cfg.badge;
      document.getElementById('step-description').innerText = cfg.desc;
      document.getElementById('step-math').innerText = cfg.math;
    }

    document.getElementById('panel1-tag').innerText = cfg.p1Tag;
    document.getElementById('panel2-tag').innerText = cfg.p2Tag;
    document.getElementById('panel3-tag').innerText = cfg.p3Tag;
    document.getElementById('panel4-tag').innerText = cfg.p4Tag;

    document.getElementById('panel1-desc').innerText = cfg.p1Desc;
    document.getElementById('panel2-desc').innerText = cfg.p2Desc;
    document.getElementById('panel3-desc').innerText = cfg.p3Desc;
    document.getElementById('panel4-desc').innerText = cfg.p4Desc;

    document.querySelectorAll('.wizard-step').forEach(stepBtn => {
      const s = parseInt(stepBtn.dataset.step);
      stepBtn.classList.remove('active', 'completed');
      if (s === state.currentStep) stepBtn.classList.add('active');
      else if (s < state.currentStep) stepBtn.classList.add('completed');
    });

    document.getElementById('btn-optimize-step').style.display = isStep5 ? 'inline-flex' : 'none';
    document.getElementById('btn-optimize-all').style.display = isStep5 ? 'inline-flex' : 'none';
    document.getElementById('btn-reset-opt').style.display = isStep5 ? 'inline-flex' : 'none';

    updateUI();
  }

  function updateUI() {
    const metrics = computeInterventionalState();

    document.getElementById('metric-suff').innerText = (metrics.suff > 0 ? '+' : '') + metrics.suff.toFixed(3);
    document.getElementById('metric-margin').innerText = (metrics.margin > 0 ? '+' : '') + metrics.margin.toFixed(3);
    document.getElementById('metric-overlap').innerText = metrics.overlap.toFixed(3);
    document.getElementById('metric-sparse').innerText = metrics.sparse.toFixed(1) + ' tiles';

    document.getElementById('iter-counter').innerText = `Optimization Iteration: ${state.iteration} / ${state.maxIterations}`;

    const sliderScrubber = document.getElementById('slider-iter-scrubber');
    if (sliderScrubber) sliderScrubber.value = state.iteration;

    renderViewPanels();
    renderLossChart();
  }

  // ---------------------------------------------------------------------------
  // Event Listeners & Setup
  // ---------------------------------------------------------------------------
  function setupEventListeners() {
    document.querySelectorAll('.wizard-step').forEach(btn => {
      btn.addEventListener('click', () => {
        state.currentStep = parseInt(btn.dataset.step);
        updateStepUI();
      });
    });

    document.getElementById('btn-prev').addEventListener('click', () => {
      if (state.currentStep > 1) {
        state.currentStep--;
        updateStepUI();
      }
    });

    document.getElementById('btn-next').addEventListener('click', () => {
      if (state.currentStep < 6) {
        state.currentStep++;
        updateStepUI();
      }
    });

    document.getElementById('btn-autoplay').addEventListener('click', () => {
      if (state.autoPlaying) {
        clearInterval(state.autoPlayTimer);
        state.autoPlaying = false;
        document.getElementById('btn-autoplay').innerText = '▶ Auto Play';
      } else {
        state.autoPlaying = true;
        document.getElementById('btn-autoplay').innerText = '⏸ Pause';
        state.autoPlayTimer = setInterval(() => {
          if (state.currentStep === 5 && state.iteration < state.maxIterations) {
            loadTrajectoryIteration(state.iteration + 1);
          } else {
            state.currentStep = (state.currentStep % 6) + 1;
            updateStepUI();
          }
        }, 1200);
      }
    });

    // Scenario dropdown
    document.getElementById('example-select').addEventListener('change', (e) => {
      state.scenario = e.target.value;
      initScenarioData();
      updateStepUI();
    });

    // Substep tabs
    document.querySelectorAll('.substep-tab').forEach(tab => {
      tab.addEventListener('click', () => {
        document.querySelectorAll('.substep-tab').forEach(t => t.classList.remove('active'));
        tab.classList.add('active');
        state.activeSubstep = tab.dataset.substep;
        updateStepUI();
      });
    });

    // Milestone buttons & Scrubber slider
    document.querySelectorAll('.milestone-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const iter = parseInt(btn.dataset.iter);
        loadTrajectoryIteration(iter);
      });
    });

    const scrubber = document.getElementById('slider-iter-scrubber');
    if (scrubber) {
      scrubber.addEventListener('input', (e) => {
        const iter = parseInt(e.target.value);
        loadTrajectoryIteration(iter);
      });
    }

    // Optimization Buttons
    document.getElementById('btn-optimize-step').addEventListener('click', () => {
      loadTrajectoryIteration(state.iteration + 1);
    });

    document.getElementById('btn-optimize-all').addEventListener('click', () => {
      loadTrajectoryIteration(50);
    });

    document.getElementById('btn-reset-opt').addEventListener('click', () => {
      loadTrajectoryIteration(0);
    });

    // Hyperparameter sliders
    const bindSlider = (id, stateKey, displayId) => {
      const slider = document.getElementById(id);
      if (!slider) return;
      slider.addEventListener('input', (e) => {
        const val = parseFloat(e.target.value);
        state[stateKey] = val;
        document.getElementById(displayId).innerText = val.toFixed(2);
        precomputeTrajectory();
        updateUI();
      });
    };

    bindSlider('slider-lambda-suff', 'lambdaSuff', 'val-lambda-suff');
    bindSlider('slider-lambda-margin', 'lambdaMargin', 'val-lambda-margin');
    bindSlider('slider-lambda-overlap', 'lambdaOverlap', 'val-lambda-overlap');
    bindSlider('slider-lambda-sparse', 'lambdaSparse', 'val-lambda-sparse');
    bindSlider('slider-lr', 'lr', 'val-lr');
  }

  // ---------------------------------------------------------------------------
  // App Initialization
  // ---------------------------------------------------------------------------
  function init() {
    buildGridOverlays();
    initScenarioData();
    setupEventListeners();
    updateStepUI();
  }

  window.addEventListener('DOMContentLoaded', init);

})();
