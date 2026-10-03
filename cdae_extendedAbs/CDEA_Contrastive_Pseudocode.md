# CDEA-Contrastive Method — Algorithm Pseudocode

This document presents the complete pseudocode algorithm for **Class Distribution Evidence Allocation (CDEA-Contrastive)**, including base evidence extraction, mask parameterization, keep interventions, loss formulation, and penalized joint optimization over a frozen classifier.

---

```python
"""
================================================================================
ALGORITHM: CDEA-Contrastive (Class Distribution Evidence Allocation)
================================================================================
Inputs:
    X                : Input image tensor (3 x H x W), e.g., 3 x 224 x 224
    f_theta          : Pretrained image classifier (convolutional or ViT)
    K                : Number of top predicted rival classes to explain (e.g., K = 2 or K = 5)
    grid_h, grid_w   : Spatial allocation grid dimensions (e.g., 7 x 7 -> R = 49 tiles)
    num_steps        : Number of Adam optimization iterations (default: 50)
    lr               : Optimizer learning rate on mask logits (default: 0.5)
    lambda_suff      : Weight for predictive sufficiency (default: 1.0)
    lambda_margin    : Weight for contrastive margin (default: 1.0)
    lambda_overlap   : Penalty weight for spatial overlap (default: 0.20)
    lambda_sparse    : Penalty weight for mask size/sparsity (default: 0.05)
    lambda_mass      : Penalty weight for mass drift from base evidence (default: 0.10)
    use_shared       : Boolean, whether to allocate an explicit shared mask (default: False)

Outputs:
    M_unique         : Disjoint class-unique masks (K x R), M_k in [0, 1]
    M_shared         : Shared evidence mask (1 x R), optional
    metrics          : Interventional metrics (Sufficiency, Margin, Overlap Drop)
================================================================================
"""

Algorithm CDEA_Contrastive(X, f_theta, K, grid_h, grid_w, num_steps, lr, 
                          lambda_suff, lambda_margin, lambda_overlap, 
                          lambda_sparse, lambda_mass, use_shared):

    # --------------------------------------------------------------------------
    # STEP 1: Frozen Classifier Forward Pass & Rival Hypothesis Selection
    # --------------------------------------------------------------------------
    FREEZE_PARAMETERS(f_theta)  # Ensure neural network weights remain unchanged
    logits = f_theta(X)          # Shape: (NumClasses,)
    top_k_indices = TOP_K(logits, K)  # Select top-K rival class IDs, e.g., [Cat_ID, Dog_ID]

    # --------------------------------------------------------------------------
    # STEP 2: Base Evidence Extraction (Post-Hoc Attribution, e.g., Grad-CAM)
    # --------------------------------------------------------------------------
    target_layer = GET_LAST_CONV_OR_TRANSFORMER_BLOCK(f_theta)
    R = grid_h * grid_w
    E_base = ZEROS(K, R)

    for k in Range(K):
        c_k = top_k_indices[k]
        activations, gradients = BACKWARD_PASS_ATTRIBUTION(f_theta, target_layer, X, c_k)
        # Compute channel weights alpha_c = mean(gradients)
        weights = MEAN_SPATIAL(gradients) 
        # Grad-CAM saliency map L = ReLU( sum_c weights_c * activations_c )
        cam_raw = RELU( SUM_CHANNELS(weights * activations) )
        # Coarsen / pool saliency map onto grid_h x grid_w grid
        cam_grid = ADAPTIVE_AVG_POOL(cam_raw, (grid_h, grid_w)).FLATTEN()
        E_base[k] = cam_grid

    # Normalize base evidence per class so sum_r E_{k,r} = 1.0
    for k in Range(K):
        E_base[k] = E_base[k] / MAX(SUM(E_base[k]), 1e-8)

    # --------------------------------------------------------------------------
    # STEP 3: Mask Logit Parameterization & Initialization
    # --------------------------------------------------------------------------
    # To enable unconstrained gradient descent, define logit parameters z_k
    # Clamped inverse-sigmoid: z_k^{(0)} = logit( clamp(E_base, eps, 1-eps) )
    eps = 1e-6
    E_clamped = CLAMP(E_base, eps, 1.0 - eps)
    z_unique_logits = LOGIT(E_clamped)  # Shape: (K, R)
    z_unique_logits = CLAMP(z_unique_logits, -12.0, 12.0)
    ENABLE_GRADIENTS(z_unique_logits)

    if use_shared:
        z_shared_logits = ZEROS(R)
        ENABLE_GRADIENTS(z_shared_logits)
    else:
        z_shared_logits = None

    optimizer = ADAM([z_unique_logits, z_shared_logits], lr=lr)

    # Pre-compute average blur baseline for keep interventions
    X_blur = AVERAGE_POOL_2D(X, kernel_size=15, stride=1, padding=7)

    # --------------------------------------------------------------------------
    # STEP 4 & 5: Penalized Joint Optimization Loop (40-50 Iterations)
    # --------------------------------------------------------------------------
    for step in Range(num_steps):
        ZERO_GRADIENTS(optimizer)

        # Sigmoid activation maps logits to valid mask range (0, 1)
        M_unique = SIGMOID(z_unique_logits)  # Shape: (K, R)
        if use_shared:
            M_shared = SIGMOID(z_shared_logits)  # Shape: (R,)
            M_total = M_unique + M_shared.UNSQUEEZE(0)
        else:
            M_total = M_unique

        # ----------------------------------------------------------------------
        # Substep 5.1: Interventional Evaluation ("Covering Up")
        # ----------------------------------------------------------------------
        suff_scores = ZEROS(K)
        margin_scores = ZEROS(K)

        for k in Range(K):
            # Bilinear upsample 7x7 grid mask M_{total, k} to input pixel size (3 x H x W)
            M_pixel_k = UPSAMPLE_BILINEAR(M_total[k], target_size=(H, W))
            
            # Keep intervention: keep masked regions, replace rest with blur baseline
            X_keep_k = M_pixel_k * X + (1.0 - M_pixel_k) * X_blur
            
            # Forward pass through FROZEN network
            logits_keep = f_theta(X_keep_k)
            
            # Sufficiency: Logit of target class k under keep intervention
            z_k = logits_keep[top_k_indices[k]]
            suff_scores[k] = z_k

            # Contrastive Margin: z_k - max_{l != k} z_l
            z_competitors = [ logits_keep[top_k_indices[l]] for l in Range(K) if l != k ]
            margin_scores[k] = z_k - MAX(z_competitors)

        # Aggregate mean sufficiency and contrastive margin across top-K classes
        mean_suff = MEAN(suff_scores)
        mean_margin = MEAN(margin_scores)

        # ----------------------------------------------------------------------
        # Substep 5.3: Overlap, Sparsity, and Mass Drift Penalties
        # ----------------------------------------------------------------------
        # Overlap penalty: sum over pairs k < l of elementwise dot product (M_k . M_l)
        overlap_penalty = 0.0
        for k in Range(K):
            for l in Range(k + 1, K):
                overlap_penalty += SUM(M_unique[k] * M_unique[l])

        # Sparsity penalty: L1 norm of unique masks
        sparsity_penalty = MEAN( ABS_SUM_SPATIAL(M_unique) )

        # Mass drift penalty: deviation from base evidence mass
        mass_drift_penalty = MEAN( ABS( SUM_SPATIAL(M_unique) - SUM_SPATIAL(E_base) ) )

        # ----------------------------------------------------------------------
        # Combined Loss Computation & Adam Gradient Step
        # ----------------------------------------------------------------------
        Loss = - (lambda_suff * mean_suff + lambda_margin * mean_margin) \
               + lambda_overlap * overlap_penalty \
               + lambda_sparse * sparsity_penalty \
               + lambda_mass * mass_drift_penalty

        # Backpropagate loss gradients with respect to mask logits z_unique_logits
        BACKWARD(Loss)
        OPTIMIZER_STEP(optimizer)

        # Clamp logits so mask parameters stay within stable numeric range
        z_unique_logits = CLAMP(z_unique_logits, -12.0, 12.0)
        if use_shared:
            z_shared_logits = CLAMP(z_shared_logits, -12.0, 12.0)

    # --------------------------------------------------------------------------
    # STEP 6: Final Output Extraction & Interventional Verification
    # --------------------------------------------------------------------------
    M_unique_final = SIGMOID(z_unique_logits)
    M_shared_final = SIGMOID(z_shared_logits) if use_shared else None

    final_metrics = {
        "Sufficiency"        : mean_suff,
        "Contrastive_Margin" : mean_margin,
        "Spatial_Overlap"    : overlap_penalty,
        "Overlap_Reduction"  : (1.0 - overlap_penalty / MAX(INITIAL_OVERLAP, 1e-5)) * 100.0
    }

    Return M_unique_final, M_shared_final, final_metrics
```

---

## Explanation of Key Mathematical Components

1. **Unconstrained Logit Parameterization**:
   Direct optimization of bounded variables $M_k \in [0, 1]$ causes gradient saturation at boundaries ($0$ or $1$). CDEA parameterizes masks via unconstrained logits $\mathbf{z}_k \in \mathbb{R}^{49}$ passed through a sigmoid $\sigma(\mathbf{z}_k)$, enabling stable Adam gradient descent.

2. **Keep Intervention Baseline**:
   $$X_{\text{keep}, k} = M_k^{\text{pixel}} \odot X + (1 - M_k^{\text{pixel}}) \odot \text{blur}(X)$$
   Replaces non-selected regions with a local average blur baseline, preventing high-frequency edge artifacts while removing spatial evidence.

3. **Multi-Term Objective Function**:
   $$\mathcal{L} = - (\lambda_{\text{suff}} \cdot \text{Suff} + \lambda_{\text{margin}} \cdot \text{Margin}) + \lambda_{\text{overlap}} \sum_{k < l} (M_k \cdot M_l) + \lambda_{\text{sparse}} \| M_k \|_1 + \lambda_{\text{mass}} | \text{Mass} - \text{BaseMass} |$$
   - **Sufficiency** ensures the mask retains critical predictive features for target class $k$.
   - **Margin** forces mask tiles to favor target class $k$ over its top rival $l$.
   - **Overlap Penalty** eliminates spatial collisions where multiple classes claim the same pixels.
