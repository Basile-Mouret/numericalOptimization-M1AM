**1. Gradient Descent (GD) & Variants**
*   **When to use:** Baseline for smooth optimization. Good for small/medium datasets.
*   **Specificities:** Needs a step size. Without line-search, you guess the step size. Using a line search (Armijo/Wolfe) automates finding a good step. Accelerated GD (Momentum/Nesterov) speeds up convergence for ill-conditioned problems.

**2. Newton & Quasi-Newton (BFGS)**
*   **When to use:** Mid-sized problems requiring fast (quadratic or superlinear) convergence where evaluating gradients is cheap compared to the steps taken.
*   **Specificities:**
    *   **Newton:** Uses exact Hessian. Extremely fast in iteration count but very expensive per step (requires matrix inversion). Fails if Hessian isn't positive definite.
    *   **BFGS:** Approximates the inverted Hessian dynamically. Robust, no second derivatives needed, but takes more memory than basic GD.

**3. Conjugate Gradient (CG)**
*   **When to use:** Very large-scale quadratic systems ($Ax = b$) or strictly non-linear problems where storing a Hessian or BFGS matrix is totally impossible due to memory.
*   **Specificities:** Relies only on exact matrix-vector products. Minimal memory footprint.

**4. Stochastic Algorithms (SGD & SAGA)**
*   **When to use:** Massive machine learning datasets where computing the full gradient over all points is too slow or impossible. 
*   **Specificities:**
    *   **SGD:** Computes gradient on single samples. Very noisy. Requires a gradually decreasing learning rate to converge.
    *   **SAGA:** Keeps a memory of past gradients to reduce variance/noise. Allows a constant learning rate and converges much faster than SGD.

**5. Adaptive Methods (Adagrad, Adam)**
*   **When to use:** Machine learning and neural networks, especially with noisy data or sparse features of differing scales.
*   **Specificities:**
    *   **Adagrad:** Shrinks the learning rate per-parameter based on historical gradient sizes.
    *   **Adam:** Combines momentum (moving average of gradients) and RMSProp (moving average of squared gradients). Best all-rounder for complex ML tasks.

**6. Constrained & Non-Smooth Optimization**
*   **When to use:** Hard mathematical constraints (e.g. $x \geq 0$) or non-differentiable penalties (e.g. $L_1$ norm/Lasso).
*   **Specificities:**
    *   **Projected GD:** Take a standard GD step, then mathematically "project" the point back into the valid constraint zone.
    *   **POCS:** Projection Onto Convex Sets. Finds the intersection of constraint sets by repeatedly projecting onto each one sequentially.
    *   **Proximal Gradient:** Splits the problem into $f(x) + g(x)$ (smooth + non-smooth). Takes a gradient step on $f$, and applies a proximal operator (like soft-thresholding) for $g$.

---

### 📂 Lab Sessions Recap


*   **TP1: Intro.** Basic Python setup and an early implementation of basic **BFGS**. Look here for boilerplate structures.
*   **TP2: Line Searches & 2nd Order.** Contains **GD**, **Armijo** (backtracking) line search, and exact **Newton/Hessian** GD. *Go here for step-size tuning loops.*
*   **TP3: Robust Line Search & Quasi-Newton.** Contains **Wolfe** line search, **Newton with Line Search**, and full **BFGS**. *Go here for the most rock-solid line search logic.*
*   **TP4: Acceleration & Large Scale.** Contains **Accelerated GD**, **Quadratic CG**, and **Non-linear CG**. *Go here for momentum implementations and matrix-free solvers.*
*   **TP5: Stochasticity.** Contains **GD**, **SGD**, and **SAGA**. *Go here for sampling logic, mini-batching, and variance reduction.*
*   **TP6: Adaptive ML Solvers.** Contains **Adagrad** and **Adam**. *Go here for deep learning style optimizers.*
*   **TP7: Constrained Optimization.** Contains **Projected GD** and **POCS**. *Go here if the problem has physical bounds or constraints.*
*   **TP8: Proximal Methods.** Contains **Proximal GD**. *Go here if optimizing something with an $L_1$ penalty or "soft-thresholding" wording.*