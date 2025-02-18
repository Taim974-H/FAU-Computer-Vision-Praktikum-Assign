# Analysis of RANSAC Variants: MLESAC and Preemptive RANSAC

## 1. MLESAC Analysis

### Parameter Sensitivity
- MLESAC shows reduced sensitivity to the threshold parameter (ε) compared to standard RANSAC
- The gamma parameter (outlier cost multiplier) provides additional control over model fitting

### Advantages
1. **Robust Cost Function**
   - Uses a maximum likelihood estimation approach
   - Better handles mixed distributions of inliers and outliers
   - More sophisticated scoring mechanism than binary inlier/outlier classification

2. **Quality of Fit**
   - Generally produces more accurate plane estimates
   - Better handles cases with varying noise levels
   - More reliable in scenes with multiple structural elements

3. **Consistency**
   - More consistent results across different runs
   - Less sensitive to random initialization

### Disadvantages
1. **Computational Overhead**
   - Higher computational cost due to distance calculations
   - More complex scoring mechanism
   - Additional parameter (gamma) needs tuning

2. **Parameter Tuning**
   - Requires careful selection of gamma parameter
   - Trade-off between robustness and efficiency

In our case, the results were almost identical even after a lot of parameter tuning but the number of inliers differed:
RANSAC floor inliers: 167887
MLESAC floor inliers: 167677
RANSAC top plane inliers: 22886
MLESAC top plane inliers: 22877

Due to small difference in inliers, the dimension of box was almost identical:
RANSAC Plane Distance: 0.956411
MLESAC Plane Distance: 0.929692
RANSAC - Length: 0.51, Width: 0.36
MLESAC - Length: 0.51, Width: 0.36

## 2. Preemptive RANSAC Analysis

### Parameter Sensitivity Study

#### Impact of M (Initial Hypotheses)
- **M = 50**: 
  - Fast execution but less reliable results
  - May miss optimal solutions
  - Higher variance in plane estimation

- **M = 100**:
  - Good balance of speed and accuracy
  - More stable plane estimates
  - Sufficient hypothesis space for most scenes

- **M = 200-500**:
  - Most reliable results but slower execution
  - Diminishing returns after certain point
  - Higher memory requirements

#### Impact of B (Batch Size)
- Small B (50):
  - More frequent hypothesis pruning
  - Faster execution but potentially less accurate
  - More sensitive to local noise

- Medium B (100):
  - Good balance of evaluation thoroughness
  - Stable hypothesis elimination
  - Reasonable memory usage

- Large B (200+):
  - More thorough hypothesis evaluation
  - Higher memory requirements
  - Slower execution with marginal benefits

### Advantages
1. **Efficiency**
   - Significantly faster than standard RANSAC
   - Early elimination of poor hypotheses
   - Scalable to large point clouds

2. **Resource Management**
   - Controlled memory usage through batching
   - Progressive reduction in computational load
   - Adaptable to different time budgets

3. **Quality Control**
   - Maintains good quality of results
   - Can be combined with MLESAC for better accuracy
   - Flexible parameter adjustment based on requirements

### Disadvantages
1. **Parameter Sensitivity**
   - Results depend on M and B choices
   - Trade-off between speed and accuracy
   - May require tuning for different scenes

2. **Early Elimination Risk**
   - Potential elimination of good hypotheses early
   - More sensitive to noise in early batches
   - May miss global optimum

 In our case, M=100, and B = 100, had a balanced output for the bottom plane as can be seen from output figures when the code is run.