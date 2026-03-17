ApexTitan V15 - Final Edition

ApexTitan V15 is a PyTorch second-order low-rank optimizer designed to accelerate deep learning training while saving memory. It combines numerical stability, bias correction, and adaptive preconditioning for large weight matrices.

🔹 Key Features

Low-Rank Optimization: Reduces the complexity of preconditioners for matrix weights.

Momentum & Bias Correction: Adam/AdamW-inspired updates with bias-adjusted gradients for stable convergence.

Memory-Efficient: Minimal storage for orthogonal bases and preconditioning matrices.

Stable Cholesky Inversion: Safely solves linear systems for gradient preconditioning.

Weight Decay Support: Integrated like in AdamW for regularization.

Power-Method Subspace Tracking: Dynamically updates the orthogonal basis to capture directions of maximal variance.
