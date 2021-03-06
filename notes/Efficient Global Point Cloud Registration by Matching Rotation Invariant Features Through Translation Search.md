# Efficient Global Point Cloud Registration by Matching Rotation Invariant Features Through Translation Search——GoTS

## Intro

- Decoupling the point set registration of 6D space to translation and rotation.
- Compared to Go-ICP, the speed and accuracy improved.
- By decoupling the transformation to translation and rotation, by BnBs approach to find the global optimal of translation. Afterwards, use RANSAC to estimate the Rotation( Or use local optimal(GoTS+));

## Notes

- Traditional local methods suffice only when the relative transformation is small and there is a large proportion of true overlap[1–3].
- In contrast, a global method is needed when there is a large relative transformation or when the overlap proportion is small.
- However, the time complexity of BnB optimization is exponential in the dimensionality of the problem.
- Most existing global methods optimize an objective function in the parameter space of SE(3), which has a dimensionality of six; thus, they are usually very slow.



- BnB optimization is first used to globally optimize the rotation to align translation invariant features, namely, the surface normal distributions, constructed from the original point clouds.

## Motivation

- **Efficient Global Point Cloud Alignment using Bayesian Nonparametric Mixtures**
  - Decoupling

## Preknowledge

- $\infty norm$ 无穷范数
  $$
  ||x||_{\infty} = max({|x_1|, |x_2|, |x_3|,...|x_n|})
  $$
  