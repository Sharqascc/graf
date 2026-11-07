# GETAD: Graph Enhanced Trajectory Anomaly Detection

Reference paper saved for related work, methodology, and paper-structure
comparison. Not our work unless stated otherwise; cite as the SIGSPATIAL
'25 paper below.

## Citation

> Jonathan Kabala Mbuya, Dieter Pfoser, and Antonios Anastasopoulos. 2025.
> **Graph Enhanced Trajectory Anomaly Detection.**
> In *The 33rd ACM International Conference on Advances in Geographic
> Information Systems (SIGSPATIAL '25)*, November 3-6, 2025, Minneapolis,
> MN, USA. ACM, New York, NY, USA, 12 pages.
> https://doi.org/10.1145/3748636.3762752
>
> arXiv:2509.18386v1 [cs.LG], 22 Sep 2025. https://arxiv.org/abs/2509.18386

**Authors:** Jonathan Kabala Mbuya, Dieter Pfoser, Antonios Anastasopoulos
(all George Mason University)

## Abstract (verbatim)

Trajectory anomaly detection is a crucial task in intelligent
transportation systems, ride-hailing platforms, and mobility safety
monitoring. Existing methods only consider limited aspects of the
trajectory nature and its movement space by treating trajectories as
sequences of sampled locations, with sampling determined by positioning
technology, e.g. GPS, or by high-level abstractions such as staypoints.
Trajectories are analyzed in Euclidean space, neglecting the constraints
and connectivity information of the underlying movement network, e.g.
road or transit networks. The proposed Graph Enhanced Trajectory
Anomaly Detection (GETAD) framework tightly integrates road network
topology, segment semantics, and historical travel patterns to model
trajectory data. GETAD uses a Graph Attention Network to learn
road-aware embeddings that capture both physical attributes and
transition behavior, and augments these with graph-based positional
encodings that reflect the spatial layout of the road network. A
Transformer-based decoder models sequential movement, while a
multiobjective loss function combining autoregressive prediction and
supervised link prediction ensures realistic and structurally coherent
representations. To improve the robustness of anomaly detection, we
introduce Confidence Weighted Negative Log Likelihood (CW-NLL), an
anomaly scoring function that emphasizes high-confidence deviations.
Experiments on real-world and synthetic datasets demonstrate that GETAD
achieves consistent improvements over existing methods, particularly in
detecting subtle anomalies in road-constrained environments. These
results highlight the benefits of incorporating graph structure and
contextual semantics into trajectory modeling, enabling more precise
and context-aware anomaly detection.

## Contributions

1. **Graph-enhanced architecture.** GAT learns road-segment embeddings
   from static infrastructure attributes (length, road type, lanes,
   in/out degree) plus dynamic statistics (normalized visit frequency),
   with attention modulated by empirical transition probability.
2. **Multi-objective training.** Combines autoregressive next-segment
   cross-entropy with supervised link prediction over road transitions.
3. **Graph positional encodings (GPE).** Based on shortest-path hop
   distances between segments; gives the decoder spatial layout without
   relying on sequential indices.
4. **CW-NLL anomaly score.** Weights token-level NLL by model confidence
   (entropy-based) so that low-likelihood AND high-certainty predictions
   dominate the score. Distinguishes rare-but-plausible from truly
   anomalous.
5. **Extensive experiments.** Porto taxi + synthetic Pattern-of-Life
   San Francisco; beats SAE, VSAE, GM-VSAE, LM-TAD on F1 and PR-AUC.

## Section structure

1. Introduction
2. Related Work
   - 2.1 Trajectory anomaly detection (heuristic + learning-based)
   - 2.2 Graph learning on trajectory data
3. Preliminaries
   - 3.1 Road network as directed graph G = (V, E)
   - 3.2 Map-matched trajectory (sequence of road-segment tokens)
   - 3.3 Problem formulation (online anomaly detection)
4. Method
   - 4.1 Graph-based road segment embedding (GAT)
   - 4.2 Graph positional embedding (GPE)
   - 4.3 Trajectory modeling with Transformer decoder
   - 4.4 Learning objectives (CE + link prediction)
   - 4.5 Anomaly score (CW-NLL)
5. Experiments
   - 5.1 Datasets & preprocessing
     (synthetic Pattern-of-Life, Porto; ground truth via
     constrained/unconstrained detour generation; tokenization)
   - 5.2 Baselines (SAE, VSAE, GM-VSAE, LM-TAD)
   - 5.3 Evaluation metrics (PR-AUC, F1)
6. Results (Simulation + Porto, constrained + unconstrained)
7. Ablation
   - 7.1 Anomaly scoring (CW-NLL vs NLL vs perplexity)
   - 7.2 Positional encoding x structural supervision
8. Conclusion

## Key equations (paraphrased clean)

- Empirical transition probability:
  p(v_i -> v_j) = count(v_i -> v_j) / count(v_i)
- GAT attention score: standard GAT concat-linear scoring, plus a term
  p(v_i, v_j) * a_2^T (so frequently observed transitions get higher
  attention weight).
- GPE distance matrix: D_ij = shortest-path hop distance between
  segments v_i and v_j (d_max if disconnected). Learnable embedding
  lookup E_dist[D_ij] in R^{d_model}. Final ge_i = sum_j P_ij.
- Token embedding: z_i = h(v_i) + ge(v_i)  (GAT + GPE)
- CE loss: L_CE = -sum log P(v_i | v_1..v_{i-1})
- Link loss: L_link = binary CE on transition labels vs random pairs
- Total: L_total = lambda_CE * L_CE + lambda_link * L_link
- CW-NLL: score = sum_i c_i * loss_i
  where c_i = 1 - H(p_i) / log(V), H = entropy, V = vocab size.
  High c_i = model was confident; weight those failures more.

## Results (PR-AUC from Table 2)

| Dataset | Setting | SAE | VSAE | GM-VSAE | LM-TAD | **GETAD** |
|---|---|---|---|---|---|---|
| Simulation | constrained | 0.720 | 0.696 | 0.839 | 0.973 | **0.988** |
| Simulation | unconstrained | 0.711 | 0.703 | 0.777 | 0.965 | **0.983** |
| Porto | constrained | 0.566 | 0.564 | 0.664 | 0.732 | **0.756** |
| Porto | unconstrained | 0.598 | 0.609 | 0.796 | 0.794 | **0.825** |

## Datasets

- **Synthetic (Pattern-of-Life, San Francisco).** ABM simulating 5000
  agents over 2 months on OSM SF road network. 68.4M points, 1.05M
  trajectories, 27K road segments. Already map-matched (generated on
  the graph). Filter: >=25 segments, agents with >100 trajectories.
- **Porto taxi (real-world).** 442 taxis, Jan 2013 - Jun 2014, 15 s GPS
  sampling. 33.5M points, 852K trajectories, 10.5K OSM road segments.
  Map-matched via Fast Map Match.

## Baselines

- SAE (sequence autoencoder, reconstruction error)
- VSAE (variational autoencoder)
- GM-VSAE (Gaussian mixture prior)
- LM-TAD (Transformer LM, prior work by the same authors)

## Ablation findings

Anomaly scoring (Porto, PR-AUC):
- constrained:  Perplexity 0.682, NLL 0.715, **CW-NLL 0.756**
- unconstrained: Perplexity 0.770, NLL 0.787, **CW-NLL 0.825**

Positional encoding x link supervision:
- RPE alone:        0.633 F1 / 0.652 PR-AUC
- RPE + link:       0.768 F1 / 0.829 PR-AUC  (best)
- GPE alone:        0.739 F1 / 0.810 PR-AUC
- GPE + link:       0.756 F1 / 0.825 PR-AUC

Takeaway: proximity information is critical; GPE gives it for free,
link supervision gives it via extra training signal. Combined they
overlap - RPE+link beats GPE+link slightly, but GPE-only already beats
RPE-only by a wide margin.

## Why saved / relevance to our work

- **Same domain:** trajectory data, road networks, graph structure.
- **Same ML stack:** GAT encoder + Transformer decoder + autoregressive
  next-token objective. Direct comparison point if we use a similar
  architecture.
- **Different task:** they do *anomaly detection* (score each trajectory);
  our graf work does *surrogate safety classification* (label each
  window with TTC/DCPA-critical). The label-generation pipeline (SSM
  thresholds) is orthogonal to their anomaly score.
- **Useful for related work section.** Establishes the state of the art
  for graph-based trajectory modeling. If we cite "prior work uses
  graph structure on trajectories, but for anomaly detection, not for
  surrogate safety," this is one of the anchors.
- **Useful as structure template.** The Preliminaries -> Method ->
  Experiments -> Ablation flow and their use of PR-AUC + F1 is exactly
  what our paper's structure should mirror.
- **CW-NLL idea** could inspire an analog for SSM scoring: weight
  critical-pair counts by how confident the model is that a window is
  dangerous. Worth remembering if we ever move from binary labels to
  scores.

## Full text

arXiv PDF: https://arxiv.org/abs/2509.18386
