# 8 Baseline Encoders: Original Papers & Detailed Methods

## Summary Table

| # | Baseline | Original Paper | Authors | Year / Venue | Original Task | Original Evaluation Metric | Training Time (本repo) |
|---|----------|---------------|---------|-------------|--------------|--------------------------|----------------------|
| 1 | **FlyHash** | "A neural algorithm for a fundamental computing problem" | Sanjoy Dasgupta, Charles F. Stevens, Saket Navlakha | 2017 / *Science* 358(6364):793-796 | Similarity Search (LSH) | Mean Average Precision (mAP) | Instant (no training) |
| 2 | **Diehl & Cook** | "Unsupervised learning of digit recognition using spike-timing-dependent plasticity" | Peter U. Diehl, Matthew Cook | 2015 / *Frontiers in Computational Neuroscience* 9:99 | Digit Classification | Classification Accuracy (95%) | ~6 h (60K, GPU) |
| 3 | **SoftHebb** | "SoftHebb: Bayesian inference in unsupervised Hebbian soft winner-take-all networks" | Leo Kozachkov, Mikio C. Aoi, Jean-Jacques E. Slotine | 2022 / *Neuromorphic Computing and Engineering* 2(4):044019 | Feature Learning / Classification | Classification Accuracy | ~2 min |
| 4 | **Krotov** | "Unsupervised learning by competing hidden units" | Dmitry Krotov, John J. Hopfield | 2019 / *PNAS* 116(16):7723-7731 | Feature Learning + Classification | Classification Accuracy (98.3%) | ~1 min |
| 5 | **BioHash** | "Bio-Inspired Hashing for Unsupervised Similarity Search" | Chaitanya Ryali, John Hopfield, Leopold Grinberg, Dmitry Krotov | 2020 / *ICML* (PMLR 119:8295-8306) | Similarity Search / Retrieval | mAP, Precision@K | ~2 min |
| 6 | **WTA Hash** | "The Power of Comparative Reasoning" | Jay Yagnik, Dennis Strelow, David A. Ross, Ruei-Sung Lin | 2011 / *ICCV* pp.2431-2438 | Ordinal Embedding / Retrieval | Retrieval Accuracy, Precision | Instant |
| 7 | **SOM** | "Self-organized formation of topologically correct feature maps" | Teuvo Kohonen | 1982 / *Biological Cybernetics* 43:59-69 | Topographic Feature Mapping | Topological Ordering, Visualization | ~5 min |
| 8 | **LSH / SimHash** | "Similarity estimation techniques from rounding algorithms" | Moses S. Charikar | 2002 / *STOC* pp.380-388 | Similarity Estimation / Nearest Neighbor | Approximation Ratio, Retrieval Precision | Instant |

---

## Detailed Method Descriptions

### 1. FlyHash

| Item | Description |
|------|------------|
| **Paper** | Dasgupta S, Stevens CF, Navlakha S. "A neural algorithm for a fundamental computing problem." *Science*, 2017. DOI: [10.1126/science.aam9868](https://doi.org/10.1126/science.aam9868) |
| **Biological Inspiration** | Fruit fly (*Drosophila*) olfactory circuit: 50 Projection Neurons (PNs) --> 2000 Kenyon Cells (KCs), sparse random connectivity, APL neuron feedback inhibition |
| **Algorithm** | (1) Sparse random binary projection matrix (each output neuron连接~10% input); (2) Expand dimensionality (input_dim -> input_dim * 20); (3) Winner-Take-All: keep top 5% activations |
| **Learning Rule** | None (random, no training) |
| **Original Task** | Locality-Sensitive Hashing for similarity search. Paper discovered that the fly brain naturally implements a variant of LSH |
| **Original Metric** | mAP on MNIST, SIFT, GLOVE retrieval |
| **Clustering?** | No. Paper evaluates retrieval, not clustering |
| **Key Hyperparameters** | `projection_dim` (expansion ratio), `hash_length` (top-k), `sampling_ratio` (sparsity=0.1) |
| **Code** | `baselines/flyhash/encoder.py` |

---

### 2. Diehl & Cook (STDP-SNN)

| Item | Description |
|------|------------|
| **Paper** | Diehl PU, Cook M. "Unsupervised learning of digit recognition using spike-timing-dependent plasticity." *Frontiers in Computational Neuroscience*, 2015. DOI: [10.3389/fncom.2015.00099](https://doi.org/10.3389/fncom.2015.00099) |
| **Biological Inspiration** | Spiking neural network with biologically plausible STDP learning, lateral inhibition (via inhibitory neuron群), adaptive firing threshold (homeostasis) |
| **Algorithm** | (1) Input: Poisson spike trains (pixel intensity -> firing rate); (2) Excitatory LIF neurons with all-to-all input connections; (3) PostPre STDP rule (post-before-pre potentiation, otherwise depression); (4) Lateral inhibition via inhibitory layer; (5) Adaptive threshold for homeostasis |
| **Learning Rule** | STDP (Spike-Timing-Dependent Plasticity) |
| **Original Task** | Digit classification: unsupervised STDP training + supervised neuron-to-class label assignment + voting |
| **Original Metric** | Classification Accuracy = **95.0%** (6400 neurons, MNIST) |
| **Clustering?** | No. Paper evaluates classification via neuron labeling, not clustering |
| **Key Hyperparameters** | `n_neurons` (400), `simulation_time` (350ms), `dt` (1.0ms), `nu` (1e-4, 1e-2), `intensity` (128), `thresh` (-52mV), `norm` (78.4) |
| **Framework** | BindsNET |
| **Code** | `baselines/diehl_cook/encoder.py` |

---

### 3. SoftHebb

| Item | Description |
|------|------------|
| **Paper** | Kozachkov L, Aoi MC, Slotine JJE. "SoftHebb: Bayesian inference in unsupervised Hebbian soft winner-take-all networks." *Neuromorphic Computing and Engineering* 2(4):044019, 2022. DOI: [10.1088/2634-4386/ac98a9](https://doi.org/10.1088/2634-4386/ac98a9) |
| **Biological Inspiration** | Hebbian learning ("neurons that fire together, wire together") + soft WTA (probabilistic neural competition, all neurons participate with weighted probabilities) |
| **Algorithm** | (1) Multi-layer feedforward: Input(784) -> Dense+SoftWTA(1000) -> Dense+SoftWTA(500) -> Output(400); (2) Temperature-scaled softmax for soft competition; (3) Hebbian update: dW = eta * y * (x - u*w), y=softmax output; (4) L2-normalized weight rows |
| **Learning Rule** | Hebbian with self-normalization |
| **Original Task** | Unsupervised feature learning, Bayesian inference interpretation. Evaluated on downstream classification |
| **Original Metric** | Classification accuracy on MNIST, CIFAR-10 |
| **Clustering?** | No. Paper focuses on feature learning and Bayesian interpretation |
| **Key Hyperparameters** | `hidden_dims` ([1000,500]), `output_dim` (400), `t_invert` (5.0), `eta` (0.01), `n_epochs` (10) |
| **Official Repo** | https://github.com/NeuromorphicComputing/SoftHebb |
| **Code** | `baselines/softhebb/encoder.py` |

---

### 4. Krotov-Hopfield

| Item | Description |
|------|------------|
| **Paper** | Krotov D, Hopfield JJ. "Unsupervised learning by competing hidden units." *PNAS* 116(16):7723-7731, 2019. DOI: [10.1073/pnas.1820458116](https://doi.org/10.1073/pnas.1820458116). ArXiv: [1806.10122](https://arxiv.org/abs/1806.10122) |
| **Biological Inspiration** | Competing hidden units: winner strengthens connections (Hebbian), loser weakens them (anti-Hebbian). Power-law synaptic weights |
| **Algorithm** | (1) Compute activation Q = sign(W) * \|W\|^(p-1) @ X; (2) k-WTA selection: winner gets +1 signal, k-th neuron gets -delta (anti-Hebbian); (3) Weight update: dW = eta * [winner @ input - (winner * activation) * W]; (4) Lebesgue p-norm regularization; (5) Learning rate linearly annealed |
| **Learning Rule** | Hebbian / anti-Hebbian competitive learning |
| **Original Task** | Unsupervised feature learning + supervised linear readout for classification |
| **Original Metric** | Classification Accuracy = **98.3%** (MNIST, 400 hidden units + linear readout) |
| **Clustering?** | No. Paper evaluates classification with supervised readout, not clustering |
| **Key Hyperparameters** | `n_neurons` (100), `n_epochs` (200), `lr` (0.02), `delta` (0.4), `p` (2.0), `k` (2) |
| **Official Repo** | https://github.com/DimaKrotov/Biological_Learning |
| **Code** | `baselines/krotov/encoder.py` |

---

### 5. BioHash

| Item | Description |
|------|------------|
| **Paper** | Ryali C, Hopfield J, Grinberg L, Krotov D. "Bio-Inspired Hashing for Unsupervised Similarity Search." *ICML 2020*, PMLR 119:8295-8306. ArXiv: [2001.04907](https://arxiv.org/abs/2001.04907) |
| **Biological Inspiration** | Extension of FlyHash: sparse expansive motif (common in neurobiology) + Hebbian/anti-Hebbian synaptic plasticity. Unlike FlyHash, BioHash is **data-driven** |
| **Algorithm** | (1) Initialize sparse random projection matrix (fixed sparsity pattern); (2) Hebbian learning: dW = eta * (x^T @ y), y = top-k winners; (3) Only update active connections (maintain sparsity); (4) Column-wise L2 normalization; (5) Top-k binarization |
| **Learning Rule** | Hebbian with sparse constraints |
| **Original Task** | Unsupervised similarity search / nearest-neighbor retrieval |
| **Original Metric** | mAP, Precision@K on MNIST, CIFAR-10, GLOVE, SIFT. BioHash achieves ~3x mAP improvement over FlyHash at small hash lengths |
| **Clustering?** | No. Paper evaluates retrieval, not clustering |
| **Key Hyperparameters** | `hash_dim` (256), `sparse_ratio` (0.1), `k_winners` (5%), `n_epochs` (5), `lr` (0.01) |
| **Code** | `baselines/biohash/encoder.py` |

---

### 6. WTA Hash

| Item | Description |
|------|------------|
| **Paper** | Yagnik J, Strelow D, Ross DA, Lin R-S. "The Power of Comparative Reasoning." *ICCV 2011*, pp.2431-2438. [IEEE Xplore](https://ieeexplore.ieee.org/document/6126527/) |
| **Biological Inspiration** | Random dendritic compartmentalization + local lateral inhibition. Each "window" is a micro-circuit where only the strongest signal propagates (one-hot) |
| **Algorithm** | (1) Randomly group input features into windows (e.g. 64 windows of 8 features); (2) Within each window, find argmax -> one-hot encoding; (3) Concatenate one-hot codes from all windows; Output dim = n_hashes * window_size; Sparsity = 1/window_size per window |
| **Learning Rule** | None (random, no training) |
| **Original Task** | Ordinal embedding for image retrieval and ranking. WTA is a generalization of MinHash |
| **Original Metric** | Retrieval accuracy on visual recognition tasks |
| **Clustering?** | No. Paper evaluates retrieval/ranking |
| **Key Hyperparameters** | `n_hashes` (64), `window_size` (8), output_dim = 512, sparsity ~87.5% |
| **Code** | `baselines/wta_hash/encoder.py` |

---

### 7. SOM (Self-Organizing Map)

| Item | Description |
|------|------------|
| **Paper** | Kohonen T. "Self-organized formation of topologically correct feature maps." *Biological Cybernetics* 43:59-69, 1982. DOI: [10.1007/BF00337288](https://doi.org/10.1007/BF00337288) |
| **Biological Inspiration** | Cortical self-organization: neurons in 2D grid self-organize to reflect input statistics while maintaining topological order, similar to retinotopic/tonotopic maps in cortex |
| **Algorithm** | (1) 2D grid of neurons with weight vectors (e.g. 20x20=400 neurons); (2) Find Best Matching Unit (BMU) for each input; (3) Update BMU and neighborhood with Gaussian kernel; (4) Learning rate and neighborhood radius exponentially decay; (5) Encoding: compute negative distances to all neurons, top-k nearest activated |
| **Learning Rule** | Competitive learning with neighborhood cooperation |
| **Original Task** | Topographic feature mapping / visualization. Later widely used for clustering and data exploration |
| **Original Metric** | Topological ordering quality, visual inspection of learned maps |
| **Clustering?** | Partially. SOM is often used for clustering/visualization, but the 1982 paper focused on topology preservation, not clustering metrics |
| **Key Hyperparameters** | `map_height/width` (20x20), `n_epochs` (10), `lr_init/final` (0.5/0.01), `sigma_init/final`, `k_active` (5%) |
| **Code** | `baselines/som/encoder.py` |

---

### 8. LSH / SimHash

| Item | Description |
|------|------------|
| **Paper** | Charikar MS. "Similarity estimation techniques from rounding algorithms." *STOC 2002*, pp.380-388. 另见: Indyk P, Motwani R. "Approximate nearest neighbors: towards removing the curse of dimensionality." *STOC 1998* (LSH 理论基础) |
| **Biological Inspiration** | 非生物启发, 但与 population coding 有共通之处: random projections (like random synaptic weights), binary threshold activation (like neuronal firing), distributed binary representation |
| **Algorithm** | (1) Generate random Gaussian projection matrix R [input_dim x hash_dim]; (2) Hash: sign(data @ R) -> binary code; (3) Preserves cosine similarity: Pr[h(x)=h(y)] = 1 - angle(x,y)/pi; Non-parametric, no training |
| **Learning Rule** | None (random, no training) |
| **Original Task** | Approximate nearest neighbor search / similarity estimation |
| **Original Metric** | Approximation ratio, retrieval precision/recall |
| **Clustering?** | No. Paper provides theoretical guarantees for similarity search |
| **Key Hyperparameters** | `hash_dim` (128), random Gaussian hyperplanes, L2-normalized |
| **Code** | `baselines/lsh/encoder.py` |

---

## Categorization by Type

### By Learning Rule

| Category | Baselines | Description |
|----------|-----------|-------------|
| **No Training (Random)** | FlyHash, WTA Hash, LSH/SimHash | Fixed random projections, instant encoding |
| **Hebbian Learning** | SoftHebb, Krotov, BioHash | Local learning rules, biologically plausible |
| **Spiking (STDP)** | Diehl & Cook | Spike-timing-dependent plasticity in SNN |
| **Competitive Learning** | SOM | Self-organization with neighborhood function |

### By Original Task

| Original Task | Baselines |
|--------------|-----------|
| **Similarity Search / Retrieval** | FlyHash, BioHash, WTA Hash, LSH/SimHash |
| **Classification (via feature learning)** | Diehl & Cook, Krotov, SoftHebb |
| **Topographic Mapping** | SOM |

### Key Observation

**None of the 8 baselines originally evaluated clustering as the primary task.** This repository adapts all methods to clustering evaluation (KMeans, KMedoids, Spectral Clustering) with metrics like NMI, ARI, ACC, Purity -- this is a novel contribution.

---

## BibTeX References

```bibtex
% 1. FlyHash
@article{dasgupta2017neural,
  title={A neural algorithm for a fundamental computing problem},
  author={Dasgupta, Sanjoy and Stevens, Charles F and Navlakha, Saket},
  journal={Science},
  volume={358},
  number={6364},
  pages={793--796},
  year={2017},
  doi={10.1126/science.aam9868}
}

% 2. Diehl & Cook
@article{diehl2015unsupervised,
  title={Unsupervised learning of digit recognition using spike-timing-dependent plasticity},
  author={Diehl, Peter U and Cook, Matthew},
  journal={Frontiers in Computational Neuroscience},
  volume={9},
  pages={99},
  year={2015},
  doi={10.3389/fncom.2015.00099}
}

% 3. SoftHebb
@article{kozachkov2022softhebb,
  title={SoftHebb: Bayesian inference in unsupervised Hebbian soft winner-take-all networks},
  author={Kozachkov, Leo and Aoi, Mikio C and Slotine, Jean-Jacques E},
  journal={Neuromorphic Computing and Engineering},
  volume={2},
  number={4},
  pages={044019},
  year={2022},
  doi={10.1088/2634-4386/ac98a9}
}

% 4. Krotov-Hopfield
@article{krotov2019unsupervised,
  title={Unsupervised learning by competing hidden units},
  author={Krotov, Dmitry and Hopfield, John J},
  journal={Proceedings of the National Academy of Sciences},
  volume={116},
  number={16},
  pages={7723--7731},
  year={2019},
  doi={10.1073/pnas.1820458116}
}

% 5. BioHash
@inproceedings{ryali2020bio,
  title={Bio-Inspired Hashing for Unsupervised Similarity Search},
  author={Ryali, Chaitanya and Hopfield, John and Grinberg, Leopold and Krotov, Dmitry},
  booktitle={Proceedings of the 37th International Conference on Machine Learning (ICML)},
  pages={8295--8306},
  year={2020},
  volume={119},
  series={PMLR}
}

% 6. WTA Hash
@inproceedings{yagnik2011power,
  title={The power of comparative reasoning},
  author={Yagnik, Jay and Strelow, Dennis and Ross, David A and Lin, Ruei-Sung},
  booktitle={2011 International Conference on Computer Vision (ICCV)},
  pages={2431--2438},
  year={2011},
  organization={IEEE}
}

% 7. SOM
@article{kohonen1982self,
  title={Self-organized formation of topologically correct feature maps},
  author={Kohonen, Teuvo},
  journal={Biological Cybernetics},
  volume={43},
  number={1},
  pages={59--69},
  year={1982},
  doi={10.1007/BF00337288}
}

% 8. LSH / SimHash
@inproceedings{charikar2002similarity,
  title={Similarity estimation techniques from rounding algorithms},
  author={Charikar, Moses S},
  booktitle={Proceedings of the 34th Annual ACM Symposium on Theory of Computing (STOC)},
  pages={380--388},
  year={2002},
  doi={10.1145/509907.509965}
}

% LSH Theoretical Foundation
@inproceedings{indyk1998approximate,
  title={Approximate nearest neighbors: towards removing the curse of dimensionality},
  author={Indyk, Piotr and Motwani, Rajeev},
  booktitle={Proceedings of the 30th Annual ACM Symposium on Theory of Computing (STOC)},
  pages={604--613},
  year={1998}
}
```
