# Top Quark Tagging with a Graph Neural Network

For more information refer to the repository
[xai4hep](https://github.com/farakiko/xai4hep/): an explainable AI (XAI) toolbox for interpreting state-of-the-art ML algorithms for high energy physics and the paper [1].

[1] Farouk Mokhtar et. al., *Do graph neural networks learn traditional jet substructure?*, [ML4PS @ NeurIPS 2022](https://ml4physicalsciences.github.io/2022/) [`arXiv:2211.09912`](https://arxiv.org/abs/2211.09912)

## Explaining ParticleNet with layerwise relevance propagation (LRP)
<img src="https://raw.githubusercontent.com/farakiko/xai4hep/main/docs/_static/images/rgraphs.png" alt="Trulli" style="width:100%">
The jet constituents are represented as nodes in (eta, phi) space with interconnections as edges, whose intensities correspond to the connection's edge R score.
Each node's intensity corresponds to the relative p<sub>T</sub> of the corresponding particle.
Constituents belonging to the three different CA subjets are shown in blue, red, and green in descending p<sub>T</sub> order. We observe that by the last EdgeConv block the model learns to rely more on edge connections between the different subjets.
