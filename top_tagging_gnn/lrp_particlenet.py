import torch


class LRP_ParticleNet:

    """
    LRP class that introduces useful helper functions defined on a ParticleNet-based model,
    and an explain method that runs layerwise-relevance propagation.
    """

    def __init__(self, device, model, epsilon):
        self.device = device
        self.model = model.to(device)
        self.epsilon = epsilon  # for stability reasons in the lrp-epsilon rule (by default: a very small number)

    """
    explanation functions
    """

    def explain(self, input):
        """
        Primary function to call on an LRP instance to start explaining predictions.
        It registers hooks and runs a forward pass on the input, then it attempts to
        explain the whole model by looping over EdgeConv blocks.

        Args:
            input: tensor containing the input sample you wish to explain

        Returns:
            R_scores: a vector containing the relevance scores of the input features
            R_edges: a dictionary containing R_scores for the edges for each EdgeConv block
            edge_index: a dictionary containing the edge_index of each graph constructed during an EdgeConv block
        """

        # register forward hooks to retrieve intermediate activations
        activations = {}

        def get_activation(name):
            def hook(model, input, output):
                activations[name] = input[0]

            return hook

        # unpack the EdgeConv Block module to register appropriate hooks for the child modules
        self.num_convs = self.model.num_edge_conv_blocks
        for name, module in self.model.named_modules():
            if "edge_conv" in name:
                if ".edge_conv" not in name:
                    for num in range(self.num_convs):
                        if f".{num}" in name:
                            for n, m in module.named_modules():
                                if ("nn" in n) and (n != "edge_conv.nn"):
                                    m.register_forward_hook(get_activation(name + "." + n))
            elif "fc" in name:
                module.register_forward_hook(get_activation(name))

        # define parameters
        self.num_nodes = input.x.shape[0]
        self.num_neighbors = self.model.k

        # run forward pass
        self.model.eval()
        (
            preds,
            self.edge_activations,
            self.edge_block_activations,
            self.edge_index,
        ) = self.model(input)

        # detach gradients to speed-up quick LRP operations
        for elem in [
            self.edge_activations,
            self.edge_block_activations,
            self.edge_index,
        ]:
            for key, value in elem.items():
                elem[key] = value.detach().to(self.device)

        # get the activations
        self.activations = {}
        for key, value in activations.items():
            self.activations[key] = value.detach().to(self.device)

        # initialize the R_scores vector using the output predictions
        R_scores = preds.detach().to(self.device)
        print(f"Sum of R_scores of the output: {round(R_scores.sum().item(),4)}")

        # run LRP
        R_scores = self.redistribute_across_fc_layer(R_scores, "fc2")
        print(f"R_scores after 'fc2' layer: {round(R_scores.sum().item(),4)}")

        R_scores = self.redistribute_across_fc_layer(R_scores, "fc1")
        print(f"R_scores after 'fc1' layer: {round(R_scores.sum().item(),4)}")

        R_scores = self.redistribute_across_global_pooling(R_scores)
        print(f"R_scores after global_pooling {round(R_scores.sum().item(),4)}")

        # run LRP over EdgeConv blocks
        R_edges = {}
        self.skip_connection_R_scores = None

        # loop over EdgeConv blocks
        for idx in range(self.num_convs - 1, -1, -1):
            R_scores, R_edges[f"edge_conv_{idx}"] = self.redistribute_EdgeConv(R_scores, idx)
            print(f"R_scores after EdgeConv # {idx}: {round((R_scores.sum()).item(),4)}")

        # detach and put on cpu to save
        for key, value in self.edge_index.items():
            self.edge_index[key] = value.detach().cpu()

        for key, value in R_edges.items():
            R_edges[key] = value.detach().cpu()

            # sum over the latent features' edge Rscores
            R_edges[key] = torch.abs(R_edges[key]).sum(axis=1)

            # normalize the sum of edge Rscores of all jets to be 1
            R_edges[key] = R_edges[key] / sum(R_edges[key])

        return R_edges, self.edge_index

    """
    EdgeConv redistribution
    """

    def redistribute_EdgeConv(self, R_scores, idx):
        """
        Function that redistributes R_scores over an EdgeConv block.

        takes R_scores ~ (num_nodes, latent_dim_old)
        and returns R_scores ~ (num_nodes, latent_dim_new)
        """

        # seperate the skip connection neurons from the others
        self.skip_connection_R_scores = R_scores[:, self.model.kernel_sizes[idx + 1] :]
        R_scores = R_scores[:, : self.model.kernel_sizes[idx + 1]]

        R_edges = self.redistribute_across_edge_pooling(R_scores, idx)

        R_scores = self.redistribute_across_DNN(R_edges, idx)

        R_scores = self.redistribute_concat_step(R_scores, idx)

        return R_scores + self.skip_connection_R_scores, R_edges

    """
    special redistribution rules unique to ParticleNet
    """

    def redistribute_concat_step(self, R_old, idx):
        """
        Useful to redistribute the R_scores backward from the concatenation step that happens
        at the begining of the EdgeConv block.

        Args
            R_old ~ (num_nodes*k, latent_dim)

        Returns
            R_new ~ (num_nodes, latent_dim/2)

        Note: latent_dim should be an even number as it is a concat of two node features.
        Note: Assumes that the concat is [x_i, x_j] not [x_i, x_i-x_j].
        """

        latent_dim_old = R_old.shape[-1]
        latent_dim_new = int(latent_dim_old / 2)

        R_new = torch.zeros([self.num_nodes, latent_dim_new]).to(self.device)

        # loop over nodes
        for i in range(self.num_nodes):
            for num_x, x in enumerate(self.edge_index[f"edge_conv_{idx}"][1]):
                if i == x:
                    R_new[i] += R_old[num_x, :latent_dim_new]

            for num_x, x in enumerate(self.edge_index[f"edge_conv_{idx}"][0]):
                if i == x:
                    R_new[i] += R_old[num_x, latent_dim_new:]

        return R_new

    def redistribute_across_edge_pooling(self, R_old, idx):
        """
        Useful to reditsribute the R_scores backward from the averaging of neighboring edges.
        It redistributes R_scores from the nodes over the edges.

        Args
            R_old ~ (num_nodes, latent_dim) and edge_activations ~ (num_nodes*k, latent_dim)

        Returns
            R_new ~ (num_nodes*k, latent_dim)
        """

        latent_dim = R_old.shape[1]

        R_new = torch.ones([self.num_nodes * self.num_neighbors, latent_dim]).to(self.device)

        # loop over nodes
        for i in range(self.num_nodes):
            # loop over neighbors
            for j in range(self.num_neighbors):
                # summing the edge_activations node_i (i.e. the edge_activations of the neighboring nodes)
                deno = self.edge_activations[f"edge_conv_{idx}"][
                    (i * self.num_neighbors) : (i * self.num_neighbors) + self.num_neighbors
                ].sum(axis=0)

                # redistribute the R_scores of node_i according to how activated each edge_activation (normalized by deno)
                R_new[(i * self.num_neighbors) + j] = (
                    R_old[i]
                    * self.edge_activations[f"edge_conv_{idx}"][(i * self.num_neighbors) + j]
                    / (deno + self.epsilon)
                )

        return R_new

    def redistribute_across_global_pooling(self, R_old):
        """
        Useful to reditsribute the R_scores backward from the averaging of all nodes.
        It redistributes R_scores from the whole jet over the nodes.

        Args
            R_old ~ (1, latent_dim)

        Returns
            R_new ~ (num_nodes, latent_dim)
        """

        latent_dim = R_old.shape[1]

        R_new = torch.ones([self.num_nodes, latent_dim]).to(self.device)
        x = self.edge_block_activations[f"edge_conv_{self.num_convs-1}"]

        # loop over nodes
        for i in range(self.num_nodes):
            deno = x.sum(axis=0) + self.epsilon  # summing the activations of all nodes

            # redistribute the R_scores of the whole jet over the nodes (normalized by deno)
            R_new[i] = R_old[0] * x[i] / deno

        return R_new

    """
    lrp-epsilon rule
    """

    def redistribute_across_DNN(self, R_scores, idx):
        """
        Implements the lrp-epsilon rule presented in the following reference: https://doi.org/10.1007/978-3-030-28954-6_10.
        Follows simple DNN LRP redistribution over the Sequential FC layers of a given EdgeConv.

        Args
            R_old ~ (num_nodes*k, latent_dim_old)

        Returns
            R_new ~ (num_nodes*k, latent_dim_new)
        """

        layer_names = []

        # loop over DNN layers
        for name, layer in self.model.named_modules():
            if f"edge_conv_blocks.{idx}.edge_conv.nn." in str(name):
                if "Linear" not in str(layer):
                    continue
                layer_names.append(name)

        layer_names.reverse()

        for _i, name in enumerate(layer_names):
            layer = self.name2layer(name)
            W = layer.weight.detach()  # get weight matrix
            W = torch.transpose(W, 0, 1)  # sanity check of forward pass: (torch.matmul(x, W) + layer.bias) == layer(x)

            # (1) compute the denominator
            denominator = torch.matmul(self.activations[name], W) + self.epsilon

            # (2) scale the R_scores
            scaledR = R_scores / denominator

            # (3) compute the new R_scores
            R_scores = torch.matmul(scaledR, torch.transpose(W, 0, 1)) * self.activations[name]

        return R_scores

    def redistribute_across_fc_layer(self, R_scores, layer_name):
        """
        Implements the lrp-epsilon rule presented in the following reference: https://doi.org/10.1007/978-3-030-28954-6_10.
        Follows simple DNN LRP redistribution over a single FC layer.

        takes R_old ~ (num_nodes, latent_dim_old)
        and returns R_new ~ (num_nodes, latent_dim_new)
        """

        # loop over DNN layers
        layer = self.name2layer(layer_name)

        W = layer.weight.detach()  # get weight matrix
        W = torch.transpose(W, 0, 1)  # sanity check of forward pass: (torch.matmul(x, W) + layer.bias) == layer(x)

        # (1) compute the denominator
        denominator = torch.matmul(self.activations[layer_name], W) + self.epsilon

        # (2) scale the R_scores
        scaledR = R_scores / denominator

        # (3) compute the new R_scores
        R_scores = torch.matmul(scaledR, torch.transpose(W, 0, 1)) * self.activations[layer_name]

        return R_scores

    """
    helper functions
    """

    def index2name(self, layer_index):
        """
        Given the index of a layer (e.g. 3) returns the name of the layer (e.g. .nn1.3)
        """
        layer_name = list(self.activations.keys())[layer_index - 1]
        return layer_name

    def name2layer(self, layer_name):
        """
        Given the name of a layer (e.g. .nn1.3) returns the corresponding torch module (e.g. Linear(...))
        """
        for name, module in self.model.named_modules():
            if layer_name == name:
                return module
