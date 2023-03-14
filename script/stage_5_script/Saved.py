class GCN(nn.Module):
    def __init__(self, input_features, hidden_dim, output_features, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(input_features, hidden_dim)
        self.gc2 = GraphConvolution(hidden_dim, output_features)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))

    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        output = torch.spmm(adj, support)
        output = output + self.bias
        return output