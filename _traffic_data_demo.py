import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import time
import numpy as np
import networkx as nx
import scipy.sparse as sp


class VGAE(nn.Module):
	def __init__(self, adj):
		super(VGAE,self).__init__()
		self.base_gcn = GraphConvSparse(args.input_dim, args.hidden1_dim, adj)
		self.gcn_mean = GraphConvSparse(args.hidden1_dim, args.hidden2_dim, adj, activation=lambda x:x)
		self.gcn_logstddev = GraphConvSparse(args.hidden1_dim, args.hidden2_dim, adj, activation=lambda x:x)

	def encode(self, X):
		hidden = self.base_gcn(X)
		self.mean = self.gcn_mean(hidden)
		self.logstd = self.gcn_logstddev(hidden)
		gaussian_noise = torch.randn(X.size(0), args.hidden2_dim)
		sampled_z = gaussian_noise*torch.exp(self.logstd) + self.mean
		return sampled_z

	def forward(self, X):
		Z = self.encode(X)
		A_pred = dot_product_decode(Z)
		return A_pred

class GraphConvSparse(nn.Module):
	def __init__(self, input_dim, output_dim, adj, activation = F.relu, **kwargs):
		super(GraphConvSparse, self).__init__(**kwargs)
		self.weight = glorot_init(input_dim, output_dim) 
		self.adj = adj
		self.activation = activation

	def forward(self, inputs):
		x = inputs
		x = torch.mm(x,self.weight)
		x = torch.mm(self.adj, x)
		outputs = self.activation(x)
		return outputs


def dot_product_decode(Z):
	A_pred = torch.sigmoid(torch.matmul(Z,Z.t()))
	return A_pred

def glorot_init(input_dim, output_dim):
	init_range = np.sqrt(6.0/(input_dim + output_dim))
	initial = torch.rand(input_dim, output_dim)*2*init_range - init_range
	return nn.Parameter(initial)


class GAE(nn.Module):
	def __init__(self,adj):
		super(GAE,self).__init__()
		self.base_gcn = GraphConvSparse(args.input_dim, args.hidden1_dim, adj)
		self.gcn_mean = GraphConvSparse(args.hidden1_dim, args.hidden2_dim, adj, activation=lambda x:x)

	def encode(self, X):
		hidden = self.base_gcn(X)
		z = self.mean = self.gcn_mean(hidden)
		return z

	def forward(self, X):
		Z = self.encode(X)
		A_pred = dot_product_decode(Z)
		return A_pred
		

# class GraphConv(nn.Module):
# 	def __init__(self, input_dim, hidden_dim, output_dim):
# 		super(VGAE,self).__init__()
# 		self.base_gcn = GraphConvSparse(args.input_dim, args.hidden1_dim, adj)
# 		self.gcn_mean = GraphConvSparse(args.hidden1_dim, args.hidden2_dim, adj, activation=lambda x:x)
# 		self.gcn_logstddev = GraphConvSparse(args.hidden1_dim, args.hidden2_dim, adj, activation=lambda x:x)

# 	def forward(self, X, A):
# 		out = A*X*self.w0
# 		out = F.relu(out)
# 		out = A*X*self.w0
#

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(adj_normalized)

def mask_test_edges(adj):
    # Function to build test set with 10% positive links
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    # TODO: Clean up.

    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]
    num_test = int(np.floor(edges.shape[0] / 10.))
    num_val = int(np.floor(edges.shape[0] / 20.))

    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])

    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], train_edges):
            continue
        if ismember([idx_j, idx_i], train_edges):
            continue
        if ismember([idx_i, idx_j], val_edges):
            continue
        if ismember([idx_j, idx_i], val_edges):
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])

    assert ~ismember(test_edges_false, edges_all)
    assert ~ismember(val_edges_false, edges_all)
    assert ~ismember(val_edges, train_edges)
    assert ~ismember(test_edges, train_edges)
    assert ~ismember(val_edges, test_edges)

    data = np.ones(train_edges.shape[0])

    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false


from sklearn.preprocessing import OneHotEncoder 
from sklearn.preprocessing import LabelEncoder

def one_hot_encode(arr_values):
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(arr_values)

    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    return onehot_encoded

def feature_merge(fea_list):
    feature = np.empty(shape=(fea_list[0].shape[0], 0))
    for fea in fea_list:
        feature = np.append(feature, fea, axis = 1)
    return feature

def load_data(filename):
    data_link_id = []
    data_width = []
    data_direction = []
    data_snodeid = []
    data_enodeid = []
    data_length = []
    data_speedclass = []
    data_lanenum = []
    lines_num = 0
    with open(filename, "r") as file:
        for line in file:
            lines_num += 1
            if lines_num == 1:
                continue
            line=line.strip('\n')   #将\n去掉
            line_arr = line.split('\t')
            data_link_id.append(line_arr[0])
            data_width.append(line_arr[1])
            data_direction.append(line_arr[2])
            data_snodeid.append(line_arr[3])
            data_enodeid.append(line_arr[4])
            data_length.append(line_arr[5])
            data_speedclass.append(line_arr[6])
            data_lanenum.append(line_arr[7])
    data_link_id = np.array(data_link_id, dtype = np.int64)
    # data_width = np.array(data_width, dtype = np.int)
    data_direction = np.array(data_direction, dtype = np.int)
    data_snodeid = np.array(data_snodeid, dtype = np.int)
    data_enodeid = np.array(data_enodeid, dtype = np.int)
    data_length = np.array(data_length, dtype = np.float)
    # data_speedclass = np.array(data_speedclass, dtype = np.int)
    # data_lanenum = np.array(data_lanenum, dtype = np.int)oh_width = one_hot_encode(data_width)
    
    oh_width = one_hot_encode(data_width)
    oh_speedclass = one_hot_encode(data_speedclass)
    oh_lanenum = one_hot_encode(data_lanenum)
    oh_length = np.array([data_length]).T

    oh_feature = feature_merge([oh_width, oh_length, oh_speedclass, oh_lanenum])


    G = nx.Graph()
    for i in range(lines_num - 1):
        G.add_edge(data_snodeid[i], data_enodeid[i])
    
    node_list = list(G.nodes)
    fea_matrix = np.zeros((len(node_list), oh_feature.shape[1]))

    for i in range(len(data_link_id)):
        s_node_id = data_snodeid[i]
        e_node_id = data_enodeid[i]
        link_feature = oh_feature[i]
        snode_index = node_list.index(s_node_id)
        enode_index = node_list.index(e_node_id)
        
        fea_matrix[snode_index, :] += link_feature
        fea_matrix[enode_index, :] += link_feature
    
    adj_matrix = nx.adjacency_matrix(G)
    
    return adj_matrix, fea_matrix

############################## train proc #################################
def print_msg(msg):
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), msg)
    return

def train():
    model = 'GAE'
    filename = r"C:\Users\pengl\Downloads\road_network_sub-dataset\road_network_sub-dataset"
    
    print_msg("begin loading data ...")
    adj, features = load_data(filename)
    print_msg("load data ok.")

    print_msg("begin initial data ...")
    input_dim = adj.shape[0] 
    hidden1_dim = 32
    hidden2_dim = 16
    use_feature = True
    num_epoch = 200
    learning_rate = 0.01

    # Store original adjacency matrix (without diagonal entries) for later
    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()

    adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
    adj = adj_train

    # Some preprocessing
    adj_norm = preprocess_graph(adj)

    num_nodes = adj.shape[0]

    features = sparse_to_tuple(features.tocoo())
    num_features = features[2][1]
    features_nonzero = features[1].shape[0]
    print_msg("initial data ok.")

    print_msg("begin create model...")
    # Create Model
    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

    adj_label = adj_train + sp.eye(adj_train.shape[0])
    adj_label = sparse_to_tuple(adj_label)

    adj_norm = torch.sparse.FloatTensor(torch.LongTensor(adj_norm[0].T), 
                                torch.FloatTensor(adj_norm[1]), 
                                torch.Size(adj_norm[2]))
    adj_label = torch.sparse.FloatTensor(torch.LongTensor(adj_label[0].T), 
                                torch.FloatTensor(adj_label[1]), 
                                torch.Size(adj_label[2]))
    features = torch.sparse.FloatTensor(torch.LongTensor(features[0].T), 
                                torch.FloatTensor(features[1]), 
                                torch.Size(features[2]))

    weight_mask = adj_label.to_dense().view(-1) == 1
    weight_tensor = torch.ones(weight_mask.size(0)) 
    weight_tensor[weight_mask] = pos_weight

    # init model and optimizer
    model = getattr(model,args.model)(adj_norm)
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    print_msg("create model ok.")

    def get_scores(edges_pos, edges_neg, adj_rec):

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        # Predict on test set of edges
        preds = []
        pos = []
        for e in edges_pos:
            # print(e)
            # print(adj_rec[e[0], e[1]])
            preds.append(sigmoid(adj_rec[e[0], e[1]].item()))
            pos.append(adj_orig[e[0], e[1]])

        preds_neg = []
        neg = []
        for e in edges_neg:

            preds_neg.append(sigmoid(adj_rec[e[0], e[1]].data))
            neg.append(adj_orig[e[0], e[1]])

        preds_all = np.hstack([preds, preds_neg])
        labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
        roc_score = roc_auc_score(labels_all, preds_all)
        ap_score = average_precision_score(labels_all, preds_all)

        return roc_score, ap_score

    def get_acc(adj_rec, adj_label):
        labels_all = adj_label.to_dense().view(-1).long()
        preds_all = (adj_rec > 0.5).view(-1).long()
        accuracy = (preds_all == labels_all).sum().float() / labels_all.size(0)
        return accuracy

    print_msg("begin train data...")
    # train model
    for epoch in range(args.num_epoch):
        t = time.time()

        A_pred = model(features)
        optimizer.zero_grad()
        loss = log_lik = norm*F.binary_cross_entropy(A_pred.view(-1), adj_label.to_dense().view(-1), weight = weight_tensor)
        if args.model == 'VGAE':
            kl_divergence = 0.5/ A_pred.size(0) * (1 + 2*model.logstd - model.mean**2 - torch.exp(model.logstd)).sum(1).mean()
            loss -= kl_divergence

        loss.backward()
        optimizer.step()

        train_acc = get_acc(A_pred,adj_label)

        val_roc, val_ap = get_scores(val_edges, val_edges_false, A_pred)
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(loss.item()),
            "train_acc=", "{:.5f}".format(train_acc), "val_roc=", "{:.5f}".format(val_roc),
            "val_ap=", "{:.5f}".format(val_ap),
            "time=", "{:.5f}".format(time.time() - t))


    test_roc, test_ap = get_scores(test_edges, test_edges_false, A_pred)
    print("End of training!", "test_roc=", "{:.5f}".format(test_roc),
        "test_ap=", "{:.5f}".format(test_ap))


if __name__ == "__main__":
    train()