import dgl
import dgl.function
import scipy.sparse as sp
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import os.path as osp
import pickle
import numpy as np
import itertools
import collections
import warnings

warnings.filterwarnings("ignore")

# 保存处理好的数据
Data = collections.namedtuple("Data", ["x", "y", "dgl_graph", "trn_mask", "val_mask", "test_mask"])


# 处理Cora数据
class CoraData(object):
    download_url = "https://github.com/kimiyoung/planetoid/raw/master/data"
    filenames = ["ind.cora.{}".format(name) for name in ['x', 'tx', 'allx', 'y', 'ty', 'ally', 'graph', 'test.index']]

    def __init__(self, data_root="E:\\datas\\Algs\\GNN\\cora", rebuild=False):
        """包括数据下载、处理、加载等功能 当数据的缓存文件存在时，将使用缓存文件，否则将下载、处理，并缓存到磁盘
        Args:-------
        data_root: string, optional 存放数据的目录，原始数据路径: {data_root}/raw 缓存数据路径: {data_root}/processed_cora_dgl.pkl
        rebuild: boolean, optional 是否需要重新构建数据集，当设为True时，如果缓存数据存在也会重建数据"""
        self.data_root = data_root
        save_file = osp.join(self.data_root, "processed_cora_dgl.pkl")
        if osp.exists(save_file) and not rebuild:
            print("Using Cached file: {}".format(save_file))
            self._data = pickle.load(open(save_file, "rb"))
        else:
            # self.maybe_download()
            self._data = self.process_data()
            with open(save_file, "wb") as f:
                pickle.dump(self.data, f)
        print("Cached file: {}".format(save_file))

    @property
    def data(self):
        """返回Data数据对象，包括x, y, adjacency, train_mask, val_mask, test_mask"""
        return self._data

    def process_data(self):
        """ 处理数据，得到节点特征和标签，邻接矩阵，训练集、验证集以及测试集 """
        print("Process data ...")
        _, tx, allx, y, ty, ally, graph, test_index = [self.read_data(osp.join(self.data_root, name)) for name in
                                                       self.filenames]
        train_index = np.arange(y.shape[0])
        val_index = np.arange(y.shape[0], y.shape[0] + 500)
        sorted_test_index = sorted(test_index)
        x = np.concatenate((allx, tx), axis=0)
        y = np.concatenate((ally, ty), axis=0).argmax(axis=1)
        x[test_index] = x[sorted_test_index]
        y[test_index] = y[sorted_test_index]
        num_nodes = x.shape[0]
        train_mask = np.zeros(num_nodes, dtype=bool)
        val_mask = np.zeros(num_nodes, dtype=bool)
        test_mask = np.zeros(num_nodes, dtype=bool)
        train_mask[train_index] = True
        val_mask[val_index] = True
        test_mask[test_index] = True
        dgl_graph = self.build_dgl_graph(graph)
        print("Node's feature shape: ", x.shape)
        print("Node's label shape: ", y.shape)
        print("Number of total edges: ", dgl_graph.num_edges())
        print("Number of training nodes: ", train_mask.sum())
        print("Number of validation nodes: ", val_mask.sum())
        print("Number of test nodes: ", test_mask.sum())
        return Data(x=x, y=y, dgl_graph=dgl_graph, trn_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

    @staticmethod
    def build_dgl_graph(adj_dict):
        """根据邻接表创建邻接矩阵"""
        edge_index = []
        for key in adj_dict.keys():
            edge_index.append([key, key])
            for item in adj_dict[key]:
                edge_index.append([key, item])
                edge_index.append([item, key])
        u, v = torch.tensor([item[0] for item in edge_index]), torch.tensor([item[1] for item in edge_index])
        return dgl.graph((u, v))

    @staticmethod
    def read_data(path):
        """使用不同的方式读取原始数据以进一步处理"""
        name = osp.basename(path)
        if name == "ind.cora.test.index":
            out = np.genfromtxt(path, dtype="int64")
            return out
        else:
            out = pickle.load(open(path, "rb"), encoding="latin1")
        out = out.toarray() if hasattr(out, "toarray") else out
        return out


class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=True):
        super(GraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight)
        if self.use_bias:
            nn.init.zeros_(self.bias)

    def message_func(self, edges):
        message = torch.mm(
            edges.src["in_feature"] / torch.sqrt(edges.src['degree'] * edges.dst['degree']).reshape((-1, 1)),
            self.weight)
        if self.use_bias:
            message += self.bias
        return {"m": message}

    def forward(self, dgl_graph: dgl.graph, input_feature):
        with dgl_graph.local_scope():
            dgl_graph.ndata["in_feature"] = input_feature
            dgl_graph.ndata["degree"] = dgl_graph.in_degrees()
            dgl_graph.update_all(
                self.message_func,
                dgl.function.sum("m", "neigh_sum")
            )
            return dgl_graph.ndata["neigh_sum"]


class GCNNet(nn.Module):
    """ 定义一个包含两层GraphConvolution的模型 """

    def __init__(self, input_dim=1433):
        super(GCNNet, self).__init__()
        self.gcn1 = GraphConvolution(input_dim, 16)
        self.gcn2 = GraphConvolution(16, 7)

    def forward(self, dgl_graph, feature):
        new_feature = F.relu(self.gcn1(dgl_graph, feature))
        logits = self.gcn2(dgl_graph, new_feature)
        return logits


def accuracy(mask):
    model.eval()
    with torch.no_grad():
        logits = model(dataset.dgl_graph, tensor_x)
        test_mask_logits = logits[mask]
        predict_y = test_mask_logits.max(1)[1]
        accuracy = torch.eq(predict_y, tensor_y[mask]).float().mean()
    return accuracy


if __name__ == "__main__":
    # 超参数定义
    learning_rate = 0.01
    weight_decay = 5e-4
    epochs = 200
    # 模型定义，包括模型实例化、损失函数与优化器定义
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GCNNet().to(device)
    # 损失函数使用交叉熵
    criterion = nn.CrossEntropyLoss().to(device)
    # 优化器使用Adam
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # 加载数据，并转换为torch.Tensor
    dataset = CoraData().data
    x = dataset.x / dataset.x.sum(1, keepdims=True)
    # 归一化数据，使得每一行和为1
    tensor_x = torch.from_numpy(x).to(device)
    tensor_y = torch.from_numpy(dataset.y).to(device)
    tensor_train_mask = torch.from_numpy(dataset.trn_mask).to(device)
    tensor_val_mask = torch.from_numpy(dataset.val_mask).to(device)
    tensor_test_mask = torch.from_numpy(dataset.test_mask).to(device)

    # 训练
    loss_history = []
    val_acc_history = []
    model.train()
    train_y = tensor_y[tensor_train_mask]
    for epoch in range(epochs):
        logits = model(dataset.dgl_graph, tensor_x)
        # 前向传播
        train_mask_logits = logits[tensor_train_mask]
        # 只选择训练节点进行监督
        loss = criterion(train_mask_logits, train_y)
        # 计算损失值
        optimizer.zero_grad()
        loss.backward()
        # 反向传播计算参数的梯度
        optimizer.step()
        # 使用优化方法进行梯度更新
        train_acc = accuracy(tensor_train_mask)
        # 计算当前模型在训练集上的准确率
        val_acc = accuracy(tensor_val_mask)
        # 计算当前模型在验证集上的准确率
        # 记录训练过程中损失值和准确率的变化，用于画图
        loss_history.append(loss.item())
        val_acc_history.append(val_acc.item())
        print(
            "Epoch {:03d}: Loss {:.4f}, TrainAcc {:.4}, ValAcc {:.4f}".format(epoch, loss.item(), train_acc.item(),
                                                                              val_acc.item()))
    # 查看accuracy
    plt.figure(figsize=(8, 8))
    plt.plot(val_acc_history)
    plt.xticks([])
    plt.yticks([])
    plt.show()
