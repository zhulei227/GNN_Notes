import numpy as np
import torch
import dgl
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import HeteroGraphConv, GraphConv

n_users = 1000
n_items = 500
n_follows = 3000
n_clicks = 5000
n_dislikes = 500
n_hetero_features = 10
n_user_classes = 5
n_max_clicks = 10

follow_src = np.random.randint(0, n_users, n_follows)
follow_dst = np.random.randint(0, n_users, n_follows)
click_src = np.random.randint(0, n_users, n_clicks)
click_dst = np.random.randint(0, n_items, n_clicks)
dislike_src = np.random.randint(0, n_users, n_dislikes)
dislike_dst = np.random.randint(0, n_items, n_dislikes)

hetero_graph = dgl.heterograph({
    ('user', 'follow', 'user'): (follow_src, follow_dst),
    ('user', 'followed-by', 'user'): (follow_dst, follow_src),
    ('user', 'click', 'item'): (click_src, click_dst),
    ('item', 'clicked-by', 'user'): (click_dst, click_src),
    ('user', 'dislike', 'item'): (dislike_src, dislike_dst),
    ('item', 'disliked-by', 'user'): (dislike_dst, dislike_src)})

hetero_graph.nodes['user'].data['feature'] = torch.randn(n_users, n_hetero_features)
hetero_graph.nodes['item'].data['feature'] = torch.randn(n_items, n_hetero_features)
hetero_graph.nodes['user'].data['label'] = torch.randint(0, n_user_classes, (n_users,))
hetero_graph.edges['click'].data['label'] = torch.randint(1, n_max_clicks, (n_clicks,)).float()
# 在user类型的节点和click类型的边上随机生成训练集的掩码
hetero_graph.nodes['user'].data['train_mask'] = torch.zeros(n_users, dtype=torch.bool).bernoulli(0.6)
hetero_graph.edges['click'].data['train_mask'] = torch.zeros(n_clicks, dtype=torch.bool).bernoulli(0.6)


# Define a Heterograph Conv model


class RGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names):
        super().__init__()
        # 实例化HeteroGraphConv，in_feats是输入特征的维度，out_feats是输出特征的维度，aggregate是聚合函数的类型
        self.conv1 = HeteroGraphConv({
            rel: GraphConv(in_feats, hid_feats)
            for rel in rel_names}, aggregate='sum')
        self.conv2 = HeteroGraphConv({
            rel: GraphConv(hid_feats, out_feats)
            for rel in rel_names}, aggregate='sum')

    def forward(self, blocks, inputs):
        # 输入是节点的特征字典
        h = self.conv1(blocks[0], inputs)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv2(blocks[1], h)
        return h


model = RGCN(n_hetero_features, 20, n_user_classes, hetero_graph.etypes)
user_feats = hetero_graph.nodes['user'].data['feature']
item_feats = hetero_graph.nodes['item'].data['feature']
labels = hetero_graph.nodes['user'].data['label']
train_mask = hetero_graph.nodes['user'].data['train_mask']

sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
train_idx = torch.nonzero(train_mask, as_tuple=False).squeeze()
dataloader = dgl.dataloading.NodeDataLoader(hetero_graph, {"user": train_idx}, sampler, batch_size=128, shuffle=True,num_workers=0)


opt = torch.optim.Adam(model.parameters())

if __name__ == '__main__':
    model.train()
    for _ in range(80):
        for input_nodes, output_nodes, blocks in dataloader:
            # blocks = [b.to(torch.device('cuda')) for b in blocks]
            input_features = blocks[0].srcdata['feature']  # returns a dict
            output_labels = blocks[-1].dstdata['label']  # returns a dict
            output_predictions = model(blocks, input_features)
            loss = F.cross_entropy(output_predictions['user'], output_labels['user'])
            opt.zero_grad()
            loss.backward()
            opt.step()
            print(loss.item())