import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from metrics import pair_euclidean_distances, pair_euclidean_distances_dim3
from metrics.norm_cosine_distances import pair_norm_cosine_distances, pair_norm_cosine_distances_dim3
from utils import one_hot


class TripletLoss(nn.Module):
    def __init__(self, margin=0, num_instances=8, w=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.num_instances = num_instances
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.w = w

    def forward(self, inputs, targets):
        n, embed_dim = inputs.size()
        n_ways = int(n / self.num_instances)
        n_shots = self.num_instances

        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        # TODO The training data of cub is uneven
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max())
            dist_an.append(dist[i][mask[i] == 0].min())

        dist_ap = torch.stack(dist_ap)
        dist_an = torch.stack(dist_an)
        # Compute ranking hinge loss
        y = dist_an.data.new()
        y.resize_as_(dist_an.data)
        y.fill_(1)
        y = Variable(y)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        dist_p = torch.mean(dist_ap).item()
        dist_n = torch.mean(dist_an).item()

        protos = inputs.reshape(n_ways, n_shots, -1).mean(dim=1)
        logits = pair_euclidean_distances(inputs, protos)
        estimate = torch.argmin(logits, dim=1)

        labels = torch.arange(n_ways).unsqueeze(1).expand(n_ways, n_shots).reshape(-1).cuda()

        acc = (estimate == labels).sum() / n

        return self.w * loss, acc.item(), dist_p, dist_n

class PNLoss(nn.Module):
    def __init__(self, num_instances=8, w=1.0):
        super(PNLoss, self).__init__()
        self.num_instances = num_instances
        self.loss_func = nn.CrossEntropyLoss()
        self.w = w

    def forward(self, inputs, targets):
        batch, embed_dim = inputs.size()
        n_ways = int(batch / self.num_instances)
        n_shots = self.num_instances

        protos = inputs.reshape(n_ways, n_shots, -1).mean(dim=1)

        logits = -pair_euclidean_distances(inputs, protos)
        estimate = torch.argmax(logits, dim=1)

        labels = torch.arange(n_ways).unsqueeze(1).expand(n_ways, n_shots).reshape(-1).cuda()

        acc = (estimate == labels).sum() / batch

        loss = self.loss_func(logits, labels)

        return loss * self.w, acc.item(), 0.0, 0.0

class TripletLossNoHardMining(nn.Module):
    def __init__(self, margin=0, num_instances=8, w=1.0):
        super(TripletLossNoHardMining, self).__init__()
        self.margin = margin
        self.num_instances = num_instances
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.w = w

    def forward(self, inputs, targets):
        dist = pair_euclidean_distances(inputs, inputs)
        dist = dist.clamp(min=1e-12).sqrt()

        n = inputs.size(0)
        # # Compute pairwise distance, replace by the official when merged
        # dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        # dist = dist + dist.t()
        # dist.addmm_(1, -2, inputs, inputs.t())
        # dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        # TODO The training data of cub is uneven
        for i in range(n):
            for j in range(self.num_instances-1):
                tmp = dist[i][mask[i]]
                dist_ap.append(tmp[j+1])
                tmp = dist[i][mask[i] == 0]
                dist_an.append(tmp[j+1])
        dist_ap = torch.stack(dist_ap)
        dist_an = torch.stack(dist_an)
        # Compute ranking hinge loss
        y = dist_an.data.new()
        y.resize_as_(dist_an.data)
        y.fill_(1)
        y = Variable(y)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        prec = (dist_an.data > dist_ap.data).sum() * 1. / y.size(0)
        dist_p = torch.mean(dist_ap).item()
        dist_n = torch.mean(dist_an).item()
        return self.w * loss, prec, dist_p, dist_n

class CosineTripletLossNoHardMining(nn.Module):
    def __init__(self, margin=0, num_instances=8, w=1.0, sigma=10):
        super(CosineTripletLossNoHardMining, self).__init__()
        self.margin = margin
        self.num_instances = num_instances
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.w = w
        self.sigma = sigma

    def forward(self, inputs, targets):
        dist = pair_norm_cosine_distances(inputs, inputs)

        dist = dist.clamp(min=1e-14).sqrt() * self.sigma

        n = inputs.size(0)
        # # Compute pairwise distance, replace by the official when merged
        # dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        # dist = dist + dist.t()
        # dist.addmm_(1, -2, inputs, inputs.t())
        # dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        # TODO The training data of cub is uneven
        for i in range(n):
            for j in range(self.num_instances-1):
                tmp = dist[i][mask[i]]
                dist_ap.append(tmp[j+1])
                tmp = dist[i][mask[i] == 0]
                dist_an.append(tmp[j+1])
        dist_ap = torch.stack(dist_ap)
        dist_an = torch.stack(dist_an)
        # Compute ranking hinge loss
        y = dist_an.data.new()
        y.resize_as_(dist_an.data)
        y.fill_(1)
        y = Variable(y)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        prec = (dist_an.data > dist_ap.data).sum() * 1. / y.size(0)
        dist_p = torch.mean(dist_ap).item()
        dist_n = torch.mean(dist_an).item()
        return self.w * loss, prec, dist_p, dist_n

class TripletNoiseLoss(nn.Module):
    def __init__(self, margin=0, num_instances=8, w=1.0):
        super(TripletNoiseLoss, self).__init__()
        self.margin = margin
        self.num_instances = num_instances
        self.loss_func = TripletLossNoHardMining(margin, num_instances, w=1.0)
        self.w = w

    def forward(self, original_output, noise_output, random_times, labels):
        batch, embed_dim = original_output.size()
        n_ways = int(batch / self.num_instances)
        n_shots = self.num_instances

        original_output = original_output.reshape(n_ways, n_shots, -1)

        noise_output = noise_output.reshape(random_times, n_ways, n_shots, -1)

        proto = original_output.mean(dim=1)
        original_output = original_output.reshape(batch, -1)
        noise_proto = noise_output.mean(dim=2)
        noise_output = noise_output.reshape(random_times, batch, -1)

        logits = -pair_euclidean_distances(original_output, proto)
        noise_logits = -pair_euclidean_distances_dim3(noise_output, noise_proto)

        estimate = torch.argmax(logits, dim=1)
        noise_estimate = torch.argmax(noise_logits, dim=2)

        estimate = estimate.unsqueeze(0).expand(random_times, batch)
        similarity = (estimate == noise_estimate).sum() / float(random_times * batch)

        loss = 0.0
        for scores in noise_output:
            l, _, _, _ = self.loss_func(scores, labels)
            loss += l

        loss = loss / random_times
        return loss * self.w, similarity.item()

class CosineTripletNoiseLoss(nn.Module):
    def __init__(self, margin=0, num_instances=8, w=1.0, sigma=10.0):
        super(CosineTripletNoiseLoss, self).__init__()
        self.margin = margin
        self.num_instances = num_instances
        self.loss_func = CosineTripletLossNoHardMining(margin, num_instances, w=1.0, sigma=sigma)
        self.w = w

    def forward(self, original_output, noise_output, random_times, labels):
        batch, embed_dim = original_output.size()
        n_ways = int(batch / self.num_instances)
        n_shots = self.num_instances

        original_output = original_output.reshape(n_ways, n_shots, -1)

        noise_output = noise_output.reshape(random_times, n_ways, n_shots, -1)

        proto = F.normalize(original_output, dim=2).mean(dim=1)
        original_output = original_output.reshape(batch, -1)

        noise_proto = F.normalize(noise_output, dim=3).mean(dim=2)
        noise_output = noise_output.reshape(random_times, batch, -1)

        logits = -pair_norm_cosine_distances(original_output, proto)

        noise_logits = -pair_norm_cosine_distances_dim3(noise_output, noise_proto)

        estimate = torch.argmax(logits, dim=1)
        noise_estimate = torch.argmax(noise_logits, dim=2)

        estimate = estimate.unsqueeze(0).expand(random_times, batch)
        similarity = (estimate == noise_estimate).sum() / float(random_times * batch)

        loss = 0.0
        for scores in noise_output:
            l, _, _, _ = self.loss_func(scores, labels)
            loss += l

        loss = loss / random_times
        return loss * self.w, similarity.item()

class CosineFixNoiseLoss(nn.Module):
    def __init__(self, w=1.0):
        super(CosineFixNoiseLoss, self).__init__()
        self.w = w

    def forward(self, original_output, noise_output, random_times):
        cos = nn.CosineSimilarity(dim=1)
        l_total = 0.0
        for output in noise_output:
            loss = 1 - cos(original_output, output).mean(dim=0)
            l_total += loss
        l_total = l_total / random_times

        return l_total * self.w

class AllGreatNoiseLoss(nn.Module):
    def __init__(self, w):
        super(AllGreatNoiseLoss, self).__init__()
        self.w = w
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, logits, noise_logits, random_times, labels):
        batch, n_ways = logits.size()

        estimate = torch.argmax(logits, dim=1)
        noise_estimate = torch.argmax(noise_logits, dim=2)

        estimate = estimate.unsqueeze(0).expand(random_times, batch)
        similarity = (estimate == noise_estimate).sum() / float(random_times * batch)

        loss = 0.0
        for scores in noise_logits:
            l = self.loss_func(scores, labels)
            loss += l
        loss = loss / random_times

        return loss * self.w, similarity.item()

class CosineIncrementalMetricCFLoss(nn.Module):
    def __init__(self, w=1.0, num_instances=5):
        super(CosineIncrementalMetricCFLoss, self).__init__()
        self.w = w
        self.num_instances = num_instances
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, former_proto, former_pt_labels, feats, labels):
        batch, embed_dim = feats.size()
        n_ways = int(batch / self.num_instances)
        n_shots = self.num_instances

        former_pt_labels = former_pt_labels.cpu().numpy().tolist()

        class_labels = labels.reshape(n_ways, n_shots)[:,0]
        feats = feats.reshape(n_ways, n_shots, -1)

        labels = []
        novel_prototypes = []
        for i, class_label in enumerate(class_labels):
            class_label = class_label.item()
            if class_label in former_pt_labels:
                index = int(former_pt_labels.index(class_label))
                labels.append(index)
            else:
                index = int(len(former_pt_labels) + len(novel_prototypes))
                labels.append(index)
                proto = feats[i, :, :]
                F.normalize(proto, dim=1)
                proto = proto.mean(dim=0)
                novel_prototypes.append(proto)

        novel_prototypes = torch.stack(novel_prototypes)
        all_proto = torch.cat((former_proto, novel_prototypes), dim=0)

        labels = torch.tensor(labels).unsqueeze(1).expand(n_ways, n_shots).reshape(-1).cuda()
        feats = feats.reshape(batch, -1)

        logits = -pair_norm_cosine_distances(feats, all_proto)
        estimate = torch.argmax(logits, dim=1)

        acc = (estimate == labels).sum() / estimate.shape[0]

        loss = self.loss_func(logits, labels)

        return loss * self.w, acc.item()

class SimilarityNoiseLoss(nn.Module):
    def __init__(self, w):
        super(SimilarityNoiseLoss, self).__init__()
        self.w = w
        # Sigmoid + BCELoss
        self.loss_func = nn.BCEWithLogitsLoss()
        self.m = nn.Sigmoid()

    def forward(self, logits, noise_logits, random_times, labels):
        batch, n_ways = logits.size()

        estimate = torch.argmax(logits, dim=1)
        noise_estimate = torch.argmax(noise_logits, dim=2)

        estimate = estimate.unsqueeze(0).expand(random_times, batch)
        similarity = (estimate == noise_estimate).sum() / float(random_times * batch)

        # labels_one_hot = one_hot(labels, num_class=n_ways)
        # index = torch.tensor(range(n_ways)).unsqueeze(0).expand(logits.shape[0],
        #                                                                              -1).cuda()
        # labels_one_hot = labels_one_hot.scatter_(dim=1, index=index, src=logits)
        # temp = (labels_one_hot == logits)

        logits = self.m(logits)
        loss = 0.0
        for scores in noise_logits:
            l = self.loss_func(scores, logits)
            loss += l
        loss = loss / random_times

        return loss * self.w, similarity

class FixProtoLoss(nn.Module):
    def __init__(self, num_instances=8, w=1.0):
        super(FixProtoLoss, self).__init__()
        self.w = w
        self.num_instances = num_instances

    def forward(self, original_output, noise_output, random_times):
        batch, embed_dim = original_output.size()
        n_ways = int(batch / self.num_instances)
        n_shots = self.num_instances

        original_output = original_output.reshape(n_ways, n_shots, -1)

        noise_output = noise_output.reshape(random_times, n_ways, n_shots, -1)

        proto = original_output.mean(dim=1)
        noise_proto = noise_output.mean(dim=2)

        proto = proto.unsqueeze(dim=0).expand(random_times, n_ways, -1)

        loss = ((proto - noise_proto) ** 2).sum()

        return loss * self.w

class NoiseProtoCFLoss(nn.Module):
    def __init__(self, num_instances=8, w=1.0):
        super(NoiseProtoCFLoss, self).__init__()
        self.w = w
        self.num_instances = num_instances
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, original_output, noise_output, random_times):
        batch, embed_dim = original_output.size()
        n_ways = int(batch / self.num_instances)
        n_shots = self.num_instances

        original_output = original_output.reshape(n_ways, n_shots, -1)

        proto = original_output.mean(dim=1)
        labels = torch.arange(n_ways).unsqueeze(1).expand(n_ways, n_shots).reshape(-1).cuda()

        loss = 0.0
        for queries in noise_output:
            logits = -pair_euclidean_distances(queries, proto)
            l = self.loss_func(logits, labels)
            loss += l

        loss = loss / random_times

        return loss * self.w

class NoiseLoss(nn.Module):
    def __init__(self, w, times=1):
        super(NoiseLoss, self).__init__()
        self.w = w
        self.times = times

    def forward(self, logits, noise_logits, random_times):
        batch, n_ways = logits.size()

        estimate = torch.argmax(logits, dim=1)
        noise_estimate = torch.argmax(logits, dim=2)

        estimate = estimate.unsqueeze(0).expand(random_times, batch)
        similarity = (estimate == noise_estimate).sum() / float(random_times * batch)

        S1 = nn.Softmax(dim=1)
        S2 = nn.Softmax(dim=2)

        logits_labels = torch.arange(n_ways).repeat(batch, 1).cuda()
        soft_estimate = (S1(logits * self.times) * logits_labels).sum(dim=1)

        noise_logits_labels = torch.arange(n_ways).repeat(random_times, batch, 1).cuda()
        noise_soft_estimate = (S2(noise_logits * self.times) * noise_logits_labels).sum(dim=2)

        soft_estimate = soft_estimate.unsqueeze(0).expand(random_times, batch)
        loss = ((noise_soft_estimate - soft_estimate) ** 2).sum()

        return loss * self.w

class MetricNoiseLoss(nn.Module):
    def __init__(self, w):
        super(MetricNoiseLoss, self).__init__()
        self.w = w

    def forward(self, original_output, noise_output, n_shots, n_ways, random_times):
        original_output = original_output.reshape(n_ways, n_shots, -1)

        noise_output = noise_output.reshape(random_times, n_ways, n_shots, -1)

        proto = original_output.mean(dim=1)
        original_output = original_output.reshape(n_ways * n_shots, -1)
        noise_proto = noise_output.mean(dim=2)
        noise_output = noise_output.reshape(random_times, n_ways * n_shots, -1)

        logits = -pair_euclidean_distances(original_output, proto)
        noise_logits = -pair_euclidean_distances_dim3(noise_output, noise_proto)

        estimate = torch.argmax(logits, dim=1)
        noise_estimate = torch.argmax(noise_logits, dim=2)

        S1 = nn.Softmax(dim=1)
        S2 = nn.Softmax(dim=2)

        logits_labels = torch.arange(n_ways).repeat(n_shots * n_ways, 1).cuda()
        soft_estimate = (S1(logits) * logits_labels).sum(dim=1)
        noise_logits_labels = torch.arange(n_ways).repeat(random_times, n_shots * n_ways, 1).cuda()
        noise_soft_estimate = (S2(noise_logits) * noise_logits_labels).sum(dim=2)

        soft_estimate = soft_estimate.unsqueeze(0).expand(random_times, n_shots * n_ways)
        loss = ((noise_soft_estimate - soft_estimate) ** 2).sum()

        return loss * self.w

class IncrementalMetricCFLoss(nn.Module):
    def __init__(self, w=1.0, num_instances=5):
        super(IncrementalMetricCFLoss, self).__init__()
        self.w = w
        self.num_instances = num_instances
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, former_proto, former_pt_labels, feats, labels):
        batch, embed_dim = feats.size()
        n_ways = int(batch / self.num_instances)
        n_shots = self.num_instances

        former_pt_labels = former_pt_labels.cpu().numpy().tolist()

        class_labels = labels.reshape(n_ways, n_shots)[:,0]
        feats = feats.reshape(n_ways, n_shots, -1)

        labels = []
        novel_prototypes = []
        for i, class_label in enumerate(class_labels):
            class_label = class_label.item()
            if class_label in former_pt_labels:
                index = int(former_pt_labels.index(class_label))
                labels.append(index)
            else:
                index = int(len(former_pt_labels) + len(novel_prototypes))
                labels.append(index)
                proto = feats[i, :, :]
                proto = proto.mean(dim=0)
                novel_prototypes.append(proto)

        novel_prototypes = torch.stack(novel_prototypes)
        all_proto = torch.cat((former_proto, novel_prototypes), dim=0)

        labels = torch.tensor(labels).unsqueeze(1).expand(n_ways, n_shots).reshape(-1).cuda()
        feats = feats.reshape(batch, -1)

        logits = -pair_euclidean_distances(feats, all_proto)
        estimate = torch.argmax(logits, dim=1)

        acc = (estimate == labels).sum() / estimate.shape[0]

        loss = self.loss_func(logits, labels)

        return loss * self.w, acc.item()

class ProtoCFLoss(nn.Module):
    def __init__(self, w, norm = -10):
        super(ProtoCFLoss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.w = w
        self.norm = norm

    def forward(self, x, tf_prototypes, labels):
        tf_prototypes_0 = tf_prototypes[0]
        tf_prototypes_1 = tf_prototypes[1]
        dis_0 = pair_euclidean_distances(x, tf_prototypes_0)
        dis_1 = pair_euclidean_distances(x, tf_prototypes_1)

        dis_0_min = torch.min(dis_0, dim=1)
        dis_1_min = torch.min(dis_1, dim=1)

        dis_0 = -torch.norm(dis_0, p=self.norm, dim=1)
        dis_1 = -torch.norm(dis_1, p=self.norm, dim=1)

        logits = torch.stack([dis_0, dis_1], dim=1)

        return self.w * self.cross_entropy(logits, labels)

class ProtoCFLoss2(nn.Module):
    def __init__(self, w):
        super(ProtoCFLoss2, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.w = w

    def forward(self, x, tf_prototypes, labels):
        tf_prototypes_0 = tf_prototypes[0]
        tf_prototypes_1 = tf_prototypes[1]
        dis_0 = pair_euclidean_distances(x, tf_prototypes_0)
        dis_1 = pair_euclidean_distances(x, tf_prototypes_1)

        dis_0_min = torch.min(dis_0, dim=1)
        dis_1_min = torch.min(dis_1, dim=1)

        dis_0 = -torch.mean(dis_0, dim=1)
        dis_1 = -torch.norm(dis_1, p=-5, dim=1)

        logits = torch.stack([dis_0, dis_1], dim=1)

        return self.w * self.cross_entropy(logits, labels)

class ProtoRegularization(nn.Module):
    def __init__(self, w=1.0):
        super(ProtoRegularization, self).__init__()
        self.w = w

    def forward(self, former_proto_list, former_proto_label, tf_feat_list, tf_label_list):
        proto_former = []
        proto_current = []
        for i, label in enumerate(former_proto_label):
            index_list = (tf_label_list == label.item()).nonzero().squeeze()
            tf_feat = tf_feat_list[index_list]
            if len(tf_feat) != 0:
                proto_current.append(tf_feat.mean(dim=0))
                proto_former.append(former_proto_list[i])
        proto_former = torch.stack(proto_former)
        proto_current = torch.stack(proto_current)
        return self.w * ((proto_former - proto_current) ** 2).mean()

class LessForget(nn.Module):
    """
    Less-Forget Constraint Loss

    Args:
        lmbda (Float): the weight value
    """
    def __init__(self, lmbda):
        super(LessForget, self).__init__()
        self.lmbda = lmbda

    def forward(self, former_feat, current_feat):
        """
        Args:
            former_feat (torch.Tensor): the features calculated by former model. The shape is (batch_size, embed_dim)
            current_feat (torch.Tensor): the features calculated by fine_tune model. The shape is (batch_size, embed_dim)
        """
        loss = nn.CosineEmbeddingLoss()(former_feat, current_feat, torch.ones(current_feat.shape[0]).cuda())
        return loss * self.lmbda

class InterClassSeparation(nn.Module):
    """
    Inter-Class Separation Loss

    Args:
        K (Int): top-K scores for novel classes
        weight (Float): the weight value
        margin (Float): the margin for margin ranking loss
    """
    def __init__(self, K, weight, margin):
        super(InterClassSeparation, self).__init__()
        self.K = K
        self.weight = weight
        self.margin = margin

    def forward(self, scores, labels, num_old_classes):
        """
        Args:
            scores (torch.Tensor): the output scores with shape (batch_size, num_classes)
            labels (torch.Tensor): the softmax labels of input data with shape (batch_size,)
            num_old_classes (Int): the number of seen classes
        """
        gt_index = torch.zeros(scores.size()).cuda()
        gt_index = gt_index.scatter(1, labels.view(-1, 1), 1).ge(0.5)
        gt_scores = scores.masked_select(gt_index)

        max_novel_scores = scores[:, num_old_classes:].topk(self.K, dim=1)[0]

        hard_index = labels.lt(num_old_classes)
        hard_num = torch.nonzero(hard_index).size(0)
        if hard_num > 0:
            gt_scores = gt_scores[hard_index].reshape(-1, 1).repeat(1, self.K)
            max_novel_scores = max_novel_scores[hard_index]
            assert (gt_scores.size() == max_novel_scores.size())
            assert (gt_scores.size(0) == hard_num)
            MRL = nn.MarginRankingLoss(margin=self.margin)
            loss = MRL(gt_scores.reshape(-1, 1), max_novel_scores.reshape(-1, 1),
                       torch.ones(hard_num * self.K).cuda())
            loss = loss * self.weight
        else:
            loss = torch.zeros(1).cuda()

        return loss