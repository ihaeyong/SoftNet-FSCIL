import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from metrics import pair_euclidean_distances, pair_euclidean_distances_dim3, norm_cosine_distances
from metrics.norm_cosine_distances import pair_norm_cosine_distances, pair_norm_cosine_distances_dim3
from .losses import TripletLossNoHardMining, CosineTripletLossNoHardMining, BufferCenterLoss, \
    CosineBufferCenterLoss, SimilarBufferCenterLoss, CosineSimilarBufferCenterLoss
from utils import one_hot, Averager

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

        if self.log is not None:
            self.log['Similarity'] = similarity.item()
            self.log['CTriNLoss'] = loss.item() * self.w

        return loss * self.w

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
            l, _ = self.loss_func(scores, labels)
            loss += l

        loss = loss / random_times

        log = {}
        log['Similarity'] = similarity.item()
        log['CosTriNLoss'] = loss.item() * self.w

        return loss * self.w, log

class CosineFixNoiseLoss(nn.Module):
    def __init__(self, w=1.0):
        super(CosineFixNoiseLoss, self).__init__()
        self.w = w

    def forward(self, original_output, noise_output, random_times):
        l_total = 0.0
        for output in noise_output:
            loss = norm_cosine_distances(original_output, output)
            loss = loss.mean(dim=0)
            l_total += loss
        l_total = l_total / random_times

        log = {'CosFixNLoss': l_total.item()}

        return l_total * self.w, log

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

        log = {}
        log['Similarity'] = similarity.item()
        log['AllGreatNLoss'] = loss.item()

        return loss * self.w, log

class NoisePTFixLoss(nn.Module):
    def __init__(self, w):
        super(NoisePTFixLoss, self).__init__()
        self.w = w
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, prototypes, pt_labels, embeddings, noise_embedding, labels):
        log = {}

        pt_labels_dict = {pt_label.item(): i  for i, pt_label in enumerate(pt_labels)}
        labels = [pt_labels_dict[label.item()] for label in labels]
        labels = torch.tensor(labels).cuda()

        logits_original = -pair_euclidean_distances(embeddings, prototypes)
        estimate = logits_original.argmax(dim=1)
        acc = (estimate == labels).sum() / labels.shape[0]
        log['PNAcc'] = acc.item()

        loss = self.loss_func(logits_original, labels)
        log['PNLoss'] = loss.item()

        noise_loss = Averager()
        noise_acc = Averager()
        for queries in noise_embedding:
            logits = -pair_euclidean_distances(queries, prototypes)
            l = self.loss_func(logits, labels)
            loss += l
            noise_loss.add(l.item())

            estimate = logits.argmax(dim=1)
            acc = (estimate == labels).sum() / labels.shape[0]
            noise_acc.add(acc.item())

        loss = loss / (noise_embedding.shape[0] + 1)

        log['PTFixLoss'] = loss.item()
        log['NoisePTLoss'] = noise_loss.item()
        log['NoiseAcc'] = noise_acc.item()

        return loss * self.w, log

class NoiseProtoFixLoss(nn.Module):
    """
    Ensure the prototypes will not change after applying noise
    """
    def __init__(self, w, shots):
        super(NoiseProtoFixLoss, self).__init__()
        self.w = w
        self.n_shots = shots

    def forward(self, embedding, noise_embeddings, random_times):
        n_batch, _ = embedding.size()
        n_ways = int (n_batch / self.n_shots)

        prototypes = embedding.reshape(n_ways, self.n_shots, -1).mean(dim=1)

        loss = 0.0
        for embed in noise_embeddings:
            noise_prototypes = embed.reshape(n_ways, self.n_shots, -1).mean(dim=1)
            l = ((prototypes - noise_prototypes) ** 2).sum(dim=1).mean()
            loss += l

        loss = loss / random_times

        log = {'NoiseProtoFixLoss': loss.item()}

        return loss.item() * self.w, log


class NoiseBufferCenterLoss(nn.Module):
    def __init__(self, buffer_size, w=1.0):
        super(NoiseBufferCenterLoss, self).__init__()
        self.w = w
        self.buffer_size = buffer_size
        self.loss_func = BufferCenterLoss(buffer_size=buffer_size)

    def forward(self, noise_embed, noise_buffer_embed, labels, buffer_labels):
        random_times = noise_embed.shape[0]

        log = {'BFCLoss': 0.0, 'norm': 0.0, 'dis_pt': 0.0}
        l_total = 0.0
        for i in range(random_times):
            embed = noise_embed[i]
            buffer_embed = noise_buffer_embed[i]
            loss, lg = self.loss_func(embed, buffer_embed, labels, buffer_labels)
            l_total += loss
            for k, v in lg.items():
                log[k] += v

        l_total = l_total / random_times
        for k in log.keys():
            log[k] = log[k] / random_times

        log = {'Noise_' + k: v for k, v in log.items()}

        return l_total * self.w, log

class NoiseSimilarBufferCenterLoss(nn.Module):
    def __init__(self, buffer_size, n_shots, w=1.0):
        super(NoiseSimilarBufferCenterLoss, self).__init__()
        self.w = w
        self.n_shots = n_shots
        self.buffer_size = buffer_size
        self.loss_func = SimilarBufferCenterLoss(buffer_size=buffer_size, n_shots=n_shots)

    def forward(self, noise_embed, noise_buffer_embed, labels, buffer_labels):
        random_times = noise_embed.shape[0]

        log = {'SBFCLoss': 0.0, 'norm': 0.0, 'dis_pt': 0.0}
        l_total = 0.0
        for i in range(random_times):
            embed = noise_embed[i]
            buffer_embed = noise_buffer_embed[i]
            loss, lg = self.loss_func(embed, buffer_embed, labels, buffer_labels)
            l_total += loss
            for k, v in lg.items():
                log[k] += v

        l_total = l_total / random_times
        for k in log.keys():
            log[k] = log[k] / random_times

        log = {'Noise_' + k: v for k, v in log.items()}

        return l_total * self.w, log

class NoiseCosBufferCenterLoss(nn.Module):
    def __init__(self, buffer_size, w=1.0):
        super(NoiseCosBufferCenterLoss, self).__init__()
        self.w = w
        self.buffer_size = buffer_size
        self.loss_func = CosineBufferCenterLoss(buffer_size=buffer_size)

    def forward(self, noise_embed, noise_buffer_embed, labels, buffer_labels):
        random_times = noise_embed.shape[0]

        log = {'CosBFCLoss': 0.0, 'norm': 0.0, 'dis_pt': 0.0}
        l_total = 0.0
        for i in range(random_times):
            embed = noise_embed[i]
            buffer_embed = noise_buffer_embed[i]
            loss, lg = self.loss_func(embed, buffer_embed, labels, buffer_labels)
            l_total += loss
            for k, v in lg.items():
                log[k] += v

        l_total = l_total / random_times
        for k in log.keys():
            log[k] = log[k] / random_times

        log = {'Noise_' + k: v for k, v in log.items()}

        return l_total * self.w, log

class NoiseCosSimilarBufferCenterLoss(nn.Module):
    def __init__(self, buffer_size, n_shots, w=1.0):
        super(NoiseCosSimilarBufferCenterLoss, self).__init__()
        self.w = w
        self.buffer_size = buffer_size
        self.n_shots = n_shots
        self.loss_func = CosineSimilarBufferCenterLoss(buffer_size=buffer_size, n_shots=n_shots)

    def forward(self, noise_embed, noise_buffer_embed, labels, buffer_labels):
        random_times = noise_embed.shape[0]

        log = {'SBFCLoss': 0.0, 'dis_pt': 0.0}
        l_total = 0.0
        for i in range(random_times):
            embed = noise_embed[i]
            buffer_embed = noise_buffer_embed[i]
            loss, lg = self.loss_func(embed, buffer_embed, labels, buffer_labels)
            l_total += loss
            for k, v in lg.items():
                log[k] += v

        l_total = l_total / random_times
        for k in log.keys():
            log[k] = log[k] / random_times

        log = {'Noise_' + k: v for k, v in log.items()}

        return l_total * self.w, log

# class SimilarityNoiseLoss(nn.Module):
#     def __init__(self, w):
#         super(SimilarityNoiseLoss, self).__init__()
#         self.w = w
#         # Sigmoid + BCELoss
#         self.loss_func = nn.BCEWithLogitsLoss()
#         self.m = nn.Sigmoid()
#
#     def forward(self, logits, noise_logits, random_times, labels):
#         batch, n_ways = logits.size()
#
#         estimate = torch.argmax(logits, dim=1)
#         noise_estimate = torch.argmax(noise_logits, dim=2)
#
#         estimate = estimate.unsqueeze(0).expand(random_times, batch)
#         similarity = (estimate == noise_estimate).sum() / float(random_times * batch)
#
#         # labels_one_hot = one_hot(labels, num_class=n_ways)
#         # index = torch.tensor(range(n_ways)).unsqueeze(0).expand(logits.shape[0],
#         #                                                                              -1).cuda()
#         # labels_one_hot = labels_one_hot.scatter_(dim=1, index=index, src=logits)
#         # temp = (labels_one_hot == logits)
#
#         logits = self.m(logits)
#         loss = 0.0
#         for scores in noise_logits:
#             l = self.loss_func(scores, logits)
#             loss += l
#         loss = loss / random_times
#
#         return loss * self.w, similarity

# class FixProtoLoss(nn.Module):
#     def __init__(self, num_instances=8, w=1.0):
#         super(FixProtoLoss, self).__init__()
#         self.w = w
#         self.num_instances = num_instances
#
#     def forward(self, original_output, noise_output, random_times):
#         batch, embed_dim = original_output.size()
#         n_ways = int(batch / self.num_instances)
#         n_shots = self.num_instances
#
#         original_output = original_output.reshape(n_ways, n_shots, -1)
#
#         noise_output = noise_output.reshape(random_times, n_ways, n_shots, -1)
#
#         proto = original_output.mean(dim=1)
#         noise_proto = noise_output.mean(dim=2)
#
#         proto = proto.unsqueeze(dim=0).expand(random_times, n_ways, -1)
#
#         loss = ((proto - noise_proto) ** 2).sum()
#
#         return loss * self.w

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

        log = {'NProtoCFLoss': loss.item() * self.w}

        return loss * self.w, log

# class NoiseLoss(nn.Module):
#     def __init__(self, w, times=1):
#         super(NoiseLoss, self).__init__()
#         self.w = w
#         self.times = times
#
#     def forward(self, logits, noise_logits, random_times):
#         batch, n_ways = logits.size()
#
#         estimate = torch.argmax(logits, dim=1)
#         noise_estimate = torch.argmax(logits, dim=2)
#
#         estimate = estimate.unsqueeze(0).expand(random_times, batch)
#         similarity = (estimate == noise_estimate).sum() / float(random_times * batch)
#
#         S1 = nn.Softmax(dim=1)
#         S2 = nn.Softmax(dim=2)
#
#         logits_labels = torch.arange(n_ways).repeat(batch, 1).cuda()
#         soft_estimate = (S1(logits * self.times) * logits_labels).sum(dim=1)
#
#         noise_logits_labels = torch.arange(n_ways).repeat(random_times, batch, 1).cuda()
#         noise_soft_estimate = (S2(noise_logits * self.times) * noise_logits_labels).sum(dim=2)
#
#         soft_estimate = soft_estimate.unsqueeze(0).expand(random_times, batch)
#         loss = ((noise_soft_estimate - soft_estimate) ** 2).sum()
#
#         return loss * self.w

# class MetricNoiseLoss(nn.Module):
#     def __init__(self, w):
#         super(MetricNoiseLoss, self).__init__()
#         self.w = w
#
#     def forward(self, original_output, noise_output, n_shots, n_ways, random_times):
#         original_output = original_output.reshape(n_ways, n_shots, -1)
#
#         noise_output = noise_output.reshape(random_times, n_ways, n_shots, -1)
#
#         proto = original_output.mean(dim=1)
#         original_output = original_output.reshape(n_ways * n_shots, -1)
#         noise_proto = noise_output.mean(dim=2)
#         noise_output = noise_output.reshape(random_times, n_ways * n_shots, -1)
#
#         logits = -pair_euclidean_distances(original_output, proto)
#         noise_logits = -pair_euclidean_distances_dim3(noise_output, noise_proto)
#
#         estimate = torch.argmax(logits, dim=1)
#         noise_estimate = torch.argmax(noise_logits, dim=2)
#
#         S1 = nn.Softmax(dim=1)
#         S2 = nn.Softmax(dim=2)
#
#         logits_labels = torch.arange(n_ways).repeat(n_shots * n_ways, 1).cuda()
#         soft_estimate = (S1(logits) * logits_labels).sum(dim=1)
#         noise_logits_labels = torch.arange(n_ways).repeat(random_times, n_shots * n_ways, 1).cuda()
#         noise_soft_estimate = (S2(noise_logits) * noise_logits_labels).sum(dim=2)
#
#         soft_estimate = soft_estimate.unsqueeze(0).expand(random_times, n_shots * n_ways)
#         loss = ((noise_soft_estimate - soft_estimate) ** 2).sum()
#
#         return loss * self.w

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

        log = {'IL-CosMetricCFLoss': loss.item() * self.w, 'Acc': acc.item()}

        return loss * self.w, log

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

        log = {'IL-MetricCFLoss': loss.item() * self.w, 'Acc': acc.item()}

        return loss * self.w, log

class CosineLessForget(nn.Module):
    def __init__(self, w):
        super(CosineLessForget, self).__init__()
        self.w = w

    def forward(self, former_feats, feats, labels, former_classes):
        former_classes = former_classes.cpu().numpy().tolist()
        labels = labels.cpu().numpy().tolist()
        mask = [label in former_classes for label in labels]
        mask = torch.tensor(mask)

        former_feats = former_feats[mask]
        feats = feats[mask]

        cos = nn.CosineSimilarity(dim=1)

        loss = 1.0 - cos(former_feats, feats).mean(dim=0)

        log = {'CosineLessForget': loss.item() * self.w}

        return loss * self.w, log

class CosineNoiseProtoCFLoss(nn.Module):
    def __init__(self, num_instances=8, w=1.0):
        super(CosineNoiseProtoCFLoss, self).__init__()
        self.w = w
        self.num_instances = num_instances
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, original_output, noise_output, random_times):
        batch, embed_dim = original_output.size()
        n_ways = int(batch / self.num_instances)
        n_shots = self.num_instances

        original_output = original_output.reshape(n_ways, n_shots, -1)

        proto = F.normalize(original_output, dim=2).mean(dim=1)

        labels = torch.arange(n_ways).unsqueeze(1).expand(n_ways, n_shots).reshape(-1).cuda()

        loss = 0.0
        for queries in noise_output:
            logits = -pair_norm_cosine_distances(queries, proto)
            l = self.loss_func(logits, labels)
            loss += l

        loss = loss / random_times

        log = {'CosNProtoCFLoss': loss.item() * self.w}

        return loss * self.w, log

