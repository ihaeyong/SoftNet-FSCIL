import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from metrics import pair_euclidean_distances, pair_euclidean_distances_dim3, norm_cosine_distances
from metrics.norm_cosine_distances import pair_norm_cosine_distances, pair_norm_cosine_distances_dim3, pair_norm_pearson_distances, pair_norm_cosine_regs
from metrics.norm_cosine_distances import pair_dot_product
from utils import one_hot, pnorm

class BufferCenterLoss(nn.Module):
    def __init__(self, buffer_size, w=1.0):
        super(BufferCenterLoss, self).__init__()
        self.w = w
        self.buffer_size = buffer_size

    def forward(self, embed, buffer_embed, labels, buffer_labels):
        class_labels = torch.unique_consecutive(buffer_labels)
        n_ways = class_labels.shape[0]
        n_shots = self.buffer_size

        # (n_ways, embed_dim)
        protos = buffer_embed.reshape(n_ways, n_shots, -1).mean(dim=1)
        dis_pt = pair_euclidean_distances(protos, protos)
        dis_pt = dis_pt.sum() / (n_ways * (n_ways - 1))

        embed_all = torch.cat((embed, buffer_embed), dim=0)
        norm = torch.norm(embed_all, dim=1).mean()

        labels_all = torch.cat((labels, buffer_labels), dim=0)
        batch_size, embed_dim = embed_all.size()

        distances = pair_euclidean_distances(embed_all, protos)

        labels_all = labels_all.unsqueeze(1).expand(batch_size, n_ways)
        class_labels = class_labels.unsqueeze(0).expand(batch_size, n_ways)
        mask = labels_all.eq(class_labels).float()

        distances = distances * mask

        loss = distances.clamp(min=1e-12, max=1e+12).sum() / batch_size

        log = {'BFCLoss': loss.item(), 'norm': norm, 'dis_pt': dis_pt.item()}

        return loss * self.w, log

class CenterLoss(nn.Module):
    def __init__(self, w=1.0):
        super(CenterLoss, self).__init__()
        self.w = w

    def forward(self, embed, labels, protos, class_labels):
        batch_size, _ = embed.size()
        n_ways = class_labels.shape[0]

        norm = torch.norm(embed, dim=1).mean()

        distances = pair_euclidean_distances(embed, protos)

        # labels = torch.arange(n_ways).unsqueeze(1).expand(n_ways, n_shots).reshape(
        #     -1).cuda()

        labels = labels.unsqueeze(1).expand(batch_size, n_ways)
        class_labels = class_labels.unsqueeze(0).expand(batch_size, n_ways)
        mask = labels.eq(class_labels).float()
        all = mask.sum()
        distances = distances * mask

        loss = distances.clamp(min=1e-12, max=1e+12).sum() / batch_size

        log = {'CenterLoss': loss.item(), 'norm': norm}

        return loss * self.w, log

class ProtoFixLoss(nn.Module):
    """
    Ensure the prototypes will not change after applying noise
    """
    def __init__(self, w, shots):
        super(ProtoFixLoss, self).__init__()
        self.w = w
        self.n_shots = shots

    def forward(self, embedding, noise_embeddings):
        n_batch, _ = embedding.size()
        n_ways = int (n_batch / self.n_shots)

        prototypes = embedding.reshape(n_ways, self.n_shots, -1).mean(dim=1)


        noise_prototypes = noise_embeddings.reshape(n_ways, self.n_shots, -1).mean(dim=1)
        loss = ((prototypes - noise_prototypes) ** 2).sum(dim=1).mean()

        log = {'ProtoFixLoss': loss.item()}

        return loss.item() * self.w, log

class SimilarBufferCenterLoss(nn.Module):
    def __init__(self, buffer_size, n_shots, w=1.0):
        super(SimilarBufferCenterLoss, self).__init__()
        self.w = w
        self.buffer_size = buffer_size
        self.n_shots = n_shots

    def forward(self, embed, buffer_embed, labels, buffer_labels):
        class_labels = torch.unique_consecutive(buffer_labels)
        n_total_ways = class_labels.shape[0]

        # (n_ways, embed_dim)
        protos = buffer_embed.reshape(n_total_ways, self.buffer_size, -1).mean(dim=1)

        dis_pt = pair_euclidean_distances(protos, protos)
        dis_pt = dis_pt.sum() / (n_total_ways * (n_total_ways - 1))

        embed_all = torch.cat((embed, buffer_embed), dim=0)
        norm = torch.norm(embed_all, dim=1).mean()

        batch_size, _ = embed.size()
        n_ways = int(batch_size / self.n_shots)

        accurate_protos = embed.reshape(n_ways, self.n_shots, -1).mean(dim=1)
        labels = torch.unique_consecutive(labels)

        pt_index_dict = {class_label.item(): i for i, class_label in enumerate(class_labels)}
        estimate_protos = torch.stack([protos[pt_index_dict[label.item()]] for label in labels])

        loss = ((accurate_protos - estimate_protos) ** 2).sum(dim=1).mean()

        log = {'SBFCLoss': loss.item(), 'norm': norm, 'dis_pt': dis_pt.item()}

        return loss * self.w, log

class PrototypesCenterLoss(nn.Module):
    def __init__(self, w):
        super(PrototypesCenterLoss, self).__init__()
        self.w = w

    def forward(self, prototypes, pt_labels, embeddings, labels):
        log = {}

        pt_labels_dict = {pt_label.item(): i for i, pt_label in enumerate(pt_labels)}
        pair_prototypes = [prototypes[pt_labels_dict[label.item()]] for label in labels]
        pair_prototypes = torch.stack(pair_prototypes)

        logits = ((pair_prototypes - embeddings)**2).sum(dim=1)
        loss = logits.mean()

        log['PTCenterLoss'] = loss.item()

        return loss * self.w, log

class PTFixPNLoss(nn.Module):
    def __init__(self, w, use_cosine=False):
        super(PTFixPNLoss, self).__init__()
        self.w  = w
        self.use_cosine = use_cosine
        # TODO support cosine similarity
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, prototypes, pt_labels, embeddings, labels):
        log = {}

        pt_labels_dict = {pt_label.item(): i for i, pt_label in enumerate(pt_labels)}
        labels = [pt_labels_dict[label.item()] for label in labels]
        labels = torch.tensor(labels).cuda()

        logits_original = -pair_euclidean_distances(embeddings, prototypes)
        estimate = logits_original.argmax(dim=1)
        acc = (estimate == labels).sum() / labels.shape[0]
        log['PNAcc'] = acc.item()

        loss = self.loss_func(logits_original, labels)
        log['PNLoss'] = loss.item()

        return loss * self.w, log

class IncrementalPTFixPNLoss(nn.Module):
    def __init__(self, w, omega, n_shots, use_cosine = False):
        super(IncrementalPTFixPNLoss, self).__init__()
        self.w = w
        self.omega = omega
        self.n_shots = n_shots
        self.use_cosine = use_cosine
        self.loss_func = nn.CrossEntropyLoss()
        self.topk = 3

    def forward(self, prototypes, pt_labels, output, labels):
        n_batch, _ = output.size()
        n_ways = int (n_batch / self.n_shots)

        # output : [25, 64]
        if self.use_cosine:
            output = F.normalize(output, dim=1)
        novel_prototypes = output.reshape(n_ways, self.n_shots, -1).mean(dim=1)

        # label : [0000011111222223333344444]
        labels = torch.arange(n_ways).unsqueeze(1).expand(n_ways, self.n_shots).reshape(
            -1).cuda()

        pro_labels = torch.arange(prototypes.size(0)).reshape(-1).cuda() + labels.max()

        # [65, 64]
        prototypes_all = torch.cat((novel_prototypes, prototypes), dim=0)

        if self.use_cosine:
            logits = - pair_norm_cosine_distances(output, prototypes_all) * self.omega
        else:
            # [25, 65]
            logits = - pair_euclidean_distances(output, prototypes_all) * self.omega


        estimate = logits.argmax(dim=1)

        acc = (estimate == labels).sum() / output.shape[0]

        loss = self.loss_func(logits, labels)

        log = {'PTFixPNLoss': loss.item(), 'Acc': acc.item()}

        return loss * self.w, log


class IncrementalKPTFixPNLoss(nn.Module):
    def __init__(self, w, omega, n_shots, use_cosine = False):
        super(IncrementalKPTFixPNLoss, self).__init__()
        self.w = w
        self.omega = omega
        self.n_shots = n_shots
        self.use_cosine = use_cosine
        self.loss_func = nn.CrossEntropyLoss()
        self.topk = 3

    def forward(self, prototypes, pt_labels, output, labels):
        n_batch, _ = output.size()
        n_ways = int (n_batch / self.n_shots)

        # ordered losses
        ordered, indices = torch.sort(labels)
        output = output[indices,:]
        labels = labels[indices]

        # output : [25, 64]
        if self.use_cosine:
            output = F.normalize(output, dim=1)
        novel_prototypes = output.reshape(n_ways, self.n_shots, -1).mean(dim=1)

        # label : [0000011111222223333344444]
        labels = torch.arange(n_ways).unsqueeze(1).expand(n_ways, self.n_shots).reshape(
            -1).cuda()

        pro_labels = torch.arange(prototypes.size(0)).reshape(-1).cuda() + labels.max()

        # [65, 64]
        prototypes_all = torch.cat((novel_prototypes, prototypes), dim=0)

        if self.use_cosine:
            logits = - pair_norm_cosine_distances(output, prototypes_all) * self.omega
        else:
            # [25, 65]
            logits = - pair_euclidean_distances(output, prototypes_all) * self.omega


        estimate = logits.argmax(dim=1)

        acc = (estimate == labels).sum() / output.shape[0]

        loss = self.loss_func(logits, labels)

        log = {'PTFixPNLoss': loss.item(), 'Acc': acc.item()}

        return loss * self.w, log


def supcon_loss_func(output1, output2, labels, tau=1.0):
    """Estimate the distances between output1 and output2

    Args:
        output1 (a * m Tensor)
        output2 (b * m Tensor)
    Returns:
        pair eculidean distances (a * b Tensor)
    """
    a = output1.shape[0]
    b = output2.shape[0]

    output1 = output1.unsqueeze(1).expand(a, b, -1)
    output2 = output2.unsqueeze(0).expand(a, b, -1)

    cos = nn.CosineSimilarity(dim=2)
    logits = cos(output1, output2) / tau
    sim = torch.exp(logits)

    pos_mask = torch.zeros_like(logits, dtype=torch.uint8)
    pos_mask.scatter_(1, labels.data.view(-1,1), 1)
    neg_mask = 1.0 - pos_mask

    pos = torch.sum(sim * pos_mask, 1)
    neg = torch.sum(sim * neg_mask, 1)
    if True:
        loss = -(torch.mean(torch.log(pos / (pos + neg))))
    else:
        loss = -(torch.mean(torch.log(pos / neg)))
    return loss

class IncrementalCPTFixPNLoss(nn.Module):
    def __init__(self, w, omega, n_shots, use_cosine = False, tau=0.1):
        super(IncrementalCPTFixPNLoss, self).__init__()
        self.w = w
        self.omega = omega
        self.n_shots = n_shots
        self.use_cosine = use_cosine
        self.loss_func = nn.CrossEntropyLoss()
        self.l_type = 'cos_temp' #'cos_temp'
        self.r_type = 'none'
        self.tau = 0.1
        self.tau_reg = 0.1

    def forward(self, prototypes, pt_labels,
                output,
                output_base,
                output_new,
                labels):

        n_batch, _ = output.size()
        n_ways = int (n_batch / self.n_shots)

        # output : [25, 64]
        if self.use_cosine:
            output = F.normalize(output, dim=1)

            if output_base is not None:
                output_base = F.normalize(output_base, dim=1)
            if output_new is not None:
                output_new = F.normalize(output_new, dim=1)

            prototypes = F.normalize(prototypes, dim=1)

        novel_prototypes = output.reshape(n_ways,
                                          self.n_shots, -1).mean(dim=1)

        labels = torch.arange(n_ways).unsqueeze(1).expand(
            n_ways, self.n_shots).reshape(-1).cuda()

        prototypes_all = torch.cat((novel_prototypes, prototypes), dim=0)

        logits = - pair_norm_cosine_distances(output,
                                              prototypes_all,
                                              tau=self.tau,
                                              mode='test') * self.omega
        if self.l_type == 'sup_con':
            loss = supcon_loss_func(output, prototypes_all, labels, tau=self.tau)

        elif self.l_type == 'cos_temp':
            logits = - pair_norm_cosine_distances(output,
                                                  prototypes_all,
                                                  tau=self.tau,
                                                  mode='test',
                                                  exp=False) * self.omega
            loss = self.loss_func(logits, labels)

        elif self.l_type == 'euc':
            logits = - pair_euclidean_distances(output, prototypes_all) * self.omega
            loss = self.loss_func(logits, labels)

        # -------- reg
        reg_labels = torch.arange(output.shape[0]).cuda()
        if self.r_type == 'sup_con':
            reg = supcon_loss_func(output, output_base, reg_labels, tau=self.tau_reg)
        elif self.r_type == 'cos_temp':
            logit_new_base = - pair_norm_cosine_distances(output,
                                                          output_base,
                                                          tau=self.tau_reg,
                                                          mode='test')
            reg = self.loss_func(logit_new_base, reg_labels) * 0.1

        elif self.r_type == 'cos_reg':
            logit_new_base = pair_norm_cosine_regs(output,
                                                   output_base,
                                                   tau=self.tau_reg)
            reg = self.loss_func(logit_new_base, reg_labels)

        elif self.r_type == 'none':
            reg = 0

        # -------- total loss
        loss = loss + reg

        estimate = logits.argmax(dim=1)
        acc = (estimate == labels).sum() / output.shape[0]
        log = {'PTFixPNLoss': loss.item(), 'Acc': acc.item()}

        return loss * self.w, log


class IncrementalNetworkFixLoss(nn.Module):
    def __init__(self, w):
        super(IncrementalNetworkFixLoss, self).__init__()
        self.w = w

    def forward(self, former_param, param):
        former_param = [p.data for p in former_param]
        param = [p.data for p in param]

        l_total = 0.0
        for i, p in enumerate(former_param):
            c_p = param[i]
            l = (p - c_p).abs().sum()
            l_total += l
        l_total = l_total / len(former_param)

        log = {'NetFixLoss': l_total.item()}

        return l_total.item() * self.w, log

class IncrementalPNormPTLoss(nn.Module):
    def __init__(self, n_instances, n_shots, w = 1.0):
        super(IncrementalPNormPTLoss, self).__init__()
        self.w = w
        self.n_instances = n_instances
        self.n_shots = n_shots
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, output, novel_output, labels, novel_labels, tau):
        n_novel_data, embed_dim = novel_output.size()
        n_way = int(n_novel_data / self.n_shots)
        novel_protos = novel_output.reshape(n_way, self.n_shots, -1).mean(dim=1)

        n_buffer_data, embed_dim = output.size()
        n_buffer_way = int(n_buffer_data / self.n_instances)
        buffer_protos = output.reshape(n_buffer_way, self.n_instances, -1).mean(dim=1)

        protos_all = torch.cat((buffer_protos, novel_protos), dim=0)
        class_labels = labels.reshape(n_buffer_way, self.n_instances)[:,0]
        novel_class_labels = novel_labels.reshape(n_way, self.n_shots)[:,0]

        protos_labels = torch.cat((class_labels, novel_class_labels), dim=0)

        tau2 = tau[protos_labels].unsqueeze(dim=1)

        protos_all = protos_all * tau2

        output_all = torch.cat((output, novel_output), dim=0)
        logits = - pair_euclidean_distances(output_all, protos_all)

        labels_buffer = torch.arange(n_buffer_way).unsqueeze(1).expand(n_buffer_way, self.n_instances).reshape(-1).cuda()
        labels_novel = torch.arange(n_buffer_way, n_way + n_buffer_way).unsqueeze(1).expand(n_way, self.n_shots).reshape(-1).cuda()
        labels_all = torch.cat((labels_buffer, labels_novel)).reshape(-1)

        estimate = logits.argmax(dim=1)
        acc = (estimate == labels_all).sum() / labels_all.shape[0]

        loss = self.loss_func(logits, labels_all)

        log = {'NovelPNLoss': loss.item(), 'Acc': acc.item()}

        return loss * self.w, log

class CosineBufferCenterLoss(nn.Module):
    def __init__(self, buffer_size, w=1.0):
        super(CosineBufferCenterLoss, self).__init__()
        self.w = w
        self.buffer_size = buffer_size

    def forward(self, embed, buffer_embed, labels, buffer_labels):
        class_labels = torch.unique_consecutive(buffer_labels)
        n_ways = class_labels.shape[0]
        n_shots = self.buffer_size

        buffer_embed = F.normalize(buffer_embed, dim=1)
        embed = F.normalize(embed, dim=1)

        # (n_ways, embed_dim)
        protos = buffer_embed.reshape(n_ways, n_shots, -1).mean(dim=1)
        dis_pt = pair_norm_cosine_distances(protos, protos)

        dis_pt = dis_pt.sum() / (n_ways * (n_ways - 1))

        embed_all = torch.cat((embed, buffer_embed), dim=0)
        
        # norm = torch.norm(embed_all, dim=1).mean()

        labels_all = torch.cat((labels, buffer_labels), dim=0)
        batch_size, embed_dim = embed_all.size()

        distances = pair_norm_cosine_distances(embed_all, protos)

        labels_all = labels_all.unsqueeze(1).expand(batch_size, n_ways)
        class_labels = class_labels.unsqueeze(0).expand(batch_size, n_ways)
        mask = labels_all.eq(class_labels).float()

        distances = distances * mask

        loss = distances.clamp(min=1e-12, max=1e+12).sum() / batch_size

        log = {'CosBFCLoss': loss.item(), 'dis_pt': dis_pt.item()}

        return loss * self.w, log

class CosineSimilarBufferCenterLoss(nn.Module):
    def __init__(self, buffer_size, n_shots, w=1.0):
        super(CosineSimilarBufferCenterLoss, self).__init__()
        self.w = w
        self.buffer_size = buffer_size
        self.n_shots = n_shots

    def forward(self, embed, buffer_embed, labels, buffer_labels):
        class_labels = torch.unique_consecutive(buffer_labels)
        n_total_ways = class_labels.shape[0]

        buffer_embed = F.normalize(buffer_embed, dim=1)
        embed = F.normalize(embed, dim=1)

        # (n_ways, embed_dim)
        protos = buffer_embed.reshape(n_total_ways, self.buffer_size, -1).mean(dim=1)

        dis_pt = pair_norm_cosine_distances(protos, protos)
        dis_pt = dis_pt.sum() / (n_total_ways * (n_total_ways - 1))

        batch_size, _ = embed.size()
        n_ways = int(batch_size / self.n_shots)

        accurate_protos = embed.reshape(n_ways, self.n_shots, -1).mean(dim=1)
        labels = torch.unique_consecutive(labels)

        pt_index_dict = {class_label.item(): i for i, class_label in enumerate(class_labels)}
        estimate_protos = torch.stack([protos[pt_index_dict[label.item()]] for label in labels])

        loss = norm_cosine_distances(accurate_protos, estimate_protos).mean()

        log = {'SBFCLoss': loss.item(), 'dis_pt': dis_pt.item()}

        return loss * self.w, log

class NovelPNLoss(nn.Module):
    def __init__(self, n_instances, n_shots, w=1.0, omega=1.0):
        super(NovelPNLoss, self).__init__()
        self.n_instances = n_instances
        self.n_shots = n_shots
        self.w = w
        self.omega = omega
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, output, novel_output, labels, novel_labels):
        n_novel_data, embed_dim = novel_output.size()
        n_way = int(n_novel_data / self.n_shots)
        novel_protos = novel_output.reshape(n_way, self.n_shots, -1).mean(dim=1)

        n_buffer_data, embed_dim = output.size()
        n_buffer_way = int(n_buffer_data / self.n_instances)
        buffer_protos = output.reshape(n_buffer_way, self.n_instances, -1).mean(dim=1)

        protos_all = torch.cat((buffer_protos, novel_protos), dim=0)
        output_all = torch.cat((output, novel_output), dim=0)
        logits = - pair_euclidean_distances(output_all, protos_all) * self.omega

        labels_buffer = torch.arange(n_buffer_way).unsqueeze(1).expand(n_buffer_way, self.n_instances).reshape(-1).cuda()
        labels_novel = torch.arange(n_buffer_way, n_way + n_buffer_way).unsqueeze(1).expand(n_way, self.n_shots).reshape(-1).cuda()
        labels_all = torch.cat((labels_buffer, labels_novel)).reshape(-1)

        estimate = logits.argmax(dim=1)
        acc = (estimate == labels_all).sum() / protos_all.shape[0]

        loss = self.loss_func(logits, labels_all)

        log = {'NovelPNLoss': loss.item(), 'Acc': acc.item()}

        return loss * self.w, log

class CosineNovelPNLoss(nn.Module):
    def __init__(self, n_instances, n_shots, w=1.0, omega=1.0):
        super(CosineNovelPNLoss, self).__init__()
        self.n_instances = n_instances
        self.n_shots = n_shots
        self.w = w
        self.omega = omega
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, output, novel_output, labels, novel_labels):
        n_novel_data, embed_dim = novel_output.size()
        n_way = int(n_novel_data / self.n_shots)
        novel_protos = novel_output.reshape(n_way, self.n_shots, -1).mean(dim=1)

        n_buffer_data, embed_dim = output.size()
        n_buffer_way = int(n_buffer_data / self.n_instances)
        buffer_protos = output.reshape(n_buffer_way, self.n_instances, -1).mean(dim=1)

        protos_all = torch.cat((buffer_protos, novel_protos), dim=0)
        output_all = torch.cat((output, novel_output), dim=0)
        logits = - pair_norm_cosine_distances(output_all, protos_all) * self.omega

        labels_buffer = torch.arange(n_buffer_way).unsqueeze(1).expand(n_buffer_way, self.n_instances).reshape(-1).cuda()
        labels_novel = torch.arange(n_buffer_way, n_way + n_buffer_way).unsqueeze(1).expand(n_way, self.n_shots).reshape(-1).cuda()
        labels_all = torch.cat((labels_buffer, labels_novel)).reshape(-1)

        estimate = logits.argmax(dim=1)
        acc = (estimate == labels_all).sum() / protos_all.shape[0]

        loss = self.loss_func(logits, labels_all)

        log = {'NovelPNLoss': loss.item(), 'Acc': acc.item()}

        return loss * self.w, log

class NormFixLoss(nn.Module):
    def __init__(self, w):
        super(NormFixLoss, self).__init__()
        self.w = w

    def forward(self, output, novel_output, former_output, former_novel_output):
        former_novel_norm = torch.norm(former_novel_output, dim=1)
        former_old_norm = torch.norm(former_output, dim=1)

        current_novel_norm = torch.norm(novel_output, dim=1)
        current_old_norm = torch.norm(output, dim=1)

        old_ratio = current_old_norm / former_old_norm
        novel_ratio = current_novel_norm / former_novel_norm

        loss = torch.abs(old_ratio - novel_ratio)

        loss = loss.mean()

        log = {'NormFixLoss': loss.item()}

        return loss.item() * self.w, log

class TripletLoss_pytorch(nn.Module):
    def __init__(self, margin=0, num_instances=8, w=1.0):
        super(TripletLoss_pytorch, self).__init__()
        self.margin = margin
        self.num_instances = num_instances
        self.loss_func = torch.nn.TripletMarginLoss(margin=margin)
        self.w = w

    def forward(self, inputs, targets):
        batch, _ = inputs.size()
        n_ways = int(batch / self.num_instances)
        inputs_all = inputs.reshape(n_ways, self.num_instances, -1)
        positive = torch.roll(inputs_all, shifts=1, dims=1)
        negative = torch.roll(inputs_all, shifts=1, dims=0)
        positive = positive.reshape(batch, -1)
        negative = negative.reshape(batch, -1)
        loss = self.loss_func(inputs, positive, negative)
        log = {}
        log['TripletLoss'] = loss.item() * self.w

        return loss * self.w, log

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
    def __init__(self, n_supports=1, n_queries=15, n_ways=5, w=1.0):
        super(PNLoss, self).__init__()
        self.w = w
        self.n_supports = n_supports
        self.n_queries = n_queries
        self.n_ways = n_ways
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, features, labels):
        batch_size, _ = features.size()
        features = torch.reshape(features,
                               (self.n_ways, self.n_supports + self.n_queries, -1))

        supports = features[:, :self.n_supports, :]
        queries = features[:, self.n_supports:, :]
        queries = queries.reshape(self.n_ways * self.n_queries, -1)
        prototypes = supports.mean(dim=1)

        labels = torch.arange(self.n_ways).unsqueeze(1).expand(self.n_ways, self.n_queries).reshape(
            -1).cuda()

        logits = - pair_euclidean_distances(queries, prototypes)


        estimate = logits.argmax(dim=1)

        acc = (estimate == labels).sum() / queries.shape[0]

        loss = self.loss_func(logits, labels)

        log = {'PNLoss': loss.item(), 'Acc': acc.item()}

        return self.w * loss, log

class TripletLossNoHardMining(nn.Module):
    def __init__(self, margin=0, num_instances=8, w=1.0):
        super(TripletLossNoHardMining, self).__init__()
        self.margin = margin
        self.num_instances = num_instances
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.w = w

    def forward(self, inputs, targets):
        # dist = pair_euclidean_distances(inputs, inputs)
        # dist = dist.clamp(min=1e-12).sqrt()
        n = inputs.size(0)

        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        dist_ap, dist_an = [], []
        # TODO The training data of cub is uneven
        for i in range(n):
            for j in range(self.num_instances-1):
                tmp = dist[i][mask[i]]
                dist_ap.append(tmp[j+1])
                tmp = dist[i][mask[i] == 0]
                dist_an.append(tmp[j+1])
        dist_ap = torch.stack(dist_ap) # 100
        dist_an = torch.stack(dist_an) # 100
        # Compute ranking hinge loss
        y = dist_an.data.new()
        y.resize_as_(dist_an.data)
        y.fill_(1) # 100
        #y = Variable(y)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        prec = (dist_an.data > dist_ap.data).sum() * 1. / y.size(0)
        dist_p = torch.mean(dist_ap).item()
        dist_n = torch.mean(dist_an).item()

        log = {}
        log['TripletLoss'] = loss.item() * self.w
        log['Pos_dist'] = dist_p
        log['Neg_dist'] = dist_n
        log['Acc'] = prec.item()

        return self.w * loss, log

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

        log = {}
        log['CosTripletLoss'] = loss.item() * self.w
        log['Pos_dist'] = dist_p
        log['Neg_dist'] = dist_n
        log['Acc'] = prec.item()

        return self.w * loss, log

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
        log = {'LFLoss': loss.item()}
        return loss * self.lmbda, log

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
        else:
            loss = torch.zeros(1).cuda()

        log = {'InterClassLoss': loss.item()}

        return loss * self.weight, log
