import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class NormalHead(nn.Module):
    def __init__(self, input_channels, output_channels=1):
        super(NormalHead, self).__init__()

        self.head = nn.Sequential(
            nn.Conv2d(input_channels, input_channels // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(input_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_channels // 4, output_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.head(x)


class MultiTaskHead(nn.Module):
    def __init__(self, input_channels, output_channels_list):
        super(MultiTaskHead, self).__init__()

        heads = []
        for output_channels in output_channels_list:
            heads.append(NormalHead(input_channels, output_channels))
        self.heads = nn.ModuleList(heads)

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=1)


class LineProposalNetwork(nn.Module):
    def __init__(self,
                 order,
                 input_channels,
                 output_channels_list,
                 junc_thresh=0.008,
                 junc_max_num=300,
                 line_max_num=5000,
                 num_pos_proposals=300,
                 num_neg_proposals=300,
                 nms_size=3
                 ):
        super(LineProposalNetwork, self).__init__()

        self.order = order
        self.junc_thresh = junc_thresh
        self.junc_max_num = junc_max_num
        self.line_max_num = line_max_num
        self.num_pos_proposals = num_pos_proposals
        self.num_neg_proposals = num_neg_proposals
        self.nms_size = nms_size

        self.head = MultiTaskHead(input_channels=input_channels, output_channels_list=output_channels_list)

        lambda_ = np.concatenate((np.linspace(1.0, 0.0, self.order + 1)[None],
                             np.linspace(0.0, 1.0, self.order + 1)[None]))
        lambda_ = torch.from_numpy(lambda_).float()
        self.register_buffer('lambda_', lambda_)

    def forward(self, features, metas=None):
        """
        Forward Line Proposal Network

        Here are notations.

        * math:`N`: batch size of the input features.
        * math:`C`: channel size of the input features.
        * math:`H`: height of the input features.
        * math:`W`: width of the input features.
        * math:`R`: number of the proposed lines.
        * math:`O`: order of the Bezier curve.

        :param features: The features extracted from images.
                Its shape is :math:`(N, C, H, W)`.
            metas: A dict of ground truth junctions and lines.
        :return:
            maps: Dict of predicted maps for confidence and offset.
            loi_preds: A list of proposed lines array.
                    Its shape is :math:`(N, (R, O, 2))`.
        """
        b, c, h, w = features.shape

        # Maps
        maps = self.head(features)
        lmap = torch.sigmoid(maps[:, 0:1])
        jmap = torch.sigmoid(maps[:, 1:2])
        joff = maps[:, 2:4]
        cmap = torch.sigmoid(maps[:, 4:5])
        coff = maps[:, 5:7]
        lvec = maps[:, 7:]
        indeces = np.arange(lvec.shape[1]).reshape((-1, 2, 2))
        indeces[:, 1] = indeces[::-1, 1]
        indeces = indeces.reshape((-1))
        lvec = lvec[:, indeces]
        maps = {'lmap': lmap, 'jmap': jmap, 'joff': joff, 'cmap': cmap, 'coff': coff, 'lvec': lvec}

        # Lois
        with torch.no_grad():
            loi_preds = []
            for i in range(b):
                jmap = maps['jmap'][i]
                joff = maps['joff'][i]
                cmap = maps['cmap'][i]
                coff = maps['coff'][i]
                lvec = maps['lvec'][i]

                # Generate junctions and lines
                jmap = non_maximum_suppression(jmap, self.nms_size)
                if self.training:
                    junc_pred, junc_score = calc_junction(jmap, joff, thresh=0.0, top_K=self.junc_max_num)
                    loi_pred, _ = calc_line(cmap, coff, lvec, self.order, thresh=0.0, top_K=self.line_max_num)
                else:
                    junc_pred, junc_score = calc_junction(jmap, joff, thresh=self.junc_thresh, top_K=self.junc_max_num)
                    loi_pred, _ = calc_line(cmap, coff, lvec, self.order, thresh=0.0, top_K=self.line_max_num)

                # Match junctions and lines
                dist_junc_to_end1, idx_junc_to_end1 = ((loi_pred[:, None, 0] - junc_pred[None]) ** 2).sum(dim=-1).min(
                    dim=-1)
                dist_junc_to_end2, idx_junc_to_end2 = ((loi_pred[:, None, -1] - junc_pred[None]) ** 2).sum(dim=-1).min(
                    dim=-1)
                idx_junc_to_end_min = torch.min(idx_junc_to_end1, idx_junc_to_end2)
                idx_junc_to_end_max = torch.max(idx_junc_to_end1, idx_junc_to_end2)
                iskeep = idx_junc_to_end_min != idx_junc_to_end_max

                if self.order == 1:
                    idx_junc = torch.unique(
                        torch.cat((idx_junc_to_end_min[iskeep, None], idx_junc_to_end_max[iskeep, None]), dim=-1), dim=0
                    )
                    loi_pred = torch.cat((junc_pred[idx_junc[:, 0], None], junc_pred[idx_junc[:, 1], None]), dim=1)
                    mask = loi_pred[:, 0, 1] > loi_pred[:, 1, 1]
                    loi_pred[mask] = loi_pred[mask][:, [1, 0]]

                else:
                    loi_pred = loi_pred[iskeep]
                    idx_junc = torch.cat((idx_junc_to_end_min[iskeep, None], idx_junc_to_end_max[iskeep, None]), dim=1)
                    dist_junc = dist_junc_to_end1[iskeep] + dist_junc_to_end2[iskeep]
                    indeces = dist_junc.argsort()
                    loi_pred = loi_pred[indeces]
                    idx_junc = idx_junc[indeces]
                    idx_junc, unique_indices = np.unique(idx_junc.detach().cpu().numpy(), return_index=True, axis=0)
                    end_pred = torch.cat((junc_pred[idx_junc[:, 0], None], junc_pred[idx_junc[:, 1], None]), dim=1)
                    mask = end_pred[:, 0, 1] > end_pred[:, 1, 1]
                    end_pred[mask] = end_pred[mask][:, [1, 0]]
                    loi_pred = loi_pred[unique_indices]
                    delta_end_pred = end_pred - loi_pred[:, [0, -1]]
                    loi_pred += (self.lambda_[None, :, :, None] * delta_end_pred[:, :, None, :]).sum(1)

                loi_preds.append(loi_pred)

        return maps, loi_preds

    def sample_lines(self, loi_preds, metas):
        """
        Sample proposed lines

        Here are notations.

        * math:`N`: batch size of the input maps.
        * math:`H`: height of the input maps.
        * math:`W`: witdh of the input maps.
        * math:`R`: number of the proposed lines.
        * math:`R'`: number of the sampled lines.
        * math:`O`: order of the Bezier curve.

        :param loi_preds:  A list of proposed lines array.
                Its shape is :math:`(N, (R, O, 2))`.
            metas: A dict of ground truth lines.
        :return:
            sample_loi_preds: A list of sampled lines array.
                    Its shape is :math:`(N, (R', O, 2))`.
            sample_loi_labels: A list of sampled line labels array.
                Its shape is :math:`(N, (R'))`.
        """
        b = len(loi_preds)

        sample_loi_preds, sample_loi_labels = [], []
        for i in range(b):
            loi_pred = loi_preds[i]
            lpre = metas['lpre'][i]
            lpre_label = metas['lpre_label'][i]
            line = metas['line'][i]

            loi_pred_mirror = loi_pred[:, range(loi_pred.shape[1])[::-1]]
            loi_pred = torch.cat((loi_pred, loi_pred_mirror))

            dists, _ = ((loi_pred[:, None] - line) ** 2).sum(-1).mean(-1).min(-1)
            label = dists <= 1.5 * 1.5
            pos_id = label.nonzero(as_tuple=False).flatten()
            neg_id = (label == 0).nonzero(as_tuple=False).flatten()

            if len(pos_id) > self.num_pos_proposals:
                idx = torch.randperm(pos_id.shape[0], device=pos_id.device)[:self.num_pos_proposals]
                pos_id = pos_id[idx]
            if len(neg_id) > self.num_neg_proposals:
                idx = torch.randperm(neg_id.shape[0], device=neg_id.device)[:self.num_neg_proposals]
                neg_id = neg_id[idx]

            keep_id = torch.cat((pos_id, neg_id))
            loi_pred = loi_pred[keep_id]
            loi_label = torch.cat([torch.ones(len(pos_id), dtype=loi_pred.dtype, device=loi_pred.device),
                                   torch.zeros(len(neg_id), dtype=loi_pred.dtype, device=loi_pred.device)])
            loi_pred = torch.cat((loi_pred, lpre))
            loi_label = torch.cat((loi_label, lpre_label))

            sample_loi_preds.append(loi_pred)
            sample_loi_labels.append(loi_label)

        return sample_loi_preds, sample_loi_labels


def calc_junction(jmap, joff, thresh, top_K):
    h, w = jmap.shape[-2], jmap.shape[-1]
    score = jmap.flatten()
    joff = joff.reshape(2, -1).t()

    num = min(int((score >= thresh).sum().item()), top_K)
    indices = torch.argsort(score, descending=True)[:num]
    score = score[indices]
    y, x = indices // w, indices % w
    junc = torch.cat((x[:, None], y[:, None]), dim=1) + joff[indices] + 0.5

    junc[:, 0] = junc[:, 0].clamp(min=0, max=w - 1e-4)
    junc[:, 1] = junc[:, 1].clamp(min=0, max=h - 1e-4)

    return junc, score


def calc_line(cmap, coff, lvec, order, thresh, top_K):
    n_pts = order + 1
    h, w = cmap.shape[-2], cmap.shape[-1]
    score = cmap.flatten()
    coff = coff.reshape(2, -1).t()
    lvec = lvec.reshape(n_pts // 2 * 2, 2, -1).permute([2, 0, 1])

    num = min(int((score >= thresh).sum().item()), top_K)
    indices = torch.argsort(score, descending=True)[:num]
    score = score[indices]
    y, x = indices // w, indices % w

    center = torch.cat((x[:, None], y[:, None]), dim=1) + coff[indices] + 0.5
    loi = center[:, None] + lvec[indices]
    if n_pts % 2 == 1:
        loi = torch.cat((loi[:, :n_pts // 2], center[:, None], loi[:, n_pts // 2:]), dim=1)

    loi[:, :, 0] = loi[:, :, 0].clamp(min=0, max=w - 1e-4)
    loi[:, :, 1] = loi[:, :, 1].clamp(min=0, max=h - 1e-4)

    return loi, score


def non_maximum_suppression(heatmap, kernel_size):
    max_heatmap = F.max_pool2d(heatmap, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    weights = (heatmap == max_heatmap).float()
    heatmap = weights * heatmap
    return heatmap
