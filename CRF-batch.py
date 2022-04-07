import torch
import torch.nn as nn

START_TAG = -2
END_TAG = -1


def log_sum_exp(vec, tag_size):
    batch_size = vec.shape[0]
    max_score, idx = torch.max(vec, 1)
    max_expand = max_score.view(batch_size, tag_size, 1).expand(batch_size, tag_size, tag_size)

    return max_score + torch.log(torch.sum(torch.exp(vec - max_expand), 1)).view(batch_size, tag_size)


class CRF(nn.Module):

    def __init__(self, tag_size, GPU=False):
        super(CRF, self).__init__()
        self.tag_size = tag_size + 2

        transition = torch.zeros(self.tag_size, self.tag_size)
        if GPU:
            transition = transition.cuda()
        self.transition = nn.Parameter(transition)

    def _forward_alg(self, feats, mask):
        """

        :param feats: batch_size * seq_len * tag_size
        :param mask: bat_size * seq_len
        :return:
        """
        shape = feats.shape
        batch_size = shape[0]
        seq_len = shape[1]
        tag_size = shape[2]
        assert tag_size == self.tag_size
        nums = seq_len * batch_size
        mask = mask.transpose(0, 1).contiguous()

        scores = feats.transpose(0, 1).contiguous().view(nums, 1, tag_size).expand(nums, tag_size, tag_size)
        scores = scores + self.transition.view(1, tag_size, tag_size).expand(nums, tag_size, tag_size)
        scores = scores.view(seq_len, batch_size, tag_size, tag_size)
        word_iter = enumerate(scores)
        _, init_values = next(word_iter)
        partition = init_values[:, START_TAG, :].clone().view(batch_size, tag_size, 1)

        for idx, cur_word in word_iter:
            cur_partition = cur_word + partition.expand(batch_size, tag_size, tag_size)
            cur_value = log_sum_exp(cur_partition, tag_size)
            mask_idx = mask[idx].view(batch_size, 1).expand(batch_size, tag_size)
            masked_cur_partition = cur_value.masked_select(mask_idx)
            mask_idx = mask_idx.view(batch_size, tag_size, 1)
            partition.masked_scatter_(mask_idx, masked_cur_partition)
        cur_partition = partition.expand(batch_size, tag_size, tag_size) + self.transition.view(1, tag_size,
                                                                                                tag_size).expand(
            batch_size, tag_size, tag_size)
        end_value = log_sum_exp(cur_partition, tag_size)
        score = end_value[:, END_TAG]

        return torch.sum(score), score

    def neg_log_likelihood_loss(self, feats, mask, label):
        s1, s2 = self._forward_alg(feats, mask)
        print(s1)


feats = torch.rand(8, 4, 5)
crf = CRF(3)
mask = torch.rand(8, 3).ge(0.3)
label = torch.rand(8, 4)
crf.neg_likilihood_loss(feats, mask, label)
