import tensorflow as tf
from packaging import version
if version.parse(tf.__version__) < version.parse("2.6"):
    import tensorflow.keras as keras
else:
    import keras

from opts import opts

opt = opts()


def singleTagLoss(y_true, y_pred, loss_type):
    """
    associative embedding loss for one image
    """
    tags = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
    pull = tf.zeros([], tf.float32)

    for i in tf.range(y_true.shape[0]):
        pairs = y_true[i]
        tmp = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        # print("joints_per_person:   ", joints_per_person)
        # traverse different class
        for j in tf.range(pairs.shape[0]):
            p = pairs[j]
            # print("gt index:  ", joint)
            if p[1] > 0:
                # 根据int(joint[0]) idx 取出 pred_tag
                idx = int(p[0])
                # print(idx)
                tmp = tmp.write(j, y_pred[idx])

        # print("+++++++++++++++++++++++++++++++++++++++++++++\n")
        if tmp.size() == 0:
            continue
        tmp = tmp.stack()
        tags = tags.write(i, tf.reduce_mean(tmp, axis=0))
        pull = pull + tf.reduce_mean((tmp - tags.read(tags.size() - 1)) ** 2)  # avg((tmp - mean)**2)

    num_tags = tags.size()
    if num_tags == 0:
        return tf.zeros([], tf.float32), tf.zeros([], tf.float32)
    elif num_tags == 1:
        return tf.zeros([], tf.float32), pull
    else:
        tags = tags.stack()

        size = [num_tags, num_tags]

        # 各个点对均值之间的差
        A = tf.broadcast_to(tags, size)
        B = tf.transpose(A, [1, 0])  # A.T

        diff = A - B

        num_tags = tf.cast(num_tags, tf.float32)
        if loss_type == 'exp':
            diff = tf.pow(diff, 2)
            push = tf.exp(-diff)
            push = tf.reduce_sum(push) - num_tags
        elif loss_type == 'max':
            # 自己和自己的diff为0，1 - diff之后为1
            diff = 1 - tf.abs(diff)
            # 去除 自己和自己的diff
            push = tf.reduce_sum(tf.clip_by_value(diff, clip_value_min=0)) - num_tags
        else:
            raise ValueError('Unkown ae loss type')

        # A - B  diff计算了两次  所以需要*0.5
        return push / ((num_tags - 1) * num_tags) * 0.5, pull / num_tags


def aeLoss(y_true, y_pred, loss_type="exp"):

    """
    accumulate the tag loss for each image in the batch
    """
    batch_size = y_pred.shape[0]
    pushes, pulls = tf.TensorArray(tf.float32, size=batch_size), tf.TensorArray(tf.float32, size=batch_size)
    # joints = joints.cpu().data.numpy()
    for i in tf.range(batch_size):
        push, pull = singleTagLoss(y_true[i], y_pred[i], loss_type)
        pushes = pushes.write(i, push)
        pulls = pulls.write(i, pull)
    return tf.reduce_sum(pushes.stack()), tf.reduce_sum(pulls.stack())



class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, need_sigmoid, reduction=tf.keras.losses.Reduction.NONE):
        super(FocalLoss, self).__init__(reduction=reduction)
        self.need_sigmoid = need_sigmoid

    def call(self, y_true, y_pred):
        # pos代表正样本，neg代表负样本

        # 4个或6个
        pos_inds = tf.cast(tf.math.equal(y_true, 1),tf.float32)  # heatmap为1的部分是正样本
        neg_inds = tf.cast(tf.math.less(y_true, 1),tf.float32)  # 其他部分为负样本

        # 正样本权重1，负样本权重高斯值越小，权重越小
        neg_weights = tf.math.pow(1 - y_true, 4)  # 对应(1-Yxyc)^4

        loss = tf.zeros([], tf.float32)

        # clip y_pred to avoid nan
        # code from tf.losses.binary_crossentropy
        if self.need_sigmoid:
            y_pred = tf.math.sigmoid(y_pred)
        # epsilon_ = tf.keras.backend.epsilon()
        epsilon_ = 1e-5
        y_pred = tf.clip_by_value(y_pred, epsilon_, 1. - epsilon_)

        pos_loss = tf.math.log(y_pred) * tf.math.pow(1 - y_pred, 2) * pos_inds
        neg_loss = tf.math.log(1 - y_pred) * tf.math.pow(y_pred, 2) * neg_weights * neg_inds

        num_pos = tf.reduce_sum(pos_inds)
        pos_loss = tf.reduce_sum(pos_loss)
        neg_loss = tf.reduce_sum(neg_loss)

        if num_pos == 0:
            loss = loss - neg_loss  # 只有负样本
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
        tf.print("pos", pos_loss)
        tf.print("neg", neg_loss)
        return loss

def gather_feat(feat, ind):
    # ind代表的是ground truth中设置的存在目标点的下角标
    # feat（即，预测的output['wh']）维度32*2*96*96----->32*96*96*2
    # feat = feat.permute(0, 2, 3, 1).contiguous()  # from [bs c h w] to [bs, h, w, c]
    # feat维度32*96*96*2----->32*9216*2
    # feat = feat.view(feat.size(0), -1, feat.size(3))  # to [bs, wxh, c]
    # 根据ind取出feat中对应的元素;  因为不是dense_wh形式，训练数据中wh的标注batch['wh']的维度是self.max_objs*2，和预测的输出feat（output['wh']）的维度32*2*96*96不相符，
    # 没有办法进行计算求损失，所以需要根据ind（对象在heatmap图上的索引）取出feat中对应的元素，使其维度和batch['wh']一样，最后维度为32*50*2

    feat = tf.reshape(feat,[feat.shape[0], -1, feat.shape[3]])
    # feat = _gather_feat(feat, ind)
    # dim = 2
    # dim  = feat.shape[2]   # feat : [bs, wxh, c]
    # ind维度 ：32*50---->32*50*2
    # ind = tf.expand_dims(ind, axis=2)
    # ind = tf.broadcast_to(ind, [ind.shape[0], ind.shape[1], dim]) # ind : [bs, index, c]
    # feat = tf.gather(feat, ind, axis=1)
    feat = tf.gather(feat, ind, axis=1, batch_dims=1)
    return feat


def RegL1Loss(y_pred, mask, ind, y_true):
    pred = gather_feat(y_pred, ind)
    # mask = mask.unsqueeze(2).expand_as(pred).float()

    mask = tf.expand_dims(mask, axis=2)
    mask = tf.cast(mask, tf.float32)

    # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
    # loss = F.l1_loss(pred * mask, y_true * mask, reduce=False)
    # loss = tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.NONE)(y_true * mask, pred * mask)
    loss = tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.SUM)(y_true * mask, pred * mask)
    loss = loss / (tf.reduce_sum(mask) + 1e-4)
    return loss


class AELoss(tf.keras.losses.Loss):
    def __init__(self, opt=opt, pull_weight=1, push_weight=1, regr_weight=0.1, reduction=tf.keras.losses.Reduction.NONE):
        super(AELoss, self).__init__(reduction=reduction)

        self.pull_weight = pull_weight  # pull_weight=
        self.push_weight = push_weight  # beta
        self.regr_weight = regr_weight  # gamma

        self.focal_loss = tf.losses.MSE() if opt.mse_loss else FocalLoss(opt.need_sigmoid)  # heatmap loss
        self.ae_loss = aeLoss  # embedding loss
        self.crit_reg = RegL1Loss
        self.opt = opt

    def call(self, y_true, y_pred):
        opt = self.opt
        hm_pred, reg_pred, em_pred, hm_pool = y_pred
        # heatmap loss
        hm_loss = self.focal_loss(y_true['hm'], hm_pred)

        #  embedding loss
        b = em_pred.shape[0]
        em_p = tf.reshape(em_pred, [b, -1, 1]) # b h*w*c 1
        push_loss, pull_loss = self.ae_loss(y_true['embedding'], em_p)

        pull_loss = self.pull_weight * pull_loss
        push_loss = self.push_weight * push_loss
        em_loss = pull_loss + push_loss
        # xy offset_loss
        off_loss = 0
        if opt.reg_offset and opt.off_weight > 0:
            off_loss = self.crit_reg(reg_pred, y_true['reg_mask'],
                                     y_true['ind'], y_true['reg']) / opt.num_stacks
        off_loss = self.regr_weight * off_loss

        # loss = opt.hm_weight * hm_loss + opt.em_weight * em_loss + \
        #        opt.off_weight * off_loss


        return tf.stack([hm_loss, pull_loss, push_loss, off_loss])



