import tensorflow as tf
from packaging import version
if version.parse(tf.__version__) < version.parse("2.6"):
    AUTOTUNE = tf.data.experimental.AUTOTUNE
else:
    AUTOTUNE = tf.data.AUTOTUNE
import tensorflow_datasets as tfds
import numpy as np
from utils import *
import cv2
from opts import opts
import math
import functools


opt = opts()
def preprocess(img, points, pt_labels,aug):

    height, width = img.shape[:2]
    c = np.array([width / 2., height / 2.], dtype=np.float32)
    s = max(height, width) * 1.0

    input_h, input_w = opt.net_height, opt.net_width  # 网络输入的宽高

    trans_input = get_affine_transform(c, s, 0, [input_w, input_h])

    # inp = cv2.warpAffine(np.array(img), trans_input, (input_w, input_h), flags=cv2.INTER_LINEAR)
    inp = cv2.warpAffine(img, trans_input, (input_w, input_h), flags=cv2.INTER_LINEAR)
    inp = (inp.astype(np.float32) / 255.)

    if aug:
        _data_rng = np.random.RandomState(123)
        _eig_val = np.array([0.2141788, 0.01817699, 0.00341571], dtype=np.float32)
        _eig_vec = np.array([
            [-0.58752847, -0.69563484, 0.41340352],
            [-0.5832747, 0.00994535, -0.81221408],
            [-0.56089297, 0.71832671, 0.41158938]
        ], dtype=np.float32)
        # 数据增强
        color_aug(_data_rng, inp, _eig_val, _eig_vec)
    # float32 -> float64
    inp = (inp - np.array([0.408, 0.447, 0.47])) / np.array([0.289, 0.274, 0.278])
    inp = inp.astype(np.float32)

    # '为输出做仿射变换做准备'
    output_h = input_h // opt.down_ratio  # heatmap的高 128
    output_w = input_w // opt.down_ratio  # heatmap的宽 128
    num_classes = opt.num_classes  # 类别数
    trans_output = get_affine_transform(c, s, 0, [output_w, output_h])

    # 定义所需张量
    hm = np.zeros((output_h, output_w, num_classes), dtype=np.float32)  # (128, 128, cls_num)  heatmap

    embedding = np.zeros((opt.max_objs, 2, 2), dtype=np.float32)  # 点对的embedding
    # embedding = np.zeros((opt.max_objs // 2, 2, 2), dtype=np.float32)  # 点对的embedding

    reg = np.zeros((opt.max_objs, 2), dtype=np.float32)  # 点的误差offset

    ind = np.zeros((opt.max_objs), dtype=np.int64)
    reg_mask = np.zeros((opt.max_objs), dtype=np.uint8)

    draw_gaussian = draw_msra_gaussian if opt.mse_loss else draw_umich_gaussian

    # '准备数据集返回变量，主要是真值标签，如热力图、embeddings 和 中心点偏移量
    num_points = points.shape[0]
    step = int(num_points / 2)
    max_distance = 0
    # obj_index = np.arange(num_points).reshape(2, int(num_points / 2)).T
    for i in range(step):
        dis_y = abs(points[i][1] - points[i + step][1])
        if dis_y > max_distance:
            max_distance = dis_y

    for k in range(step):  # '按实际有多少个目标对 进行循环'

        point = points[k]  # gt point: (x,y)
        point_em = points[k + step]

        cls_id = pt_labels[k]  # gt cls id
        cls_id_em = pt_labels[k + step] # gt cls id

        # '输出的仿射变换，这些仿射变换由相应的函数完成，在实际编写中只需确定诸如中心点c、长边长度s等参数，便于在自己的数据集中使用
        # 由于没有gt的w,h  只需要对gt point的x,y进行变换

        point = affine_transform(point, trans_output)  # 中心点仿射变换
        point[0] = np.clip(point[0], 0, output_w - 1)
        point[1] = np.clip(point[1], 0, output_h - 1)

        point_em = affine_transform(point_em, trans_output)  # 中心点仿射变换
        point_em[0] = np.clip(point_em[0], 0, output_w - 1)
        point_em[1] = np.clip(point_em[1], 0, output_h - 1)

        h, w = int(max_distance / 4), int(max_distance / 4)  # 决定高斯核的宽高

        if h > 0 and w > 0:
            radius = gaussian_radius((math.ceil(h), math.ceil(w)))
            radius = max(0, int(radius))
            # radius = [opt.]hm_gauss if opt.mse_loss else radius

            ct = np.array(point, dtype=np.float32)  # 中心点
            ct_int = ct.astype(np.int32)

            ct_em = np.array(point_em, dtype=np.float32)  # 中心点
            ct_int_em = ct_em.astype(np.int32)


            draw_gaussian(hm[..., cls_id], ct_int, radius)  # 绘制热力图，绘制在其所属类别的通道上
            draw_gaussian(hm[..., cls_id_em], ct_int_em, radius)  # 绘制热力图，绘制在其所属类别的通道上


            # k对 2(类) 2(idx, 1)
            # embedding[k][cls_id], embedding[k][cls_id_em] = (ct_int[1] * output_w + ct_int[0], 1), (
            # (output_h * output_w) + ct_int_em[1] * output_w + ct_int_em[0], 1)

            embedding[k][cls_id], embedding[k][cls_id_em] = (ct_int[1] * output_w * 2 + ct_int[0] * 2 + 0, 1), \
                                                            (ct_int_em[1] * output_w * 2 + ct_int_em[0] * 2 + 1, 1)

            # print(ct_int[1],  ct_int[0])
            # print(ct_int_em[1],  ct_int_em[0])
            # print("______")
            # print(ct[1] * output_h + ct[0], 1), ((output_h * output_w) + ct_em[1] * output_h + ct_em[0], 1)

            ind[k] = ct_int[1] * output_w + ct_int[0]  # 中心点的位置，用一维表示h*W+w
            ind[k + step] = ct_int_em[1] * output_w + ct_int_em[0]  # 中心点的位置，用一维表示h*W+w

            reg[k] = ct - ct_int  # 由取整引起的误差  中心点误差offset的gt
            reg[k + step] = ct_em - ct_int_em

            reg_mask[k] = 1  # 设为1, 表示该位置有目标存在
            reg_mask[k + step] = 1

            # if opt.dense_wh:
            #     draw_dense_reg(dense_wh, hm.max(axis=0), ct_int, embedding[k], radius)


    # ret = {'input': inp, 'hm': hm, 'reg_mask': reg_mask, 'reg': reg, 'ind': ind, 'embedding': embedding}

    return inp, hm, reg_mask, reg, ind, embedding

def preprocess_function(data,aug):
    inp, hm, reg_mask, reg, ind, embedding = tf.numpy_function(func=functools.partial(preprocess,aug=aug), inp=[data['image'], data['points']['point'], data['points']['category']],
                                Tout=[tf.float32, tf.float32, tf.uint8, tf.float32, tf.int64, tf.float32])
    # return inp, (hm, reg_mask, reg, ind, embedding)
    return inp, {"hm":hm, "reg_mask":reg_mask, "reg": reg, "ind":ind, "embedding": embedding}

def prepare(ds, batch_size, aug=True, shuffle=False):
  if shuffle:
    ds = ds.shuffle(1000)


  ds = ds.map(functools.partial(preprocess_function,aug=aug), num_parallel_calls=AUTOTUNE)

  # Batch all datasets.
  ds = ds.batch(batch_size)

  # Use buffered prefetching on all datasets.
  return ds.prefetch(buffer_size=AUTOTUNE)


if __name__ == "__main__":
    # ds, info = tfds.load("foot_robot", with_info=True)
    # print(tfds.benchmark(ds))
    # print(info.splits['train'].num_examples)

    # data = next(iter(ds['train']))
    # preprocess(data['image'], data['points']['point'], data['points']['category'])

    ds, info = tfds.load("foot_robot", with_info=True)
    ds = ds['train'].take(100)
    for data in ds:
        preprocess(data['image'].numpy(), data['points']['point'].numpy(), data['points']['category'].numpy(), aug=True)
    # # data = next(iter(ds))
    # # preprocess(data['image'], data['points']['point'], data['points']['category'], aug=True)
    #
    # ds = prepare(ds['train'], 8, shuffle=True)
    # print(tfds.benchmark(ds, batch_size=8))
    # data = next(iter(ds))
    #
    # print("asdf")



