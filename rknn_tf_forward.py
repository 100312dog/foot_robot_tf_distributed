import numpy as np
import torch
import torch.nn as nn
import cv2
import tensorflow as tf
from packaging import version
from model import PoseHighResolutionNet, extra32
if version.parse(tf.__version__) < version.parse("2.6"):
    import tensorflow.keras as keras
    from tensorflow.keras import layers
else:
    import keras
    from keras import layers


from tensorflow_model_optimization.python.core.quantization.keras.quantize_wrapper import QuantizeWrapper
from model_compression_toolkit.qat.keras.quantizer.configs.weight_quantizer_config import WeightQuantizeConfig

from dataset import prepare
import tensorflow_datasets as tfds
from loss import AELoss

def compare(a,b):
    print((a-b).min(), (a-b).max())
    print((a-b).sum()/128/128/2)


if __name__ == "__main__":

    index2label = {0: "label1", 1:"label2"}
    # load image
    # img = cv2.imread("/home/fsw/Documents/codes/CenterNet_match/snapshot/foot_robot1028/src_img.jpg")
    img = cv2.imread("/home/fsw/Documents/codes/CenterNet_match/dataset/foot_robot/train/10.jpg")
    img_h, img_w, _ = img.shape
    input_w, input_h = 512, 512

    image = cv2.resize(img, (512, 512),cv2.INTER_NEAREST)
    mean = np.array([0.408, 0.447, 0.47], dtype=np.float32) * 255
    std = np.array([0.289, 0.274, 0.278], dtype=np.float32) * 255
    image = (image - mean) / std
    image = np.expand_dims(image, axis=0)

    model_path = r'foot_robot1028_512_512_160.pth'

    model = get_hr(num_layers=32, heads={"hm": 2, "em": 2, "reg": 2}).cuda()

    checkpoint = torch.load(model_path, map_location="cpu")

    new_checkpoint = collections.OrderedDict()

    for name, module in checkpoint.items():
        name = name.replace("module.", "")
        new_checkpoint[name] = module

    model.load_state_dict(new_checkpoint, strict=True)

    model.cuda()
    model.eval()


    def nms(heat, kernel=1):
        pad = (kernel - 1) // 2

        hmax = nn.functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
        keep = (hmax == heat).float()
        return heat * keep, hmax


    def topk(scores, K=20):
        batch, cat, height, width = scores.size()

        topk_scores, topk_inds = torch.topk(scores.contiguous().view(batch, -1), K)
        topk_clses = torch.floor_divide(topk_inds, (height * width)).int()

        topk_inds = topk_inds % (height * width)
        topk_ys = torch.floor_divide(topk_inds, width).int().float()
        topk_xs = (topk_inds % width).int().float()
        return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs

    nmskey, hmax = nms(hm, 3)
    kscore, kinds, kcls, kys, kxs = topk(nmskey, K=10)

    kys = kys.cpu().data.numpy().astype(np.int)
    kxs = kxs.cpu().data.numpy().astype(np.int)

    key = [[], []]  # key是目标点在ht中的坐标
    good_scores = []
    good_cls = []

    for ind in range(kscore.shape[1]):
        score = kscore[0, ind]
        cls = kcls[0, ind]
        # if score > 0.05:
        if score > 0.3:
            key[0].append(kys[0, ind])
            key[1].append(kxs[0, ind])
            good_scores.append(score)
            good_cls.append(cls)
    stride = 4

    if key[0] is not None and len(key[0]) > 0:

        pred_centers = np.zeros([2, len(good_cls)])
        pre_centers_em = np.zeros([len(good_cls)])
        for j in range(len(good_cls)):
            # 类别
            cls = good_cls[j]

            # 中心点
            print(key[0][j], key[1][j])
            reg_x = xy[0, 0, key[0][j], key[1][j]]
            reg_y = xy[0, 1, key[0][j], key[1][j]]
            cx = (reg_x + key[1][j]) * stride
            cy = (reg_y + key[0][j]) * stride

            pred_centers[0, j] = cx
            pred_centers[1, j] = cy

            # embedding
            pre_centers_em[j] = em[0, cls.item(), key[0][j], key[1][j]]
            # print(pre_centers_em[j])

            # oem1 = reg_em1 * stride
            # oem2 = reg_em2 * stride
    else:
        pred_centers = (None, None, None, None)
        pre_centers_em = None

    x, y = pred_centers
    if x is not None:
        # image = restoreImage(images, 0)
        # image = image.astype(np.uint8).copy()
        scale_w, scale_h = input_w / img_w, input_h / img_h
        for i in range(x.shape[0]):
            x_ = max((1 / scale_w) * x[i], 0)
            y_ = max((1 / scale_h) * y[i], 0)

            if good_scores[i] > 0.3:
                cv2.circle(img, (int(x_), int(y_)), 4, (255, 0, 255), -1)
                cv2.putText(img, index2label[int(good_cls[i])], (int(x_ + 10), int(y_ + 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 1)
                # cv2.putText(img, str(round(float(good_scores[i]), 3)), (int(x_ + 30), int(y_ + 20)),
                #     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 240, 0), 1)

                cv2.putText(img, str(round(float(pre_centers_em[i]), 3)), (int(x_ + 30), int(y_ + 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 240, 0), 1)
            # txt_f.write(' '.join(
            #     [index2label[int(good_cls[i])], str(int(x_)), str(int(y_)), str(float(good_scores[i]))]) + '\n')
        # print(os.path.join(img_save_dir, img_name))
        cv2.imshow("img",img)
        cv2.waitKey()
        # cv2.imwrite('./onnx_test_lq1117_2.jpg', image_source)

    # # perf
    # print('--> Begin evaluate model performance')
    # perf_results = rknn.eval_perf(inputs=[img])
    # print('done')
