import tensorflow as tf
from tensorflow import keras
from packaging import version
if version.parse(tf.__version__) < version.parse("2.6"):
    from tensorflow.keras import layers
else:
    from keras import layers

import tensorflow_datasets as tfds
from functools import partial

# import torch.nn as nn
from yacs.config import CfgNode as CN
import functools


from collections import OrderedDict

import numpy as np


BN_MOMENTUM = 0.9

class BasicBlock:
    expansion = 1
    @staticmethod
    def func(x, planes, stride=1, downsample=None, name=None):
        residual = x

        out = layers.Conv2D(planes, 3, strides=stride, padding='SAME', use_bias=False, name=f"{name}.conv1")(x)
        out = layers.BatchNormalization(axis=3, momentum=BN_MOMENTUM, epsilon=1e-5, name=f"{name}.bn1")(out)
        out = layers.Activation('relu', name=f"{name}.relu1")(out)

        out = layers.Conv2D(planes, 3, strides=1, padding='SAME', use_bias=False, name=f"{name}.conv2")(out)
        out = layers.BatchNormalization(axis=3, momentum=BN_MOMENTUM, epsilon=1e-5, name=f"{name}.bn2")(out)

        if downsample is not None:
            residual = downsample(x)

        # layers.Add()
        out += residual
        out = layers.Activation('relu', name=f"{name}.relu2")(out)

        return out

class Bottleneck:
    expansion = 4
    @staticmethod
    def func(x, planes, stride=1, downsample=None, name=None):
        residual = x

        out = layers.Conv2D(planes, 1, strides=1, padding='VALID', use_bias=False, name=f"{name}.conv1")(x)
        out = layers.BatchNormalization(axis=3, momentum=BN_MOMENTUM, epsilon=1e-5, name=f"{name}.bn1")(out)
        out = layers.Activation('relu', name=f"{name}.relu1")(out)

        out = layers.Conv2D(planes, 3, strides=stride, padding='SAME', use_bias=False, name=f"{name}.conv2")(out)
        out = layers.BatchNormalization(axis=3, momentum=BN_MOMENTUM, epsilon=1e-5, name=f"{name}.bn2")(out)
        out = layers.Activation('relu', name=f"{name}.relu2")(out)

        out = layers.Conv2D(planes * Bottleneck.expansion, 1, strides=1, padding='VALID', use_bias=False, name=f"{name}.conv3")(out)
        out = layers.BatchNormalization(axis=3, momentum=BN_MOMENTUM, epsilon=1e-5, name=f"{name}.bn3")(out)

        if downsample is not None:
            residual = downsample(x)

        out += residual
        out = layers.Activation('relu', name=f"{name}.relu3")(out)

        return out


def _check_branches(num_branches, blocks, num_blocks,
                    num_inchannels, num_channels):
    if num_branches != len(num_blocks):
        error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
            num_branches, len(num_blocks))
        logger.error(error_msg)
        raise ValueError(error_msg)

    if num_branches != len(num_channels):
        error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
            num_branches, len(num_channels))
        logger.error(error_msg)
        raise ValueError(error_msg)

    if num_branches != len(num_inchannels):
        error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
            num_branches, len(num_inchannels))
        logger.error(error_msg)
        raise ValueError(error_msg)


def CB(x, planes, k=1, strides=1, padding='VALID', bn_momentum=BN_MOMENTUM, name=None):
    x = layers.Conv2D(planes, k, strides=strides, padding=padding, use_bias=False, name=f"{name}.0")(x)
    x = layers.BatchNormalization(axis=3, momentum=bn_momentum, epsilon=1e-5, name=f"{name}.1")(x)
    return x

def CBU(x, planes, scalar, name=None):
    x = layers.Conv2D(planes, 1, strides=1, padding='VALID', use_bias=False, name=f"{name}.0")(x)
    x = layers.BatchNormalization(axis=3,momentum=BN_MOMENTUM, epsilon=1e-5, name=f"{name}.1")(x)
    x = layers.UpSampling2D(scalar, interpolation='nearest', name=f"{name}.2")(x)
    return x

def CBR(x, planes, k=1, strides=1, padding='VALID', bn_momentum=BN_MOMENTUM, name=None):
    x = layers.Conv2D(planes, k, strides=strides, padding=padding, use_bias=False, name=f"{name}.0")(x)
    x = layers.BatchNormalization(axis=3, momentum=bn_momentum, epsilon=1e-5, name=f"{name}.1")(x)
    x = layers.Activation('relu', name=f"{name}.2")(x)
    return x

def SEQ(x, seq):
    for op in seq:
        x = op(x)
    return x

def _make_one_branch(branch_index, block, num_blocks, num_channels, num_inchannels, stride=1, name=None):
    def branch(x, name):
        if block is Bottleneck:
            b = "Bottleneck"
        elif block is BasicBlock:
            b = "BasicBlock"
        else:
            raise Exception("wrong block type")
        downsample = None
        if stride != 1 or \
           num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = partial(CB,
                                 planes=num_channels[branch_index] * block.expansion,
                                 strides=stride,
                                 name=f"{name}.0.downsample"
                                 )

        x = block.func(x, num_channels[branch_index], stride=stride, downsample=downsample, name=f"{name}.0")
        for i in range(1, num_blocks[branch_index]):
            x = block.func(x, num_channels[branch_index], name=f"{name}.{i}")
        return x
    return functools.partial(branch, name=name)


def _make_branches(num_branches, block, num_blocks, num_channels, num_inchannels, name=None):
    branches = []

    for i in range(num_branches):
        branches.append(
            _make_one_branch(i, block, num_blocks, num_channels, num_inchannels, name=f"{name}.branches.{i}")
        )

    return branches


def _make_fuse_layers(num_branches, num_inchannels, multi_scale_output, name=None):

    if num_branches == 1:
        return None

    fuse_layers = []
    for i in range(num_branches if multi_scale_output else 1):
        fuse_layer = []
        for j in range(num_branches):
            if j > i:
                tmp_scalar = int(2 ** (j - i))
                planes = num_inchannels[i]
                fuse_layer.append(functools.partial(CBU,
                                                    planes=planes,
                                                    scalar=tmp_scalar,
                                                    name=f"{name}.fuse_layers.{i}.{j}"
                                                    ))
            elif j == i:
                fuse_layer.append(None)
            else:
                conv3x3s = []
                for k in range(i - j):
                    if k == i - j - 1:
                        planes = num_inchannels[i]
                        conv3x3s.append(functools.partial(CB,
                                                          planes=planes,
                                                          k=3,
                                                          strides=2,
                                                          padding='SAME',
                                                          name=f"{name}.fuse_layers.{i}.{j}.{k}"
                                                          ))
                    else:
                        planes = num_inchannels[j]
                        conv3x3s.append(functools.partial(CBR,
                                                          planes=planes,
                                                          k=3,
                                                          strides=2,
                                                          padding='SAME',
                                                          name=f"{name}.fuse_layers.{i}.{j}.{k}"
                                                          ))

                fuse_layer.append(functools.partial(SEQ,seq=conv3x3s))

        fuse_layers.append(fuse_layer)

    return fuse_layers


def HighResolutionModule(x, num_branches, branches, fuse_layers, name=None):

    if num_branches == 1:
        return [branches[0](x[0])]

    for i in range(num_branches):
        x[i] = branches[i](x[i])

    x_fuse = []

    for i in range(len(fuse_layers)):
        y = x[0] if i == 0 else fuse_layers[i][0](x[0])
        for j in range(1, num_branches):
            if i == j:
                y = y + x[j]
            else:
                y = y + fuse_layers[i][j](x[j])
        x_fuse.append(layers.Activation('relu', name=f"{name}.{i}.hr_relu")(y))

    return x_fuse



blocks_dict = {
    'BASIC': BasicBlock,  # inplane -> plane
    'BOTTLENECK': Bottleneck  # inplane -> plane * expansion
}


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

extra32 = {"PRETRAINED_LAYERS": ['conv1', 'bn1', 'conv2', 'bn2', 'layer1', 'transition1', 'stage2', 'transition2', 'stage3', 'transition3', 'stage4'],
         "FINAL_CONV_KERNEL": 1,
         "STAGE2": {"NUM_MODULES": 1, "NUM_BRANCHES": 2, "BLOCK": "BASIC",
                    "NUM_BLOCKS":[4, 4], "NUM_CHANNELS":[32, 64], "FUSE_METHOD": "SUM"},
         "STAGE3": {"NUM_MODULES": 4, "NUM_BRANCHES": 3, "BLOCK": "BASIC",
                    "NUM_BLOCKS": [4, 4, 4], "NUM_CHANNELS": [32, 64, 128], "FUSE_METHOD":"SUM"},
         "STAGE4": {"NUM_MODULES": 3, "NUM_BRANCHES": 4, "BLOCK": "BASIC",
                    "NUM_BLOCKS": [4, 4, 4, 4], "NUM_CHANNELS": [32, 64, 128, 256], "FUSE_METHOD":"SUM"}}

extra48 = {"PRETRAINED_LAYERS": ['conv1', 'bn1', 'conv2', 'bn2', 'layer1', 'transition1', 'stage2', 'transition2', 'stage3', 'transition3', 'stage4'],
         "FINAL_CONV_KERNEL": 1,
         "STAGE2": {"NUM_MODULES": 1, "NUM_BRANCHES": 2, "BLOCK": "BASIC",
                    "NUM_BLOCKS":[4, 4], "NUM_CHANNELS":[48, 96], "FUSE_METHOD": "SUM"},
         "STAGE3": {"NUM_MODULES": 4, "NUM_BRANCHES": 3, "BLOCK": "BASIC",
                    "NUM_BLOCKS": [4, 4, 4], "NUM_CHANNELS": [48, 96, 192], "FUSE_METHOD":"SUM"},
         "STAGE4": {"NUM_MODULES": 3, "NUM_BRANCHES": 4, "BLOCK": "BASIC",
                    "NUM_BLOCKS": [4, 4, 4, 4], "NUM_CHANNELS": [48, 96, 192, 384], "FUSE_METHOD":"SUM"}}


def _make_transition_layer(num_channels_pre_layer, num_channels_cur_layer, name=None):
    num_branches_cur = len(num_channels_cur_layer)
    num_branches_pre = len(num_channels_pre_layer)

    transition_layers = []
    for i in range(num_branches_cur):
        if i < num_branches_pre:
            if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                outchannels = num_channels_cur_layer[i]
                transition_layers.append(functools.partial(CBR,
                                                           planes=outchannels,
                                                           k=3,
                                                           padding='SAME',
                                                           name=f"{name}.{i}"
                                                           ))
            else:
                transition_layers.append(None)
        else:
            conv3x3s = []
            for j in range(i + 1 - num_branches_pre):
                inchannels = num_channels_pre_layer[-1]
                outchannels = num_channels_cur_layer[i] \
                    if j == i - num_branches_pre else inchannels
                conv3x3s.append(functools.partial(CBR,
                                                  planes=outchannels,
                                                  k=3,
                                                  strides=2,
                                                  padding='SAME',
                                                  name=f"{name}.{i}.{j}"
                                                  ))

            transition_layers.append(functools.partial(SEQ, seq=conv3x3s))

    # fun1 = transition_layers[0]
    # input = keras.Input(shape=(128, 128, 256))
    # o1 = fun1(input)
    # m1 = keras.Model(inputs=input, outputs=o1)

    return transition_layers


def _make_layer(block, planes, blocks, stride=1, name=None):
    def layer(x, name):
        if block is Bottleneck:
            b = "Bottleneck"
        elif block is BasicBlock:
            b = "BasicBlock"
        else:
            raise Exception("wrong block type")
        downsample = None
        if stride != 1 or PoseHighResolutionNet.inplanes != planes * block.expansion:
            downsample = partial(CB,
                                 planes=planes * block.expansion,
                                 strides=stride,
                                 name=f"{name}.0.downsample"
                                 )
        x = block.func(x, planes, stride, downsample, name=f"{name}.0")
        PoseHighResolutionNet.inplanes = planes * block.expansion
        for i in range(1, blocks):
            x = block.func(x, planes, name=f"{name}.{i}")
        return x
    return functools.partial(layer, name=name)

def _make_stage(layer_config, num_inchannels,
                multi_scale_output=True, name=None):
    num_modules = layer_config['NUM_MODULES']
    num_branches = layer_config['NUM_BRANCHES']
    num_blocks = layer_config['NUM_BLOCKS']
    num_channels = layer_config['NUM_CHANNELS']
    block = blocks_dict[layer_config['BLOCK']]
    fuse_method = layer_config['FUSE_METHOD']

    modules = []
    for i in range(num_modules):
        # multi_scale_output is only used last module
        if not multi_scale_output and i == num_modules - 1:
            reset_multi_scale_output = False
        else:
            reset_multi_scale_output = True

        # _check_branches(num_branches, blocks, num_blocks, num_inchannels, num_channels)
        branches = _make_branches(num_branches, block, num_blocks, num_channels, num_inchannels, name=f"{name}.{i}")
        fuse_layers = _make_fuse_layers(num_branches, num_inchannels, reset_multi_scale_output, name=f"{name}.{i}")

        modules.append(functools.partial(HighResolutionModule,
                                         num_branches=num_branches,
                                         branches=branches,
                                         fuse_layers=fuse_layers,
                                         name=f"{name}.{i}",
                                         ))
    # def module(x):
    #     for m in modules:
    #         x = m(x)
    #     return x

    return functools.partial(SEQ, seq=modules), num_inchannels


class PoseHighResolutionNet():
    inplanes = 64

    @staticmethod
    def func(x, extra, heads):  # 1,3,512,512
        # define
        extra = CN(init_dict=extra)
        layer1 = _make_layer(Bottleneck, 64, 4, name="layer1")
        stage2_cfg = extra['STAGE2']
        num_channels = stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[stage2_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        # fun1 = transition1[0]
        # input = keras.Input(shape=(128, 128, 256))
        # o1 = fun1(input)
        # m1 = keras.Model(inputs=input, outputs=o1)
        transition1 = _make_transition_layer([256], num_channels, name="transition1")
        stage2, pre_stage_channels = _make_stage(stage2_cfg, num_channels, name="stage2")

        stage3_cfg = extra['STAGE3']
        num_channels = stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[stage3_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        transition2 = _make_transition_layer(
            pre_stage_channels, num_channels, name="transition2")
        stage3, pre_stage_channels = _make_stage(
            stage3_cfg, num_channels, name="stage3")

        stage4_cfg = extra['STAGE4']
        num_channels = stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[stage4_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        transition3 = _make_transition_layer(
            pre_stage_channels, num_channels, name="transition3")
        stage4, pre_stage_channels = _make_stage(
            stage4_cfg, num_channels, multi_scale_output=False, name="stage4")

        # self.final_layer = nn.Conv2d(
        #     in_channels=pre_stage_channels[0],
        #     out_channels=cfg['MODEL']['NUM_JOINTS'],
        #     kernel_size=extra['FINAL_CONV_KERNEL'],
        #     stride=1,
        #     padding=1 if extra['FINAL_CONV_KERNEL'] == 3 else 0
        # )
        fcs = []
        for head in heads:
            cls_num = heads[head]
            def fc(xx, name):
                xx = layers.Conv2D(256, 3, strides=1, padding='SAME', use_bias=True, name=f"{name}.0")(xx)  # 32 => 256
                xx = layers.Activation('relu', name=f"{name}.1")(xx)
                xx = layers.Conv2D(cls_num, 1, strides=1, padding='VALID', use_bias=True, name=f"{name}.2", dtype=tf.float32)(xx)  # 256 => cls_num
                return xx

            # if 'hm' in head:
            #     fc[-1].bias.data.fill_(-2.19)
            # else:
            #     fill_fc_weights(fc)
            fcs.append(functools.partial(fc, name=head))


        pretrained_layers = extra['PRETRAINED_LAYERS']

        x = layers.Conv2D(64, 3, strides=2, padding='SAME', use_bias=False, name="conv1")(x)  # 3=>64
        x = layers.BatchNormalization(axis=3, momentum=BN_MOMENTUM, epsilon=1e-5, name="bn1")(x)
        x = layers.Activation('relu', name="relu1")(x)
        x = layers.Conv2D(64, 3, strides=2, padding='SAME', use_bias=False, name="conv2")(x)  # 64=>64
        x = layers.BatchNormalization(axis=3, momentum=BN_MOMENTUM, epsilon=1e-5, name="bn2")(x)
        x = layers.Activation('relu', name="relu2")(x)
        x = layer1(x)
        
        x_list = []
        for i in range(stage2_cfg['NUM_BRANCHES']):
            if transition1[i] is not None:
                x_list.append(transition1[i](x))
            else:
                x_list.append(x)
        y_list = stage2(x_list)

        x_list = []
        for i in range(stage3_cfg['NUM_BRANCHES']):
            if transition2[i] is not None:
                x_list.append(transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = stage3(x_list)

        x_list = []
        for i in range(stage4_cfg['NUM_BRANCHES']):
            if transition3[i] is not None:
                x_list.append(transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = stage4(x_list)

        # x = self.final_layer(y_list[0])

        # z = []
        # for fc in fcs:
        #     z.append(fc(y_list[0]))
        #
        # return z

        o = y_list[0]
        hm = tf.math.sigmoid(fcs[0](o))
        xy = fcs[1](o)
        em = fcs[2](o)
        hm_pool = layers.MaxPooling2D(3, strides=1, padding="SAME")(hm)

        return [hm, xy, em, hm_pool]


if __name__ == "__main__":

    num_classes = 2
    x = layers.Input(shape=(512, 512, 3))
    out = PoseHighResolutionNet.func(x, extra32, heads={"hm": num_classes, "reg": 2, "em": 2})
    model = keras.Model(x, out)
    model.summary()

    # import torch
    # from pose_hrnet import get_pose_net

    # num_classes = 2
    # x = layers.Input(shape=(512,512,3))
    # out = PoseHighResolutionNet.func(x, extra32, heads={"hm": num_classes, "em": 2, "reg": 2})
    # model = keras.Model(x,out)

    # state_dict_ = torch.load("/home/fsw/Documents/codes/CenterNet_match/foot_robot_512_512_20.pth", map_location=torch.device('cpu'))
    # state_dict = OrderedDict()
    # for k in state_dict_:
    #     if k.startswith('module') and not k.startswith('module_list'):
    #         state_dict[k[7:]] = state_dict_[k]
    #     else:
    #         state_dict[k] = state_dict_[k]
    #
    # # filter bn.num_batches_tracked
    # filtered_state_dict = {k:v for k,v in state_dict.items() if 'num_batches_tracked' not in k}
    #
    #
    # keras_weights = []
    # for i,w in enumerate(model.weights):
    #     if "/kernel:0" in w.name:
    #         n = w.name.replace("/kernel:0", ".weight")
    #     elif "/bias:0" in w.name:
    #         n = w.name.replace("/bias:0", ".bias")
    #     elif "/gamma:0" in w.name:
    #         n = w.name.replace("/gamma:0", ".weight")
    #     elif "/beta:0" in w.name:
    #         n = w.name.replace("/beta:0", ".bias")
    #     elif "/moving_mean:0" in w.name:
    #         n = w.name.replace("/moving_mean:0", ".running_mean")
    #     elif "moving_variance:0" in w.name:
    #         n = w.name.replace("/moving_variance:0", ".running_var")
    #     weight = filtered_state_dict[n]
    #     if weight.dim() == 4:
    #         # conv2d layer: Torch (out,in, h, w) Keras (h,w,in,out)
    #         weight = weight.permute(2, 3, 1, 0).numpy()
    #     else:
    #         weight = weight.numpy()
    #     keras_weights.append(weight)
    #
    # model.set_weights(keras_weights)
    # i = tf.ones([1,512,512,3],dtype=tf.float32)
    # results1 = model.predict(i)
    # results1 = [r.numpy() for r in results1]
    # model.save("model")

    # keras_layers = ['conv1','bn1','relu1','conv2','bn2','relu2',
    #                 'layer1.3.relu3',
    #                 'stage2.0.0.hr_relu', 'stage2.0.1.hr_relu',
    #                 'stage3.3.0.hr_relu', 'stage3.3.1.hr_relu', 'stage3.3.2.hr_relu',
    #                 'stage4.2.0.hr_relu'
    #                 ]
    # keras_layers = [model.get_layer(k) for k in keras_layers]
    # submodels = {}
    # for layer in keras_layers:
    #     submodels[layer.name] = tf.keras.models.Model(inputs=model.input, outputs=layer.output)
    #
    # keras_activation = {k: m.predict(i, verbose=0) for k, m in submodels.items()}
    #
    # # model = MyModel()
    # pytorch_model = get_pose_net(num_layers=32, heads={"hm": num_classes, "em": 2, "reg": 2})
    # pytorch_model.load_state_dict(state_dict)
    # pytorch_model = pytorch_model.eval()
    #
    # torch_activation = {}
    # def get_activation(name):
    #     def hook(model, input, output):
    #         if isinstance(output, list):
    #             output = [o.permute(0, 2, 3, 1).detach().numpy() for o in output]
    #         else:
    #             output = output.permute(0, 2, 3, 1).detach().numpy()
    #         torch_activation[name] = output
    #     return hook
    # for name, layer in pytorch_model.named_children():
    #     layer.register_forward_hook(get_activation(name))
    #
    # with torch.no_grad():
    #     i = torch.ones([1,3,512,512], dtype=torch.float32)
    #     results2 = pytorch_model(i)
    #     results2 = [r.permute(0, 2, 3, 1).numpy() for r in results2]
    #
    # for r1,r2 in zip(results1,results2):
    #     print((r1-r2).max(), (r1-r2).min())
    #     print((r1-r2).mean(), (r1-r2).mean())
    #
    # print("asdf")






    #     results2 = [r.permute(0,2,3,1).numpy() for r in results2]
    # for r1,r2 in zip(results1,results2):
    #     print(r1-r2)

    # img = np.ones([1, 512, 512, 3], dtype=np.uint8)
    # model = keras.models.load_model("model_train/50")
    # out = model(img)
    # print("asdf")



    #
    # with open("keras.txt", 'w') as f:
    #     for i, w in enumerate(model.weights):
    #         if 'conv' in w.name and 'kernel' in w.name:
    #             s = (w.shape[3].value, w.shape[2].value, w.shape[0].value, w.shape[1].value)
    #         else:
    #             s = tuple([w.shape[i].value for i in range(w.shape.ndims)])
    #         print(i, s, w.name, file=f)
    #
    #

    #
    # with open("pytorch.txt", 'w') as f:
    #     for i, (k,v) in enumerate(filtered_state_dict.items()):
    #         print(i, tuple(v.shape), k, file=f)
    #
    # for i,((k1,v1),(k2,v2)) in enumerate(zip(d.items(),filtered_state_dict.items())):
    #     # print(k1,v1,k2,tuple(v2.shape))
    #     # print(v1,tuple(v2.shape))
    #     if v1 != tuple(v2.shape):
    #         print(i, k1)



    # print(lconv)
    # # model.summary()
    # # model.save("model")
    # print("asdf")
    #

    #
    # keras_weights = []
    # for i,(n,p) in enumerate(filtered_state_dict.items()):
    #     # print(i, n, p.shape)
    #     if i in lconv:
    #         # conv2d layer: Torch (out,in,h,w) Keras (h,w,in,out)
    #         print(n,p.shape)
    #         pp = p.permute(2,3,1,0).numpy()
    #     else:
    #         pp = p.numpy()
    #     keras_weights.append(pp)
    #
    # model.set_weights(keras_weights)
    #
    # i = tf.ones([1,512,512,3],dtype=tf.float32)
    # print(model(i))



    # def init_weights(self, pretrained=''):
    #     logger.info('=> init weights from normal distribution')
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    #             nn.init.normal_(m.weight, std=0.001)
    #             for name, _ in m.named_parameters():
    #                 if name in ['bias']:
    #                     nn.init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.BatchNorm2d):
    #             nn.init.constant_(m.weight, 1)
    #             nn.init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.ConvTranspose2d):
    #             nn.init.normal_(m.weight, std=0.001)
    #             for name, _ in m.named_parameters():
    #                 if name in ['bias']:
    #                     nn.init.constant_(m.bias, 0)
    #
    #     if os.path.isfile(pretrained):
    #         pretrained_state_dict = torch.load(pretrained)
    #         logger.info('=> loading pretrained model {}'.format(pretrained))
    #
    #         need_init_state_dict = {}
    #         for name, m in pretrained_state_dict.items():
    #             if name.split('.')[0] in self.pretrained_layers \
    #                or self.pretrained_layers[0] == '*':
    #                 need_init_state_dict[name] = m
    #         self.load_state_dict(need_init_state_dict, strict=False)
    #     elif pretrained:
    #         logger.error('=> please download pre-trained models first!')
    #         raise ValueError('{} is not exist!'.format(pretrained))