from keras.models import Model
from keras.layers import Input, merge
from keras.layers import Dense, Activation, Flatten, BatchNormalization
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.regularizers import l2

def rnpa_bottleneck_layer(act, input_tensor, nb_filters, filter_sz, stage,
    init='glorot_normal', reg=0.0, use_shortcuts=True):

    nb_in_filters, nb_bottleneck_filters = nb_filters

    bn_name = 'bn' + str(stage)
    conv_name = 'conv' + str(stage)
    relu_name = 'eswish' + str(stage)
    merge_name = "merge_hey" + str(stage)

    # batchnorm-relu-conv, from nb_in_filters to nb_bottleneck_filters via 1x1 conv
    if stage>1: # first activation is just after conv1
        x = BatchNormalization(axis=1, name=bn_name+'a')(input_tensor)
        x = Activation(act, name=relu_name+'a')(x)
    else:
        x = input_tensor

    x = Convolution2D(
            nb_bottleneck_filters, 1, 1,
            init=init,
            W_regularizer=l2(reg),
            bias=False,
            name=conv_name+'a'
        )(x)

    # batchnorm-relu-conv, from nb_bottleneck_filters to nb_bottleneck_filters via FxF conv
    x = BatchNormalization(axis=1, name=bn_name+'b')(x)
    x = Activation(act, name=relu_name+'b')(x)
    x = Convolution2D(
            nb_bottleneck_filters, filter_sz, filter_sz,
            border_mode='same',
            init=init,
            W_regularizer=l2(reg),
            bias = False,
            name=conv_name+'b'
        )(x)

    # batchnorm-relu-conv, from nb_in_filters to nb_bottleneck_filters via 1x1 conv
    x = BatchNormalization(axis=1, name=bn_name+'c')(x)
    x = Activation(act, name=relu_name+'c')(x)
    x = Convolution2D(nb_in_filters, 1, 1,
            init=init, W_regularizer=l2(reg),
            name=conv_name+'c'
        )(x)

    # merge
    if use_shortcuts:
        x = merge([x, input_tensor], mode='sum', name=merge_name)

    return x


def ResNetPreAct(act, input_shape=(32, 32, 3), nb_classes=10,
        layer1_params=(5,64,2),
        res_layer_params=(3,16,3),
        final_layer_params=None,
        init='glorot_normal', reg=0.0, use_shortcuts=True
    ):
    """
    Return a new Residual Network using full pre-activation based on the work in
    "Identity Mappings in Deep Residual Networks"  by He et al
    http://arxiv.org/abs/1603.05027
    The following network definition achieves 92.0% accuracy on CIFAR-10 test using
    `adam` optimizer, 100 epochs, learning rate schedule of 1e.-3 / 1.e-4 / 1.e-5 with
    transitions at 50 and 75 epochs:
    ResNetPreAct(layer1_params=(3,128,2),res_layer_params=(3,32,25),reg=reg)
    
    Removed max pooling and using just stride in first convolutional layer. Motivated by
    "Striving for Simplicity: The All Convolutional Net"  by Springenberg et al
    (https://arxiv.org/abs/1412.6806) and my own experiments where I observed about 0.5%
    improvement by replacing the max pool operations in the VGG-like cifar10_cnn.py example
    in the Keras distribution.
    
    Parameters
    ----------
    input_dim : tuple of (C, H, W)
    nb_classes: number of scores to produce from final affine layer (input to softmax)
    layer1_params: tuple of (filter size, num filters, stride for conv)
    res_layer_params: tuple of (filter size, num res layer filters, num res stages)
    final_layer_params: None or tuple of (filter size, num filters, stride for conv)
    init: type of weight initialization to use
    reg: L2 weight regularization (or weight decay)
    use_shortcuts: to evaluate difference between residual and non-residual network
    """

    sz_L1_filters, nb_L1_filters, stride_L1 = layer1_params
    sz_res_filters, nb_res_filters, nb_res_stages = res_layer_params
    
    use_final_conv = (final_layer_params is not None)
    if use_final_conv:
        sz_fin_filters, nb_fin_filters, stride_fin = final_layer_params
        sz_pool_fin = int(input_shape[1] / (stride_L1 * stride_fin))
    else:
        sz_pool_fin = int(input_shape[1] / (stride_L1))


    img_input = Input(shape=input_shape, name='cifar')

    x = Convolution2D(
            nb_L1_filters, sz_L1_filters, sz_L1_filters,
            border_mode='same',
            subsample=(stride_L1, stride_L1),
            init=init,
            W_regularizer=l2(reg),
            bias=False,
            name='conv0'
        )(img_input)
    x = BatchNormalization(axis=1, name='bn0')(x)
    x = Activation(act, name='relu0')(x)

    for stage in range(1,nb_res_stages+1):
        x = rnpa_bottleneck_layer(
                act, x,
                (nb_L1_filters, nb_res_filters),
                sz_res_filters, 
                stage,
                init=init, 
                reg=reg, 
                use_shortcuts=use_shortcuts
            )


    x = BatchNormalization(axis=1, name='bnF')(x)
    x = Activation(act, name='reluF')(x)

    if use_final_conv:
        x = Convolution2D(
                nb_fin_filters, sz_fin_filters, sz_fin_filters,
                border_mode='same',
                subsample=(stride_fin, stride_fin),
                init=init,
                W_regularizer=l2(reg),
                name='convF'
            )(x)

    x = AveragePooling2D((sz_pool_fin,sz_pool_fin), name='avg_pool')(x)

    x = Flatten(name='flat')(x)
    x = Dense(10, activation='softmax', name='fc10')(x)

    return Model(img_input, x, name='rnpa')