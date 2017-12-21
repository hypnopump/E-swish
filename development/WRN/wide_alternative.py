# Implementation of WRN-28-10
#
# Wide Residual Network
# https://arxiv.org/abs/1605.07146


import numpy as np
import pickle

from keras.callbacks           import LearningRateScheduler
from keras.layers              import Activation, Add, BatchNormalization, Conv2D, Dense, GlobalAveragePooling2D, Input
from keras.models              import Model, save_model
from keras.optimizers          import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers        import l2



def computational_graph(class_size):
    # Utility functions.

    def batch_normalization():
        return BatchNormalization()

    def relu():
        return Activation('relu')

    def conv(filter_size, kernel_size, stride_size=1):
        return Conv2D(filter_size, kernel_size, strides=stride_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(0.0005), use_bias=False)  # ReLUするならウェイトをHe初期化するのが基本らしい。あと、Kerasにはweight decayがなかったのでkernel_regularizerで代替したのたけど、これで正しい？

    def add():
        return Add()

    def global_average_pooling():
        return GlobalAveragePooling2D()

    def dense(unit_size, activation):
        return Dense(unit_size, activation=activation, kernel_regularizer=l2(0.0005))  # Kerasにはweight decayがなかったのでkernel_regularizerで代替したのたけど、これで正しい？

    # Define WRN-28-10.

    def first_residual_unit(filter_size, stride_size):
        return rcompose(batch_normalization(),
                        relu(),
                        ljuxt(rcompose(conv(filter_size, 3, stride_size),
                                       batch_normalization(),
                                       relu(),
                                       conv(filter_size, 3, 1)),
                              rcompose(conv(filter_size, 1, stride_size))),
                        add())

    def residual_unit(filter_size):
        return rcompose(ljuxt(rcompose(batch_normalization(),
                                       relu(),
                                       conv(filter_size, 3),
                                       batch_normalization(),
                                       relu(),
                                       conv(filter_size, 3)),
                              identity),
                        add())

    def residual_block(filter_size, stride_size, unit_size):
        return rcompose(first_residual_unit(filter_size, stride_size),
                        rcompose(*repeatedly(partial(residual_unit, filter_size), unit_size - 1)))

    k = 10  # 論文によれば、CIFAR-10に最適な値は10。
    n =  4  # 論文によれば、CIFAR-10に最適な値は4。WRN-28-10の28はconvの数で、「1（入り口のconv）+ 3 * n * 2 + 3（ショートカットの中のconv？）」みたい。n = 4 で28。

    return rcompose(conv(16, 3),
                    residual_block(16 * k, 1, n),
                    residual_block(32 * k, 2, n),
                    residual_block(64 * k, 2, n),
                    batch_normalization(),
                    relu(),
                    global_average_pooling(),
                    dense(class_size, 'softmax'))


def main():
    (x_train, y_train), (x_validation, y_validation) = load_data()

    model = Model(*juxt(identity, computational_graph(y_train.shape[1]))(Input(shape=x_train.shape[1:])))
    model.compile(loss='categorical_crossentropy', optimizer=SGD(momentum=0.9), metrics=['accuracy'])  # 論文にはnesterov=Trueだと書いてあったけど、コードだとFalseだった……。

    model.summary()
    # plot_model(model, to_file='./results/model.png')

    train_data      = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True, width_shift_range=0.125, height_shift_range=0.125, horizontal_flip=True)
    validation_data = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)

    for data in (train_data, validation_data):
        data.fit(x_train)  # 実用を考えると、x_validationでのfeaturewiseのfitは無理だと思う……。

    batch_size = 128
    epoch_size = 200

    results = model.fit_generator(train_data.flow(x_train, y_train, batch_size=batch_size),
                                  steps_per_epoch=x_train.shape[0] // batch_size,
                                  epochs=epoch_size,
                                  callbacks=[LearningRateScheduler(partial(getitem, tuple(take(epoch_size, concat(repeat(0.1, 60), repeat(0.02, 60), repeat(0.004, 40), repeat(0.0008))))))],
                                  validation_data=validation_data.flow(x_validation, y_validation, batch_size=batch_size),
                                  validation_steps=x_validation.shape[0] // batch_size)

    with open('./results/history.pickle', 'wb') as f:
        pickle.dump(results.history, f)

    save_model(model, './results/model.h5')

    del model


if __name__ == '__main__':
    main()