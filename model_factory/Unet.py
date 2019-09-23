import tensorflow as tf 


def UNET(inp,num_classes):
    ## down sampling layer (Feature extraction)
    conv_16 = tf.keras.layers.Conv2D(filters=16,kernel=(3,3),padding='same',strides=1,activation=tf.nn.relu,name='Conv16')(inp)
    max_pool_1 = tf.keras.layers.MaxPool2D(pool_size=(2,2),padding='same',strides=1,name='MaxPool1')(conv_16)
    conv_32 = tf.keras.layers.Conv2D(filters=32,kernel=(3,3),padding='same',strides=1,
    activation=tf.nn.relu,name='Conv32')(max_pool_1)
    max_pool_2 = tf.keras.layers.MaxPool2D(pool_size=(2,2),padding='same',strides=1,name='MaxPool2')(conv_32)
    conv_64 = tf.keras.layers.Conv2D(filters=64,kernel=(3,3),padding='same',strides=1,activation=tf.nn.relu,name='Conv64')(max_pool_2)
    max_pool_3 = tf.keras.layers.MaxPool2D(pool_size=(2,2),padding='same',strides=1,
    name='MaxPool3')(conv_64)
    conv_128 = tf.keras.layers.Conv2D(filters=128,kernel=(3,3),padding='same',strides=1,activation=tf.nn.relu,name='Conv128')
    max_pool_4 = tf.keras.layers.MaxPool2D(pool_size=(2,2),padding='same',strides=1,
    name='MaxPool4')

    ##Upsample
    upsamp_1 = tf.keras.layers.Upsample(size=(2,2))(max_pool_4)
    upconv_64_1 = tf.keras.layers.Conv2D(filters=64,kernel=(3,3),padding='same',strides=1,activation=tf.nn.relu,name='upconv_64_1')(upsamp_1)
    concat_64 = tf.keras.layers.concatenate([upconv_64_1,conv_64],axis=3)
    upconv_64_2 = tf.keras.layers.Conv2D(filters=64,kernel=(3,3),padding='same',strides=1,activation=tf.nn.relu,name='upconv_64_2')

    upsamp_2 = tf.keras.layers.Upsample(size=(2,2))(upconv_64_2)
    upconv_32_1 = tf.keras.layers.Conv2D(filters=32,kernel=(3,3),padding='same',strides=1,activation=tf.nn.relu,name='upconv_32_1')(upsamp_2)
    concat_32 = tf.keras.layers.concatenate([upconv_32_1,conv_32],axis=3)
    upconv_32_2 = tf.keras.layers.Conv2D(filters=32,kernel=(3,3),padding='same',strides=1,activation=tf.nn.relu,name='upconv_32_2')

    upsamp_3 = tf.keras.layers.Upsample(size=(2,2))(upconv_32_2)
    upconv_16_1 = tf.keras.layers.Conv2D(filters=16,kernel=(3,3),padding='same',strides=1,activation=tf.nn.relu,name='upconv_16_1')(upsamp_3)
    concat_16 = tf.keras.layers.concatenate([upconv_16_1,conv_16],axis=3)
    upconv_16_2 = tf.keras.layers.Conv2D(filters=16,kernel=(3,3),padding='same',strides=1,activation=tf.nn.relu,name='upconv_16_2')

    final_map = tf.keras.layers.Conv2D(filters=num_classes,kernel=(1,1),padding='same',strides=1,name='output_mask')

    return final_map



