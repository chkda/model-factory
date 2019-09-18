import tensorflow as tf 

def Vgg16(inp,input_shape,num_classes,d_rate=0.2):
    base = tf.keras.applications.VGG16(weights='imagenet',include_top=False,input_shape=input_shape)
    mod = base(inp)
    mod = tf.keras.layers.GlobalAveragePooling2D()(mod)
    mod = tf.keras.layers.Dense(units=1024,activation=tf.nn.relu,name='Dense_1')(mod)
    mod = tf.keras.layers.Dropout(rate=d_rate,name='Dropout_1')(mod)
    mod = tf.keras.layers.Dense(units=512,activation=tf.nn.relu,name='Dense_2')(mod)
    mod = tf.keras.layers.Dropout(rate=d_rate,name='Dropout_2')(mod)
    mod = tf.keras.layers.Dense(units=num_classes,name='output')(mod)

    return mod

def Vgg19(inp,input_shape,num_classes,d_rate=0.2):
    base = tf.keras.applications.VGG19(weights='imagenet',include_top=False,input_shape=input_shape)
    mod = base(inp)
    mod = tf.keras.layers.GlobalAveragePooling2D()(mod)
    mod = tf.keras.layers.Dense(units=1024,activation=tf.nn.relu,name='Dense_1')(mod)
    mod = tf.keras.layers.Dropout(rate=d_rate,name='Dropout_1')(mod)
    mod = tf.keras.layers.Dense(units=512,activation=tf.nn.relu,name='Dense_2')(mod)
    mod = tf.keras.layers.Dropout(rate=d_rate,name='Dropout_2')(mod)
    mod = tf.keras.layers.Dense(units=num_classes,name='output')(mod)

    return mod