def layer(k, x):
    x = Activation('selu')(x)
    x = Conv2D(k, 3, padding='same', kernel_initializer='lecun_normal', kernel_regularizer=regularizers.l2(1e-4))(x)
    return AlphaDropout(0.2)(x)

def transitionDown(filters, x):
    x = Activation('selu')(x)
    x = Conv2D(filters, 1, padding='same', kernel_initializer='lecun_normal', kernel_regularizer=regularizers.l2(1e-4))(x)
    squeeze = GlobalAveragePooling2D()(x)
    excitation = Dense(units=filters // 16)(squeeze)
    excitation = Activation('relu')(excitation)
    excitation = Dense(units=filters)(excitation)
    excitation = Activation('sigmoid')(excitation)
    excitation = Reshape((1, 1, filters))(excitation)
    scale = multiply([x, excitation])
    x = AlphaDropout(0.2)(scale)
    x = MaxPooling2D(pool_size=2)(x)
    return x

def transitionUp(filters, x):
    
    squeeze_1 = GlobalAveragePooling2D()(x)
    excitation_1 = Dense(units=filters // 16)(squeeze_1)
    excitation_1 = Activation('relu')(excitation_1)
    excitation_1 = Dense(units=filters)(excitation_1)
    excitation_1 = Activation('sigmoid')(excitation_1)
    excitation_1 = Reshape((1, 1, filters))(excitation_1)
    x=Conv2DTranspose(filters, 3, padding='same', strides=(2,2), kernel_initializer='lecun_normal', kernel_regularizer=regularizers.l2(1e-4))(x)
    squeeze_2 = GlobalAveragePooling2D()(x)
    excitation_2 = Dense(units=filters // 16)(squeeze_2)
    excitation_2 = Activation('relu')(excitation_2)
    excitation_2 = Dense(units=filters)(excitation_2)
    excitation_2 = Activation('sigmoid')(excitation_2)
    excitation_2 = Reshape((1, 1, filters))(excitation_2)
    excitation = add([excitation_1 , excitation_2])
    scale = multiply([x, excitation])
    return scale

def denseBlock(k, n, x):
    for i in range(n):
        x = concatenate([x, layer(k,x)])
    return x

def pspnet(x):
    x_c1 = AveragePooling2D(pool_size=16, strides=16, name='ave_c1')(x)
    x_c1 = Conv2D(128, 1, padding='same', kernel_initializer='he_normal',name = 'lll-1',kernel_regularizer=regularizers.l2(1e-4))(x_c1)
    x_c1 = BatchNormalization(momentum=0.95, axis=-1)(x_c1)
    x_c1 = Activation(activation='relu')(x_c1)
    #x_c1 = Dropout(0.2)(x_c1)
    up_1 = UpSampling2D(size=(2, 2), name='up_c11')(x_c1)
    x_c1 = UpSampling2D(size=(16, 16), name='up_c1')(x_c1)

    x_c2 = AveragePooling2D(pool_size=8, strides=8, name='ave_c2')(x)
    x_c2 = Concatenate(axis=-1, name='psp_concat_1')([up_1, x_c2])
    x_c2 = Conv2D(128, 1, padding='same', kernel_initializer='he_normal',name = 'lll-2',kernel_regularizer=regularizers.l2(1e-4))(x_c2)
    x_c2 = BatchNormalization(momentum=0.95, axis=-1)(x_c2)
    x_c2 = Activation(activation='relu')(x_c2)
    #x_c2 = Dropout(0.2)(x_c2)
    up_2 = UpSampling2D(size=(2, 2), name='up_c22')(x_c2)
    x_c2 = UpSampling2D(size=(8, 8), name='up_c2')(x_c2)

    x_c3 = AveragePooling2D(pool_size=4, strides=4, name='ave_c3')(x)
    x_c3 = Concatenate(axis=-1, name='psp_concat_2')([up_2, x_c3])
    x_c3 = Conv2D(128, 1, padding='same', kernel_initializer='he_normal',name = 'lll-3',kernel_regularizer=regularizers.l2(1e-4))(x_c3)
    x_c3 = BatchNormalization(momentum=0.95, axis=-1)(x_c3)
    x_c3 = Activation(activation='relu')(x_c3)
    #x_c3 = Dropout(0.2)(x_c3)
    up_3 = UpSampling2D(size=(2, 2), name='up_c33')(x_c3)
    x_c3 = UpSampling2D(size=(4, 4), name='up_c3')(x_c3)

    x_c4 = AveragePooling2D(pool_size=2, strides=2, name='ave_c4')(x)
    x_c4 = Concatenate(axis=-1, name='psp_concat_3')([up_3, x_c4])
    x_c4 = Conv2D(128, 1, padding='same', kernel_initializer='he_normal',name = 'lll-4',kernel_regularizer=regularizers.l2(1e-4))(x_c4)
    x_c4 = BatchNormalization(momentum=0.95, axis=-1)(x_c4)
    x_c4 = Activation(activation='relu')(x_c4)
    #x_c4 = Dropout(0.2)(x_c4)
    x_c4 = UpSampling2D(size=(2, 2), name='up_c4')(x_c4)
    x = Concatenate(axis=-1, name='concat')([x,x_c1, x_c2, x_c3, x_c4])
    return x

def _conv(**conv_params):
    # conv params
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    dilation_rate = conv_params.setdefault('dilation_rate',(1,1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    block = conv_params.setdefault("block", "assp")

    def f(input):
        conv = Conv2D(
                filters=filters,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                kernel_initializer=kernel_initializer,activation='linear')(input)
        return conv
    return f

def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)


def Nest_Net(img_rows= 128, img_cols = 128, color_type=3, num_class=1, deep_supervision=True):

    nb_filter = [32,64,128,256,512]

    global bn_axis
    if K.image_dim_ordering() == 'tf':
      bn_axis = 3
      img_input = Input(shape=(img_rows, img_cols, color_type), name='main_input')
    else:
      bn_axis = 1
      img_input = Input(shape=(color_type, img_rows, img_cols), name='main_input')
    input_1 = Conv2D(32, 3, padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(1e-4))(img_input)
    
    conv1_1 = denseBlock(16, 4, input_1)
    Concat_1 = concatenate([input_1,conv1_1],name = 'jump1',axis=bn_axis)
    trans1_1 = Conv2D(32, 1, padding='same', kernel_initializer='he_normal',name = 'tiao1-1',kernel_regularizer=regularizers.l2(1e-4))(conv1_1)
    pool1 = transitionDown(48, Concat_1)

    conv2_1 = denseBlock(16, 5, pool1)
    Concat_2 = concatenate([pool1,conv2_1],name = 'jump2',axis=bn_axis)
    trans2_1 =  Conv2D(64, 1, padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(1e-4))(conv2_1)
    pool2 = transitionDown(64, conv2_1)
    
    up1_2 = transitionUp(nb_filter[0],conv2_1)
    conv1_2 = concatenate([up1_2, trans1_1], name='merge12', axis=bn_axis)
    conv1_2 = denseBlock(16, 4, conv1_2)
    trans1_2 =  Conv2D(32, 1, padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(1e-4))(conv1_2)

    conv3_1 = denseBlock(16, 7, pool2)
    Concat_3 = concatenate([pool2,conv3_1],name = 'jump2',axis=bn_axis)
    trans3_1 =  Conv2D(96, 1, padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(1e-4))(conv3_1)
    pool3 = transitionDown(128, conv3_1)

    up2_2 = transitionUp(nb_filter[1],conv3_1)
    conv2_2 = concatenate([up2_2, trans2_1], name='merge22', axis=bn_axis)
    conv2_2 = denseBlock(16, 5, conv2_2)
    trans2_2 =  Conv2D(64, 1, padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(1e-4))(conv2_2)


    up1_3 = transitionUp(nb_filter[0],conv2_2)
    conv1_3 = concatenate([up1_3, trans1_1,trans1_2], name='merge13', axis=bn_axis)
    conv1_3 = denseBlock(16, 4, conv1_3)
    trans1_3 =  Conv2D(32, 1, padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(1e-4))(conv1_3)


    conv4_1 = denseBlock(16, 12, pool3)
    trans4_1 =  Conv2D(256, 1, padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(1e-4))(conv4_1)


    up3_2 = transitionUp(nb_filter[2],conv4_1)
    conv3_2 = concatenate([up3_2, trans3_1], name='merge32', axis=bn_axis)
    conv3_2 = denseBlock(16, 7, conv3_2)
    trans3_2 =  Conv2D(96, 1, padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(1e-4))(conv3_2)


    up2_3 = transitionUp(nb_filter[1],conv3_2)
    conv2_3 = concatenate([up2_3, trans2_1,trans2_2], name='merge23', axis=bn_axis)
    conv2_3 = denseBlock(16, 5, conv2_3)
    trans2_3 =  Conv2D(64, 1, padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(1e-4))(conv2_3)


    up1_4 = transitionUp(nb_filter[0],conv2_3)
    conv1_4 = concatenate([up1_4, trans1_1,trans1_2,trans1_3], name='merge14', axis=bn_axis)
    conv1_4 = denseBlock(16, 4, conv1_4)
    trans1_4 =  Conv2D(32, 1, padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(1e-4))(conv1_4)


    conv5_1 = pspnet(conv4_1)
    
    conv4_2 = concatenate([conv5_1, trans4_1], name='merge42', axis=bn_axis)
    conv4_2 = denseBlock(16, 12, conv4_2)


    up3_3 = transitionUp(nb_filter[2],conv4_2)
    conv3_3 = concatenate([up3_3, trans3_1,trans3_2], name='merge33', axis=bn_axis)
    conv3_3 = denseBlock(16 ,7, conv3_3)

    up2_4 = transitionUp(nb_filter[1],conv3_3)
    conv2_4 = concatenate([up2_4, conv2_1,trans2_2,trans2_3], name='merge24', axis=bn_axis)
    conv2_4 = denseBlock(16 ,5, conv2_4)

    up1_5 = transitionUp(16,conv2_4)
    conv1_5 = concatenate([up1_5, trans1_1,trans1_2,trans1_3,trans1_4], name='merge15', axis=bn_axis)
    conv1_5 = denseBlock(16, 4, conv1_5)

    nestnet_output_1 = Conv2D(num_class, (1, 1), activation='sigmoid', name='output_1', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv1_2)
    nestnet_output_2 = Conv2D(num_class, (1, 1), activation='sigmoid', name='output_2', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv1_3)
    nestnet_output_3 = Conv2D(num_class, (1, 1), activation='sigmoid', name='output_3', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv1_4)
    nestnet_output_4 = Conv2D(num_class, (1, 1), activation='sigmoid', name='output_4', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv1_5)

    if deep_supervision:
        model = Model(input=img_input, output=[nestnet_output_1,
                                               nestnet_output_2,
                                               nestnet_output_3,
                                               nestnet_output_4])
    else:
        model = Model(input=img_input, output=[nestnet_output_4])

    return model
