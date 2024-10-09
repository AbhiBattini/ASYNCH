import keras
from keras import Input, Model
from keras.src.layers import MultiHeadAttention, Conv1D, add, Activation, multiply, GlobalAveragePooling1D, \
    BatchNormalization, Dense, SeparableConv1D, MaxPooling1D, Conv1DTranspose, concatenate
​
with tpu_strategy.scope():
    def batchnorm_relu(inputs):
        x = BatchNormalization()(inputs)
        x = keras.activations.gelu(x)
        return x
​
​
    def se_block(in_block, ch, ratio=16):
        x = GlobalAveragePooling1D()(in_block)
        x = Dense(ch // ratio)(x)
        x = Activation('relu')(x)
        x = Dense(ch)(x)
        x = Activation('sigmoid')(x)
        return multiply([in_block, x])
​
​
    def residual_block(inputs, num_filters, strides=1):
        x = batchnorm_relu(inputs)
        x = Conv1D(num_filters, 3, padding="same", strides=strides, kernel_initializer=keras.initializers.HeNormal(),
                   dilation_rate=2)(x)
        x = se_block(x, num_filters)
        x = batchnorm_relu(x)
        x = Conv1D(num_filters, 3, padding="same", strides=strides, kernel_initializer=keras.initializers.HeNormal(),
                   dilation_rate=2)(x)
        x = se_block(x, num_filters)
        s = Conv1D(num_filters, 1, padding="same", strides=strides, kernel_initializer=keras.initializers.HeNormal()) \
            (inputs)
        s = se_block(s, num_filters)
        x = x + s
        return x
​
​
    def GroupConv(input_tensor, num_filters, kernelsize, dilation):
        x = Conv1D(num_filters, kernel_size=kernelsize, padding='same', kernel_initializer=keras.initializers.HeNormal(),
                   dilation_rate=dilation)(input_tensor)
        x = se_block(x, num_filters)
        x = BatchNormalization()(x)
        x = keras.activations.gelu(x)
        x = Conv1D(num_filters, kernel_size=kernelsize, padding='same', kernel_initializer=keras.initializers.HeNormal(),
                   dilation_rate=dilation)(x)
        x = se_block(x, num_filters)
        x = BatchNormalization()(x)
        x = keras.activations.gelu(x)
        return x
​
​
    def DeepConv(input_tensor, num_filters, kernelsize, dilation):
        x = SeparableConv1D(filters=num_filters, kernel_size=kernelsize, dilation_rate=dilation, padding='same',
                            bias_initializer=keras.initializers.HeNormal(),
                            depthwise_initializer=keras.initializers.HeNormal(),
                            pointwise_initializer=keras.initializers.HeNormal())(input_tensor)
        x = se_block(x, num_filters)
        x = BatchNormalization()(x)
        x = keras.activations.gelu(x)
        x = Conv1D(filters=num_filters, kernel_size=kernelsize, padding='same',
                   kernel_initializer=keras.initializers.HeNormal(), dilation_rate=dilation)(x)
        x = se_block(x, num_filters)
        x = BatchNormalization()(x)
        x = keras.activations.gelu(x)
        return x
​
​
    def encoder_block(input_tensor, num_filters, stride, kernelsize, maxkernelsize):
        encoder = DeepConv(input_tensor, num_filters, kernelsize, 2)
        encoder_pool = MaxPooling1D(maxkernelsize, strides=stride)(encoder)
        return encoder_pool, encoder
​
​
    def decoder_block(input_tensor, concat_tensor, num_filters, kernelsize, stride):
        decoder = Conv1DTranspose(num_filters, 2, strides=stride, padding='same', kernel_initializer=
        keras.initializers.LecunUniform())(input_tensor)
        decoder = concatenate([decoder, concat_tensor], axis=-1)
        decoder = GroupConv(decoder, num_filters, kernelsize, 2)
        return decoder
​
​
    def attention_gate(inp_1, inp_2, n_intermediate_channels):
        inp_1_conv = Conv1D(n_intermediate_channels, 3, padding='same',
                            kernel_initializer=keras.initializers.LecunUniform())(inp_1)
        inp_2_conv = Conv1D(n_intermediate_channels, 3, padding='same',
                            kernel_initializer=keras.initializers.LecunUniform())(inp_2)
        f = add([inp_1_conv, inp_2_conv])
        f = keras.activations.gelu(f)
        g = Conv1D(1, 1, padding='same', kernel_initializer=keras.initializers.LecunUniform())(f)
        gate = Activation('sigmoid')(g)
        return multiply([inp_2, gate])
​
​
    def Denoise(input_shape):
        inputs = Input(shape=input_shape)
        inputs = MultiHeadAttention(num_heads=1, key_dim=1)(inputs, inputs, inputs)
        encoder_pool0, encoder0 = encoder_block(inputs, 16, 5, 7, 5)
        encoder_pool1, encoder1 = encoder_block(encoder_pool0, 32, 2, 5, 2)
        encoder_pool2, encoder2 = encoder_block(encoder_pool1, 64, 2, 3, 2)
        center = DeepConv(encoder_pool2, 64, 3, 2)
        # Upsampling and establishing the skip connections
        decoder2 = decoder_block(center, encoder2, 32, 3, 2)
        res2 = residual_block(decoder2, 32)  # 32
        decoder1 = decoder_block(res2, encoder1, 32, 5, 2)
        res1 = residual_block(decoder1, 16)
        decoder0 = decoder_block(res1, encoder0, 16, 7, 5)
        # Output
        output0 = Conv1D(1, 1, kernel_initializer=keras.initializers.HeNormal())(decoder0)
        model = Model(inputs=[inputs], outputs=[output0])
        return model
