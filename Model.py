import keras
import tensorflow as tf
from keras import Input, Model, Layer
from keras.src.layers import MultiHeadAttention, Conv1D, add, Activation, multiply, GlobalAveragePooling1D, \
    BatchNormalization, Dense, SeparableConv1D, Embedding, LayerNormalization, ReLU, GlobalAveragePooling2D
from tensorflow.keras import layers
from keras_nlp.src.layers import TransformerDecoder, TransformerEncoder

with tpu_strategy.scope():
    class LinearRNNCell(layers.Layer):
        def __init__(self, units, **kwargs):
            super(LinearRNNCell, self).__init__(**kwargs)
            self.units = units
            self.state_size = units

        def build(self, input_shape):
            self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
                                          initializer='glorot_uniform',
                                          name='kernel')
            self.recurrent_kernel = self.add_weight(shape=(self.units, self.units),
                                                    initializer='glorot_uniform',
                                                    name='recurrent_kernel')
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer='zeros',
                                        name='bias')
            super(LinearRNNCell, self).build(input_shape)

        def call(self, inputs, states):
            prev_output = states[0]
            h = tf.matmul(inputs, self.kernel) + tf.matmul(prev_output, self.recurrent_kernel) + self.bias
            return h, [h]

        def get_config(self):
            config = super(LinearRNNCell, self).get_config()
            config.update({'units': self.units})
            return config

    class LinearRNN(layers.RNN):
        def __init__(self, units, **kwargs):
            cell = LinearRNNCell(units)
            super(LinearRNN, self).__init__(cell, return_sequences=True, **kwargs)
            self.units = units

        def get_config(self):
            config = super(LinearRNN, self).get_config()
            config.update({'units': self.units})
            return config


    def batchnorm_relu(inputs):
        x = BatchNormalization()(inputs)
        x = keras.activations.gelu(x)
        return x

    def se_block(in_block, ch, ratio=16):
        x = GlobalAveragePooling1D()(in_block)
        x = Dense(ch // ratio)(x)
        x = Activation('relu')(x)
        x = Dense(ch)(x)
        x = Activation('sigmoid')(x)
        return multiply([in_block, x])

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


    def encoder_block(input_tensor, num_filters, stride, kernelsize, maxkernelsize):
        encoder = DeepConv(input_tensor, num_filters, kernelsize, 2)
        encoder_pool = MaxPooling1D(maxkernelsize, strides=stride)(encoder)
        return encoder_pool, encoder


    def decoder_block(input_tensor, concat_tensor, num_filters, kernelsize, stride):
        decoder = Conv1DTranspose(num_filters, 2, strides=stride, padding='same', kernel_initializer=
        keras.initializers.LecunUniform())(input_tensor)
        decoder = concatenate([decoder, concat_tensor], axis=-1)
        decoder = GroupConv(decoder, num_filters, kernelsize, 2)
        return decoder


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

    def Generator(input_shape):
        inputs = Input(shape=input_shape)
        linear_rnn_layer = LinearRNN(units=2)
        inputs = linear_rnn_layer(inputs)
        # Encoder blocks
        encoder_pool0, encoder0 = encoder_block(inputs, 96, 5, 3, 5)
        encoder_pool1, encoder1 = encoder_block(encoder_pool0, 144, 4, 3, 2)

        # Deep convolutional layer
        center = DeepConv(encoder_pool1, 144, 3, 2)
        # Apply MultiHeadAttention
        center_mha = MultiHeadAttention(num_heads=4, key_dim=144)(center, center, center)

        # Decoder blocks with skip connections
        decoder1 = decoder_block(center_mha, encoder1, 144, 3, 4)
        attn1 = attention_gate(encoder1, decoder1, 144)
        res1 = residual_block(attn1, 144)
        decoder0 = decoder_block(res1, encoder0, 96, 3, 5)
        # Apply MultiHeadAttention
        decoder_mha = MultiHeadAttention(num_heads=4, key_dim=96)(decoder0, decoder0, decoder0)

        x = Conv1D(2, 1, kernel_initializer=keras.initializers.HeNormal())(decoder_mha)
        model = Model(inputs=[inputs], outputs=[x])
        return model
    
    class PositionalEmbedding(Layer):
        def __init__(self, sequence_length, embedding_dim, **kwargs):
            super(PositionalEmbedding, self).__init__(**kwargs)
            self.sequence_length = sequence_length
            self.embedding_dim = embedding_dim

        def build(self, input_shape):
            # Initialize the trainable positional embeddings
            self.positional_embeddings = self.add_weight(
                shape=(self.sequence_length, self.embedding_dim),
                initializer='uniform',
                trainable=True,
                name='positional_embeddings'
            )

        def call(self, inputs):
            # Adding the positional embeddings to the input embeddings
            return inputs + self.positional_embeddings

        def compute_output_shape(self, input_shape):
            # Output shape is the same as the input shape
            return input_shape
    
    class FeedForwardNetworkLayer(layers.Layer):
        def __init__(self, d_model, d_ff, output_dim, **kwargs):
            super(FeedForwardNetworkLayer, self).__init__(**kwargs)
            self.d_model = d_model
            self.d_ff = d_ff
            self.output_dim = output_dim

            # Dense layer for the feed-forward network
            self.dense1 = layers.Dense(d_ff, activation='relu')
            self.dense2 = layers.Dense(output_dim)

        def call(self, inputs):
            # inputs shape: (batch_size, 500, 96)

            # Apply the first dense layer (position-wise feed-forward)
            x = self.dense1(inputs)

            # Apply the second dense layer to get the desired output dimension
            output = self.dense2(x)

            return output

    
    def ASYNCH(input_shape):
        inputs = Input(shape=input_shape)
        linear_rnn_layer = LinearRNN(units=3)
        inputs = linear_rnn_layer(inputs)
        # Encoder blocks
        encoder_pool0, encoder0 = encoder_block(inputs, 96, 5, 3, 5)
        encoder_pool1, encoder1 = encoder_block(encoder_pool0, 144, 4, 3, 2)

        # Deep convolutional layer
        center = DeepConv(encoder_pool1, 144, 3, 2)
        # Apply MultiHeadAttention
        center_mha = MultiHeadAttention(num_heads=4, key_dim=144)(center, center, center)

        # Decoder blocks with skip connections
        decoder1 = decoder_block(center_mha, encoder1, 144, 3, 4)
        attn1 = attention_gate(encoder1, decoder1, 144)
        res1 = residual_block(attn1, 144)
        decoder0 = decoder_block(res1, encoder0, 96, 3, 5)
        # Apply MultiHeadAttention
        decoder_mha = MultiHeadAttention(num_heads=4, key_dim=96)(decoder0, decoder0, decoder0)
        x = TransformerEncoder(num_heads=4, intermediate_dim=96,
                               kernel_initializer=keras.initializers.HeUniform(), normalize_first=True)(decoder_mha)
        x = LayerNormalization()(x)
        x = TransformerDecoder(num_heads=4, intermediate_dim=96,
                               kernel_initializer=keras.initializers.HeUniform(), normalize_first=True)(x)
        x = LayerNormalization()(x)
        x = TransformerEncoder(num_heads=4, intermediate_dim=96,
                               kernel_initializer=keras.initializers.HeUniform(), normalize_first=True)(x)
        x = LayerNormalization()(x)
        x = TransformerDecoder(num_heads=4, intermediate_dim=96,
                               kernel_initializer=keras.initializers.HeUniform(), normalize_first=True)(x)
        x = LayerNormalization()(x)
        x = TransformerEncoder(num_heads=4, intermediate_dim=96,
                               kernel_initializer=keras.initializers.HeUniform(), normalize_first=True)(x)
        x = LayerNormalization()(x)
        x = TransformerDecoder(num_heads=4, intermediate_dim=96,
                               kernel_initializer=keras.initializers.HeUniform(), normalize_first=True)(x)
        x = FeedForwardNetworkLayer(d_model=96, d_ff=2048, output_dim=7)(x)
        model = Model(inputs=[inputs], outputs=[x])
        return model
    
