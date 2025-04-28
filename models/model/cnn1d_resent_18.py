import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

class BasicBlock1D(tf.keras.Model):
    def __init__(self, filters, stride=1, downsample=False):
        super(BasicBlock1D, self).__init__()
        self.conv1 = layers.Conv1D(filters, kernel_size=7, strides=stride, padding="same", use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.ReLU()
        self.dropout = layers.Dropout(0.2)
        self.conv2 = layers.Conv1D(filters, kernel_size=7, strides=1, padding="same", use_bias=False)
        self.bn2 = layers.BatchNormalization()
        self.downsample = downsample
        if downsample:
            self.downsample_layer = models.Sequential([
                layers.Conv1D(filters, kernel_size=1, strides=stride, use_bias=False),
                layers.BatchNormalization()
            ])

    def call(self, inputs, training=False):
        residual = inputs
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu(x)
        x = self.dropout(x, training=training)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        if self.downsample:
            residual = self.downsample_layer(inputs, training=training)
        x += residual
        return self.relu(x)

class ResNet1DModel:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self.build_model()

    def build_model(self):
        inputs = layers.Input(shape=self.input_shape)
        x = layers.Conv1D(64, kernel_size=15, strides=2, padding="same", use_bias=False)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling1D(pool_size=3, strides=2, padding="same")(x)

        # Define ResNet18-like block structure
        x = self._make_layer(x, filters=64, blocks=2, stride=1)
        x = self._make_layer(x, filters=128, blocks=2, stride=2)
        x = self._make_layer(x, filters=256, blocks=2, stride=2)
        x = self._make_layer(x, filters=512, blocks=2, stride=2)

        avg_pool = layers.GlobalAveragePooling1D()(x)
        max_pool = layers.GlobalMaxPooling1D()(x)
        x = layers.Concatenate()([avg_pool, max_pool])
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)

        return models.Model(inputs, outputs)

    def _make_layer(self, x, filters, blocks, stride):
        x = BasicBlock1D(filters, stride=stride, downsample=True)(x)
        for _ in range(1, blocks):
            x = BasicBlock1D(filters)(x)
        return x

    def get_model(self):
        return self.model
