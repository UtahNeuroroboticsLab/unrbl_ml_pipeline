"""TensorFlow port of decoder-design official_v2 Mamba-1 equations.

Mamba equations: Tri Dao and Albert Gu, Apache-2.0 (see licenses/MAMBA_LICENSE).
TCN topology: locuslab/TCN, MIT (see licenses/TCN_LICENSE).
No fused kernels or PyTorch are required. Runtime state is explicit, never a weight.
"""

import math

import keras
import numpy as np
import tensorflow as tf
from keras import layers

register = keras.saving.register_keras_serializable(package="unrbl")


def uniform(fan_in):
    return keras.initializers.RandomUniform(
        -1 / math.sqrt(fan_in), 1 / math.sqrt(fan_in)
    )


@register
class ReferenceLayerNorm(layers.Layer):
    """Centered variance without stopped mean gradients, including double backward."""

    def __init__(self, epsilon=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, shape):
        self.gamma = self.add_weight(
            name="gamma", shape=(int(shape[-1]),), initializer="ones"
        )
        self.beta = self.add_weight(
            name="beta", shape=(int(shape[-1]),), initializer="zeros"
        )
        super().build(shape)

    def call(self, x):
        centered = x - tf.reduce_mean(x, axis=-1, keepdims=True)
        variance = tf.reduce_mean(centered * centered, axis=-1, keepdims=True)
        return (
            centered * tf.math.rsqrt(variance + self.epsilon) * self.gamma + self.beta
        )

    def compute_output_shape(self, shape):
        return tuple(shape)

    def get_config(self):
        return dict(super().get_config(), epsilon=self.epsilon)


@register
class ReferenceAdamW(keras.optimizers.AdamW):
    """AdamW with PyTorch's corrected-moment epsilon placement."""

    def __init__(self, *args, epsilon=1e-8, **kwargs):
        super().__init__(*args, epsilon=epsilon, **kwargs)

    def update_step(self, gradient, variable, learning_rate):
        dtype = variable.dtype
        g = tf.cast(gradient, dtype)
        lr = tf.cast(learning_rate, dtype)
        step = tf.cast(self.iterations + 1, dtype)
        b1, b2 = tf.cast(self.beta_1, dtype), tf.cast(self.beta_2, dtype)
        index = self._get_variable_index(variable)
        m, v = self._momentums[index], self._velocities[index]
        self.assign_add(m, (g - m) * (1 - b1))
        self.assign_add(v, (g * g - v) * (1 - b2))
        variance = v
        if self.amsgrad:
            vmax = self._velocity_hats[index]
            self.assign(vmax, tf.maximum(vmax, v))
            variance = vmax
        self.assign_sub(
            variable,
            lr
            * (m / (1 - b1**step))
            / (tf.sqrt(variance / (1 - b2**step)) + self.epsilon),
        )


@register
class ReferenceLSTMCell(layers.Layer):
    """Native LSTM with independently optimized PyTorch input/recurrent biases."""

    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = int(units)
        self.state_size = (self.units, self.units)
        self.output_size = self.units

    def build(self, shape):
        self.kernel = self.add_weight(
            name="kernel",
            shape=(int(shape[-1]), 4 * self.units),
            initializer=uniform(self.units),
        )
        self.recurrent_kernel = self.add_weight(
            name="recurrent_kernel",
            shape=(self.units, 4 * self.units),
            initializer=uniform(self.units),
        )
        self.bias = self.add_weight(
            name="bias", shape=(4 * self.units,), initializer=uniform(self.units)
        )
        self.recurrent_bias = self.add_weight(
            name="recurrent_bias",
            shape=(4 * self.units,),
            initializer=uniform(self.units),
        )
        super().build(shape)

    def call(self, inputs, states, training=False):
        h, c = states
        z = (
            tf.matmul(inputs, self.kernel)
            + self.bias
            + tf.matmul(h, self.recurrent_kernel)
            + self.recurrent_bias
        )
        i, f, g, o = tf.split(z, 4, axis=-1)
        c = tf.sigmoid(f) * c + tf.sigmoid(i) * tf.tanh(g)
        h = tf.sigmoid(o) * tf.tanh(c)
        return h, (h, c)

    def get_config(self):
        return dict(super().get_config(), units=self.units)


@register
class WeightNormConv(layers.Layer):
    def __init__(self, width, dilation=1, **kwargs):
        super().__init__(**kwargs)
        self.width, self.dilation = width, dilation

    def build(self, shape):
        self.v = self.add_weight(
            name="v",
            shape=(3, int(shape[-1]), self.width),
            initializer=uniform(3 * int(shape[-1])),
        )
        self.g = self.add_weight(name="g", shape=(self.width,), initializer="ones")
        self.g.assign(tf.sqrt(tf.reduce_sum(self.v**2, axis=(0, 1))))
        self.bias = self.add_weight(
            name="bias", shape=(self.width,), initializer=uniform(3 * int(shape[-1]))
        )
        super().build(shape)

    def call(self, x):
        w = (
            self.v
            * self.g
            / tf.sqrt(tf.reduce_sum(self.v**2, axis=(0, 1), keepdims=True))
        )
        return (
            tf.nn.conv1d(
                tf.pad(x, [[0, 0], [2 * self.dilation, 0], [0, 0]]),
                w,
                stride=1,
                padding="VALID",
                dilations=self.dilation,
            )
            + self.bias
        )

    def compute_output_shape(self, shape):
        return tuple(shape[:-1]) + (self.width,)

    def get_config(self):
        return dict(super().get_config(), width=self.width, dilation=self.dilation)


@register
class MambaBlock(layers.Layer):
    def __init__(self, width=64, state_size=16, conv_kernel=4, expand=2, **kwargs):
        super().__init__(**kwargs)
        self.width, self.state_size, self.conv_kernel, self.expand = (
            width,
            state_size,
            conv_kernel,
            expand,
        )
        self.inner, self.rank = width * expand, math.ceil(width / 16)
        self.norm = ReferenceLayerNorm(epsilon=1e-5)
        self.final_norm = ReferenceLayerNorm(epsilon=1e-5)
        self.in_proj = layers.Dense(
            2 * self.inner, use_bias=False, kernel_initializer=uniform(width)
        )
        self.x_proj = layers.Dense(
            self.rank + 2 * state_size,
            use_bias=False,
            kernel_initializer=uniform(self.inner),
        )
        self.dt_proj = layers.Dense(self.inner, kernel_initializer=uniform(self.rank))
        self.out_proj = layers.Dense(
            width, use_bias=False, kernel_initializer=uniform(self.inner)
        )

    def build(self, shape):
        self.conv_w = self.add_weight(
            name="conv_w",
            shape=(self.conv_kernel, self.inner),
            initializer=uniform(self.conv_kernel),
        )
        self.conv_b = self.add_weight(
            name="conv_b", shape=(self.inner,), initializer=uniform(self.conv_kernel)
        )
        self.A_log = self.add_weight(
            name="A_log",
            shape=(self.inner, self.state_size),
            initializer=keras.initializers.Constant(
                np.tile(np.log(np.arange(1, self.state_size + 1)), (self.inner, 1))
            ),
        )
        self.D = self.add_weight(name="D", shape=(self.inner,), initializer="ones")
        self.norm.build(shape)
        self.final_norm.build(shape)
        self.in_proj.build(shape)
        self.x_proj.build(tuple(shape[:-1]) + (self.inner,))
        self.dt_proj.build(tuple(shape[:-1]) + (self.rank,))
        dt = tf.exp(tf.random.uniform((self.inner,), math.log(0.001), math.log(0.1)))
        self.dt_proj.bias.assign(dt + tf.math.log(-tf.math.expm1(-dt)))
        self.out_proj.build(tuple(shape[:-1]) + (self.inner,))
        super().build(shape)

    def initial(self, batch):
        return (
            tf.zeros((batch, self.conv_kernel - 1, self.inner)),
            tf.zeros((batch, self.inner, self.state_size)),
        )

    def call(self, x, state=None):
        if state is None:
            state = self.initial(tf.shape(x)[0])
        u, gate = tf.split(self.in_proj(self.norm(x)), 2, axis=-1)
        history = tf.concat((state[0], u), axis=1)
        framed = tf.signal.frame(history, self.conv_kernel, 1, axis=1)
        v = tf.nn.silu(tf.reduce_sum(framed * self.conv_w, axis=2) + self.conv_b)
        dt, b, c = tf.split(
            self.x_proj(v), [self.rank, self.state_size, self.state_size], axis=-1
        )
        dt = tf.nn.softplus(self.dt_proj(dt))
        a = -tf.exp(self.A_log)
        elems = tuple(tf.transpose(t, (1, 0, 2)) for t in (v, dt, b, c, gate))

        def advance(acc, elem):
            value, delta, bb, cc, z = elem
            s = (
                tf.exp(delta[:, :, None] * a) * acc[0]
                + (delta * value)[:, :, None] * bb[:, None, :]
            )
            y = (
                tf.reduce_sum(s * cc[:, None, :], axis=-1) + self.D * value
            ) * tf.nn.silu(z)
            return s, y

        states, output = tf.scan(
            advance,
            elems,
            initializer=(state[1], tf.zeros((tf.shape(x)[0], self.inner))),
        )
        y = self.final_norm(x + self.out_proj(tf.transpose(output, (1, 0, 2))))
        return y, (
            history[:, -(self.conv_kernel - 1) :]
            if self.conv_kernel > 1
            else history[:, :0],
            states[-1],
        )

    def compute_output_shape(self, shape):
        # Match call(): sequence features, then convolution and SSM state.
        batch, length, _ = shape
        return (batch, length, self.width), (
            (batch, self.conv_kernel - 1, self.inner),
            (batch, self.inner, self.state_size),
        )

    def get_config(self):
        return dict(
            super().get_config(),
            width=self.width,
            state_size=self.state_size,
            conv_kernel=self.conv_kernel,
            expand=self.expand,
        )


@register
class MambaDecoder(keras.Model):
    def __init__(
        self,
        kind,
        raw_features,
        num_outputs,
        window_size=10,
        architecture=None,
        state_mode="carry",
        selected_indices=None,
        norm_stats=None,
        seed=0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.kind, self.raw_features, self.num_outputs = (
            kind,
            int(raw_features),
            int(num_outputs),
        )
        self.window_size, self.state_mode, self.seed = (
            int(window_size),
            state_mode,
            int(seed),
        )
        if state_mode not in ("carry", "reset") or self.window_size < 1:
            raise ValueError("state_mode must be carry/reset and window_size positive")
        self.architecture = dict(architecture or {})
        a = self.architecture
        self.width, self.twidth = (
            int(a.get("mamba_width", 64)),
            int(a.get("tcn_width", 96)),
        )
        self.indices = (
            list(range(raw_features))
            if selected_indices is None
            else list(selected_indices)
        )
        if (
            not self.indices
            or len(set(self.indices)) != len(self.indices)
            or min(self.indices) < 0
            or max(self.indices) >= raw_features
        ):
            raise ValueError("selected_indices must be unique raw feature indices")
        self.norm_stats = norm_stats
        self.drop_rate = float(a.get("dropout", 0.2))
        if not 0 <= self.drop_rate < 1:
            raise ValueError("dropout must be in [0,1)")
        self.rng = keras.random.SeedGenerator(seed)
        mw_input = self.twidth if kind == "tcn_mamba" else len(self.indices)
        self.project = layers.Dense(
            self.width,
            kernel_initializer=uniform(mw_input),
            bias_initializer=uniform(mw_input),
        )
        self.mamba = MambaBlock(
            self.width,
            int(a.get("state_size", 16)),
            int(a.get("conv_kernel", 4)),
            int(a.get("expand", 2)),
        )
        self.has_tcn = kind in ("tcn_mamba", "mamba_tcn", "mamba_tcl")
        if self.has_tcn:
            cin = len(self.indices) if kind == "tcn_mamba" else self.width
            self.convs = [WeightNormConv(self.twidth, d) for d in (1, 1, 2, 2)]
            self.skip = (
                layers.Dense(
                    self.twidth,
                    kernel_initializer=keras.initializers.RandomNormal(stddev=0.01),
                    bias_initializer=uniform(cin),
                )
                if cin != self.twidth
                else None
            )
        self.has_lstm = kind in ("mamba_tcl", "mamba_lstm")
        feature_dim = self.width if kind == "mamba" else self.twidth + self.width
        if self.has_lstm:
            feature_dim = int(a.get("lstm_units", 96))
            self.recurrent_norm = ReferenceLayerNorm(epsilon=1e-5)
            # Native operations permit nested differentiation, including on GPU.
            self.lstm = layers.RNN(ReferenceLSTMCell(feature_dim), unroll=True)

        self.feature_norm = ReferenceLayerNorm(epsilon=1e-5)
        self.fusion_norm = ReferenceLayerNorm(epsilon=1e-5)
        head = int(a.get("head_units", 192))
        self.fc = layers.Dense(
            head,
            activation="relu",
            kernel_initializer=uniform(feature_dim),
            bias_initializer=uniform(feature_dim),
        )
        self.head = layers.Dense(
            num_outputs,
            kernel_initializer=uniform(head),
            bias_initializer=uniform(head),
        )

    def build(self, input_shape):
        if self.built:
            return
        self.predict_windows(tf.zeros((1, self.window_size, self.raw_features)))
        super().build(input_shape)

    def _drop(self, x, training):
        return (
            keras.random.dropout(x, self.drop_rate, seed=self.rng)
            if training and self.drop_rate
            else x
        )

    def _input(self, x):
        x = tf.cast(x, tf.float32)
        if self.norm_stats:
            x = (x - tf.constant(self.norm_stats["feat_mean"])) / tf.constant(
                self.norm_stats["feat_std"]
            )
        return tf.gather(x, self.indices, axis=-1)

    def _tcn(self, x, training):
        for i in range(2):
            residual = self.skip(x) if i == 0 and self.skip is not None else x
            x = self._drop(tf.nn.relu(self.convs[2 * i](x)), training)
            x = self._drop(tf.nn.relu(self.convs[2 * i + 1](x)), training)
            x = tf.nn.relu(x + residual)
        return x

    def _head(self, x, training):
        y = self.head(
            self._drop(self.fc(self.fusion_norm(self.feature_norm(x))), training)
        )
        if self.norm_stats:
            y = y * tf.constant(self.norm_stats["kin_std"]) + tf.constant(
                self.norm_stats["kin_mean"]
            )
        return y

    def predict_windows(self, features, training=False):
        """Explicit independent-window evaluation, even for carry-trained weights."""
        x = self._input(features)
        local = self._tcn(x, training) if self.kind == "tcn_mamba" else x
        h, _ = self.mamba(self.project(local))
        if self.kind == "mamba":
            f = h[:, -1]
        elif self.kind == "tcn_mamba":
            f = tf.concat((local[:, -1], h[:, -1]), -1)
        else:
            local = self._tcn(h, training) if self.has_tcn else h
            f = (
                self.lstm(self.recurrent_norm(local), training=training)
                if self.has_lstm
                else tf.concat((local[:, -1], h[:, -1]), -1)
            )
        return self._head(f, training)

    def initial_state(self, batch_size):
        conv, ssm = self.mamba.initial(batch_size)
        width = len(self.indices) if self.kind == "tcn_mamba" else self.width
        return (conv, ssm, tf.zeros((batch_size, self.window_size - 1, width)))

    def prefill(self, features, training=False):
        """Initialize from exactly W-1 raw observations, oldest first."""
        tf.debugging.assert_equal(tf.shape(features)[1], self.window_size - 1)
        if self.window_size == 1:
            return self.initial_state(tf.shape(features)[0])
        x = self._input(features)
        local = self._tcn(x, training) if self.kind == "tcn_mamba" else x
        h, (conv, ssm) = self.mamba(self.project(local))
        return conv, ssm, x if self.kind == "tcn_mamba" else h

    def step(self, features, state, training=False):
        """Consume [batch, new_samples, raw_features]; return predictions and state."""
        x = self._input(features)
        b, t = tf.shape(x)[0], tf.shape(x)[1]
        if self.kind == "tcn_mamba":
            raw = tf.concat((state[2], x), 1)
            windows = tf.signal.frame(raw, self.window_size, 1, axis=1)
            local = self._tcn(
                tf.reshape(windows, (-1, self.window_size, len(self.indices))), training
            )[:, -1]
            local = tf.reshape(local, (b, t, self.twidth))
            h, (conv, ssm) = self.mamba(self.project(local), state=state[:2])
            f, history = tf.concat((local, h), -1), raw
        else:
            h, (conv, ssm) = self.mamba(self.project(x), state=state[:2])
            history = tf.concat((state[2], h), 1)
            if self.kind == "mamba":
                f = h
            else:
                windows = tf.signal.frame(history, self.window_size, 1, axis=1)
                local = tf.reshape(windows, (-1, self.window_size, self.width))
                local = self._tcn(local, training) if self.has_tcn else local
                if self.has_lstm:
                    local = self.lstm(self.recurrent_norm(local), training=training)
                    f = tf.reshape(local, (b, t, self.lstm.cell.units))
                else:
                    f = tf.concat(
                        (tf.reshape(local[:, -1], (b, t, self.twidth)), h), -1
                    )
        history = (
            history[:, -(self.window_size - 1) :]
            if self.window_size > 1
            else history[:, :0]
        )
        return self._head(f, training), (conv, ssm, history)

    def call(self, inputs, training=False):
        if self.state_mode == "reset":
            return self.predict_windows(inputs, training=training)
        if not isinstance(inputs, dict) or "state" not in inputs:
            raise ValueError(
                "Carry mode requires features and explicit state; use stream() or predict_windows()"
            )
        return self.step(inputs["features"], inputs["state"], training=training)

    def stream(self):
        return StreamingDecoder(self)

    def get_config(self):
        return dict(
            super().get_config(),
            kind=self.kind,
            raw_features=self.raw_features,
            num_outputs=self.num_outputs,
            window_size=self.window_size,
            architecture=self.architecture,
            state_mode=self.state_mode,
            selected_indices=self.indices,
            norm_stats=self.norm_stats,
            seed=self.seed,
        )


class StreamingDecoder:
    """One runtime instance per independent stream; reset explicitly after hidden gaps."""

    def __init__(self, model):
        self.model = model
        self.reset()

    def reset(self):
        self.state, self.previous = None, None

    def predict(self, window, reset=False):
        x = np.asarray(window, dtype=np.float32)
        if (
            x.shape != (self.model.window_size, self.model.raw_features)
            or not np.isfinite(x).all()
        ):
            raise ValueError(
                "Expected a finite chronological [window_size, raw_features] window"
            )
        if self.model.state_mode == "reset":
            return self.model.predict_windows(x[None]).numpy()[0]
        if (
            reset
            or self.previous is None
            or not np.array_equal(self.previous[1:], x[:-1])
        ):
            self.state = self.model.prefill(x[None, :-1])
        y, self.state = self.model.step(x[None, -1:], self.state)
        self.previous = x.copy()
        return y.numpy()[0, 0]
