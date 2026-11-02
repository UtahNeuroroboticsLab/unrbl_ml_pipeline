"""Boundary-aware reference objectives and training-only calibration."""

import numpy as np
import tensorflow as tf


def mse(pred, target, mask):
    return tf.reduce_mean(tf.square(tf.boolean_mask(pred - target, mask)))


def temporal_terms(pred, target, edges):
    """D1/D2 SmoothL1 (beta .01), predicted squared D3; no time scaling."""
    values = []
    p, y, valid = pred, target, edges
    for order in (1, 2, 3):
        p, y = p[:, 1:] - p[:, :-1], y[:, 1:] - y[:, :-1]
        if order > 1:
            valid = valid[:, 1:] & valid[:, :-1]
        error = tf.boolean_mask(p if order == 3 else p - y, valid)
        if order == 3:
            penalty = error**2
        else:
            a = tf.abs(error)
            penalty = tf.where(a < 0.01, 0.5 * error**2 / 0.01, a - 0.005)
        values.append(
            tf.math.divide_no_nan(
                tf.reduce_sum(penalty), tf.cast(tf.size(penalty), p.dtype)
            )
        )
    return tf.stack(values)


def jacobian_probe(
    model, windows, state, mask, scale, seed, position=0, training=False
):
    """Hutchinson input-Jacobian norm; state/history are held fixed."""
    if model.state_mode == "carry":
        rows = tf.where(mask[:, position])[:, 0][:8]
        s = tuple(tf.stop_gradient(tf.gather(v, rows)) for v in state)
        if position > 0:
            _, s = model.step(
                tf.gather(windows[:, :position, -1], rows), s, training=training
            )
            s = tuple(tf.stop_gradient(v) for v in s)
        z = tf.gather(windows[:, position, -1:], rows)
        with tf.GradientTape() as inner:
            inner.watch(z)
            p = model.step(z, s, training=training)[0][:, -1]
            signs = tf.cast(
                tf.random.stateless_uniform(tf.shape(p), seed, 0, 2, dtype=tf.int32) * 2
                - 1,
                p.dtype,
            )
            scalar = tf.reduce_sum(p * signs)
        gradient = inner.gradient(scalar, z)
    else:
        z = tf.boolean_mask(windows, mask)[:8]
        with tf.GradientTape() as inner:
            inner.watch(z)
            p = model.predict_windows(z, training=training)
            signs = tf.cast(
                tf.random.stateless_uniform(tf.shape(p), seed, 0, 2, dtype=tf.int32) * 2
                - 1,
                p.dtype,
            )
            scalar = tf.reduce_sum(p * signs)
        gradient = inner.gradient(scalar, z)
    if gradient is None:
        raise ValueError("Jacobian objective requires differentiable model inputs")
    return tf.reduce_mean(
        tf.reduce_sum((gradient * scale) ** 2, axis=(1, 2))
    ) / tf.cast(model.num_outputs, p.dtype)


def predict_chunk(model, x, state, training=False):
    if model.state_mode == "carry":
        return model.step(x[:, :, -1], state, training=training)
    shape = tf.shape(x)
    p = model.predict_windows(
        tf.reshape(x, (-1, model.window_size, model.raw_features)), training=training
    )
    return tf.reshape(p, (shape[0], shape[1], model.num_outputs)), state


def grad_norm(grads):
    return sum(
        float(tf.reduce_sum(tf.cast(g, tf.float64) ** 2))
        for g in grads
        if g is not None
    )


def calibrate(model, batches, scale, cfg, use_temporal, use_jacobian):
    reference, terms, jref, jnorm = 0.0, np.zeros(3), 0.0, 0.0
    state, records = None, []
    probe_rng = np.random.default_rng(73001)
    for n, batch in enumerate(batches):
        if n >= 4:
            break
        x, y, mask, edges, _, reset = batch
        x, y, mask, edges = (tf.convert_to_tensor(v) for v in (x, y, mask, edges))
        if reset:
            state = model.prefill(x[:, 0, :-1]) if model.state_mode == "carry" else ()
        before = tuple(tf.stop_gradient(v) for v in state)
        p, state = predict_chunk(model, x, before)
        if use_temporal:
            detached = tf.cast(tf.stop_gradient(p), tf.float64)
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(detached)
                base = mse(detached, tf.cast(y, tf.float64), mask)
                penalties = tf.unstack(
                    temporal_terms(detached, tf.cast(y, tf.float64), edges)
                )
            reference += grad_norm([tape.gradient(base, detached)])
            terms += [grad_norm([tape.gradient(v, detached)]) for v in penalties]
            del tape
        probe_position, probe_seed = 0, [73001, n]
        if use_jacobian:
            valid_positions = np.flatnonzero(mask.numpy().any(0))
            probe_position = int(probe_rng.choice(valid_positions))
            probe_seed = probe_rng.integers(0, 2**30, size=2).tolist()
            with tf.GradientTape(persistent=True) as tape:
                out, _ = predict_chunk(model, x, before)
                base = mse(out, y, mask)
                j = jacobian_probe(
                    model,
                    x,
                    before,
                    mask,
                    scale,
                    tf.constant(probe_seed),
                    position=probe_position,
                )
            jref += grad_norm(tape.gradient(base, model.trainable_variables))
            jnorm += grad_norm(tape.gradient(j, model.trainable_variables))
            del tape
        records.append(
            {
                "batch": n,
                "supervised": int(tf.reduce_sum(tf.cast(mask, tf.int32))),
                "jacobian_position": probe_position,
                "jacobian_seed": probe_seed,
            }
        )
    if (use_temporal and (reference <= 0 or np.any(terms <= 1e-24))) or (
        use_jacobian and (jref <= 0 or jnorm <= 1e-24)
    ):
        raise ValueError(
            "Cannot calibrate requested penalties: zero/degenerate training gradient norms"
        )
    result = {
        "train_only": True,
        "probes": records,
        "warmup_epochs": int(cfg.get("warmup_epochs", 20)),
        "temporal_reference_norm2": reference,
        "temporal_norms2": terms.tolist(),
        "jacobian_reference_norm2": jref,
        "jacobian_norm2": jnorm,
        "temporal_weights": (
            np.minimum(
                1.0, float(cfg.get("temporal_ratio", 0.1)) * np.sqrt(reference / terms)
            )
            if use_temporal
            else np.zeros(3)
        ).tolist(),
        "jacobian_weight": min(
            100.0, float(cfg.get("jacobian_ratio", 0.03)) * np.sqrt(jref / jnorm)
        )
        if use_jacobian
        else 0.0,
        "feature_scale": scale.numpy().tolist(),
        "jacobian_scope": "newest observation conditional on detached state"
        if model.state_mode == "carry"
        else "full observed window",
    }
    return result
