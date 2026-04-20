from __future__ import annotations


def iter_prediction_items(predictions):
    """Yield per-sample prediction items from anomalib batch outputs."""
    for batch in predictions or []:
        if isinstance(batch, (list, tuple)):
            for item in batch:
                yield item
            continue
        batch_size = _infer_batch_size(batch)
        if batch_size <= 1:
            yield batch
            continue
        for index in range(batch_size):
            yield _slice_prediction_item(batch, index)


def get_prediction_field(item, key, default=None):
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def to_numpy(value):
    if value is None:
        return None
    import numpy as np

    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    if not isinstance(value, np.ndarray):
        return np.asarray(value)
    return value


def _infer_batch_size(item) -> int:
    for key in ("image_path", "pred_score", "anomaly_map", "gt_label", "label", "gt_mask", "mask"):
        value = get_prediction_field(item, key)
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            return len(value)
        shape = getattr(value, "shape", None)
        if shape and len(shape) > 0:
            return int(shape[0])
    return 1


def _slice_prediction_item(item, index: int):
    if isinstance(item, dict):
        return {key: _slice_value(value, index) for key, value in item.items()}

    class PredictionView:
        pass

    view = PredictionView()
    for key, value in vars(item).items():
        setattr(view, key, _slice_value(value, index))
    return view


def _slice_value(value, index: int):
    if isinstance(value, (list, tuple)):
        return value[index]
    if hasattr(value, "shape") and getattr(value, "ndim", 0) > 0:
        return value[index]
    return value
