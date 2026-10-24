"""Causal full-prefix or bounded video inputs (no model inference here)."""
import numpy as np


def select_video_frames(raw_wall, raw_endpoint, fps, history_seconds):
    """Sample a bounded buffer, or the full prefix when history_seconds=None.

    Select the last capture at/before each requested time; de-duplicate if
    acquisition is slower than the requested FPS. Always retain the endpoint.
    Full history uses a fixed grid starting at recording time zero and retains
    the initial image. A bounded buffer keeps V4's endpoint-anchored grid.
    """
    times = np.asarray(raw_wall, dtype=np.float64)
    if fps <= 0 or (history_seconds is not None and history_seconds <= 0) or not 0 <= raw_endpoint < len(times):
        raise ValueError('Invalid video sampling parameters')
    # Never examine future timestamps to select input frames.
    times = times[:raw_endpoint + 1] - times[0]
    if not np.isfinite(times).all() or np.any(np.diff(times) < 0):
        raise ValueError('Recorded wall timestamps must be finite and monotone')
    end = times[-1]
    lower = 0. if history_seconds is None else max(0., end - history_seconds)
    count = int(np.floor((end - lower) * fps + 1e-8))
    targets = (np.arange(count + 1) / fps if history_seconds is None
               else end - np.arange(count, -1, -1) / fps)
    ids = np.searchsorted(times, targets, side='right') - 1
    ids = ids[(ids >= 0) & (times[np.maximum(ids, 0)] >= lower - 1e-8)]
    ids = np.unique(np.append(ids, raw_endpoint)).astype(int)
    return ids.tolist(), times[ids]


def timestamp_metadata(seconds):
    """Express actual timestamps as integer microsecond ticks for HF's API.

    The native processor accepts indices/fps, not arbitrary timestamps. These
    are a time base, NOT raw frame indices or camera FPS. Frame re-sampling is
    disabled. Native temporal-patch timestamps are pair means, as documented
    in the pinned processor. Padding repeats only the final observed frame.
    """
    ticks = np.rint(np.asarray(seconds) * 1_000_000).astype(np.int64).tolist()
    return {'fps': 1_000_000., 'frames_indices': ticks,
            'total_num_frames': ticks[-1] + 1}


def prepare_video_inputs(processor, messages, arrays, seconds, preprocessing):
    import torch
    from PIL import Image
    from qwen_vl_utils.vision_process import fetch_image
    images = [fetch_image({'image': Image.fromarray(array),
                          'min_pixels': preprocessing['min_pixels_per_image'],
                          'max_pixels': preprocessing['max_pixels_per_image']},
                         image_patch_size=16) for array in arrays]
    sizes = [list(image.size) for image in images]
    if len(set(map(tuple, sizes))) != 1:
        raise ValueError('Video resolution changes inside a request')
    if any(w * h > preprocessing['max_pixels_per_image'] for w, h in sizes):
        raise ValueError('Video pixel cap exceeded')
    video = torch.from_numpy(np.stack([np.asarray(image) for image in images])).permute(0, 3, 1, 2)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    metadata = timestamp_metadata(seconds)
    inputs = processor(text=[text], videos=[video], video_metadata=[metadata],
                       do_sample_frames=False, do_resize=False, return_tensors='pt')
    grids = inputs.video_grid_thw.tolist()
    pair_times = processor._calculate_timestamps(list(metadata['frames_indices']), metadata['fps'],
                                                 processor.video_processor.merge_size)
    expanded = processor.tokenizer.decode(inputs.input_ids[0], skip_special_tokens=False)
    for value in pair_times:
        if f'<{value:.1f} seconds>' not in expanded:
            raise ValueError('Native processor omitted a recorded timestamp')
    details = {'chat_template_text': text, 'processed_sizes_wh': sizes,
               'video_grid_thw': grids, 'video_temporal_patch_timestamps_s': pair_times,
               'timestamp_timebase_hz': metadata['fps'],
               'timestamp_ticks': list(timestamp_metadata(seconds)['frames_indices']),
               'video_padding_repeated_last_frames': (2 - len(images) % 2) % 2,
               'visual_tokens': int(sum(np.prod(g) // processor.video_processor.merge_size**2 for g in grids))}
    return inputs, details
