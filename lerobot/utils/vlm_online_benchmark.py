"""Reasoning-capable VLM adapter for the existing latest-window replay scheduler.

prepare: CPU-only immutable plan. smoke: calibration-only development requests.
run: all test episodes, gated by a matching successful smoke report. No retries,
logit masks, output grammars, cross-request caches, or automatic OOM fallbacks.
"""
import argparse
import csv
import fcntl
import gc
import hashlib
import html
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import yaml

from online_benchmark import input_window, preload_episode, replay_episode
from unified_benchmark import annotation_views, episode_labels, load_labels
from vlm_answers import FinalAnswerStop, classify_completion
from vlm_video import prepare_video_inputs, select_video_frames

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO / 'lerobot/configs/benchmark/vlm/protocol.yaml'


def is_video(config):
    return config['inputs']['input_modality'] in {'bounded_history_video', 'per_model_history_video'}


def video_history_seconds(config, model_name):
    if config['inputs']['input_modality'] == 'per_model_history_video':
        return config['models'][model_name]['history_seconds']
    return config['inputs']['history_seconds']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def validate_protocol(config):
    """Reject unsupported options rather than silently ignoring a config edit."""
    required = {
        ('schema_version',): 2,
        ('inputs', 'K'): 4, ('inputs', 'L'): 4,
        ('inputs', 'views'): ['fix', 'wrist'],
        ('inference', 'dtype'): 'bfloat16', ('inference', 'attention'): 'sdpa',
        ('inference', 'window_batch_size'): 1, ('inference', 'concurrent_requests'): 1,
        ('inference', 'cpu_offload'): False, ('inference', 'quantization'): None,
        ('inference', 'tf32'): False, ('inference', 'matmul_precision'): 'highest',
        ('generation', 'do_sample'): False, ('generation', 'num_beams'): 1,
        ('generation', 'min_new_tokens'): 0, ('generation', 'label_constraint'): None,
        ('generation', 'use_cache'): True, ('generation', 'repetition_penalty'): 1.0,
        ('generation', 'temperature'): None, ('generation', 'top_p'): None,
        ('generation', 'top_k'): None,
        ('preprocessing', 'image_patch_size'): 16,
        ('preprocessing', 'processor_do_resize'): False,
        ('preprocessing', 'preserve_aspect_ratio'): True,
        ('preprocessing', 'max_pixels_per_image'): 262144,
        ('streaming', 'scheduling'): 'latest_available_eligible_endpoint',
        ('streaming', 'replay_speed'): 1.0,
        ('evaluation', 'selection'): 'all_test_episodes',
        ('development', 'split'): 'calibration',
    }
    video_mode = is_video(config)
    if video_mode:
        required[('inputs', 'first_eligible_endpoint')] = 7
        if config['inputs']['input_modality'] == 'bounded_history_video':
            required.update({('inputs', 'history_seconds'): 20.,
                             ('inputs', 'sampling'): 'endpoint_anchored_last_capture'})
        else:
            required[('inputs', 'sampling')] = 'full_history_zero_anchored_or_bounded_endpoint_anchored'
    else:
        required[('inputs', 'input_modality')] = 'five_independent_rgb_images'
        required[('inputs', 'frame_offsets_from_endpoint')] = [-7, -6, -5, -4, 0]
    for keys, expected in required.items():
        value = config
        for key in keys:
            value = value[key]
        if value != expected:
            raise ValueError(f'Unsupported protocol option {".".join(keys)}={value!r}')
    timeout = config['generation']['decision_timeout_s']
    if config.get('evaluation_protocol'):
        from benchmark_frozen_protocol import load_protocol
        spec = load_protocol(config['evaluation_protocol'])
        if config['streaming']['stop_at_first_FAIL'] is not False:
            raise ValueError('Frozen onset protocol requires full selected VLM traces')
        if config['evaluation']['primary_annotator_id'] != spec['detection']['primary_annotator']:
            raise ValueError('VLM primary annotator differs from frozen protocol')
    if timeout is not None and timeout <= 0:
        raise ValueError('Decision timeout must be positive or null')
    for name, model in config['models'].items():
        if type(model['max_new_tokens']) is not int or model['max_new_tokens'] <= 0:
            raise ValueError('Each model needs a positive max_new_tokens')
        if video_mode and model.get('video_fps', 0) <= 0:
            raise ValueError('Each video model needs a positive video_fps')
        if video_mode:
            limit = video_history_seconds(config, name)
            if limit is not None and (not isinstance(limit, (float, int)) or limit <= 0):
                raise ValueError('History duration must be positive or explicitly null for full history')


def make_plan(config_path):
    config = yaml.safe_load(config_path.read_text())
    validate_protocol(config)
    tasks = yaml.safe_load((REPO / config['task_definitions']).read_text())
    prompt_template = (REPO / config['prompt_file']).read_text()
    validate_prompt_inputs(config, tasks, prompt_template)
    labels, label_text = load_labels(REPO / config['labels_for_selection_and_reporting_only'], list(tasks['tasks']))
    sources = [config_path, REPO / config['task_definitions'], REPO / config['prompt_file'],
               Path(__file__), Path(__file__).with_name('vlm_answers.py'),
               Path(__file__).with_name('vlm_video.py'),
               Path(__file__).with_name('online_benchmark.py'),
               Path(__file__).with_name('unified_benchmark.py'),
               Path(__file__).with_name('model_paths.py')]
    if config.get('evaluation_protocol'):
        from benchmark_frozen_protocol import load_protocol
        from benchmark_consensus_onsets import derive_labels
        frozen = load_protocol(config['evaluation_protocol'])
        labels = derive_labels(labels, frozen)
        sources += [REPO/config['evaluation_protocol'], Path(__file__).with_name('benchmark_frozen_protocol.py')]
        sources += [Path(__file__).with_name('benchmark_consensus_onsets.py'),
                    Path(__file__).with_name('benchmark_onset_results.py')]
        sources += [REPO/entry['path'] for entry in frozen['frozen_inputs'].values()]
    files = {str(path.relative_to(REPO)): sha(path) for path in sources}
    models = {}
    for name, model in config['models'].items():
        path = REPO / model['path']
        if (path / 'HF_REVISION.txt').read_text().strip() != model['revision']:
            raise ValueError(f'Model revision differs: {name}')
        # The receipt records the previously verified Hub SHA256s. Bind file
        # sizes/mtimes as well; never re-download or replace a local checkpoint.
        models[name] = {'verified_download_receipt': json.loads((path / 'DOWNLOAD_VERIFIED.json').read_text()),
                        'files': {p.name: {'size': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns,
                                          **({'sha256': sha(p)} if p.suffix != '.safetensors' else {})}
                                  for p in sorted(path.iterdir()) if p.is_file()}}
    trials = []
    for task, node in tasks['tasks'].items():
        dataset = REPO / node['prepared_dataset']
        files[str((dataset / 'meta/preparation.json').relative_to(REPO))] = sha(dataset / 'meta/preparation.json')
        lookup = episode_labels(labels['tasks'][task])
        dev = lookup[config['development']['episode_index']]
        if dev['split'] != 'calibration' or dev['outcome'] != 'success':
            raise ValueError('Development images must belong to calibration successes')
        for extra in config['development'].get('additional_last_endpoints', {}).values():
            for episode in extra.get(task, []):
                if lookup[episode]['split'] != 'calibration' or lookup[episode]['outcome'] != 'success':
                    raise ValueError('Additional development cases must be calibration successes')
        for name in models:
            for view in config['inputs']['views']:
                counts = {'success': 0, 'failure': 0}
                for episode, group in lookup.items():
                    if group['split'] == 'test':
                        counts[group['outcome']] += 1
                        trials.append({'id': f'{name}__{task}__{view}__{episode:06d}',
                                       'model': name, 'task': task, 'view': view, 'episode': episode,
                                       'outcome': group['outcome'], 'failure_mode': group['failure_mode'],
                                       'dataset': node['prepared_dataset']})
                if counts != config['evaluation']['expected_counts_per_model_view'][task]:
                    raise ValueError(f'Test coverage differs: {task}: {counts}')
    if len(trials) != config['evaluation']['expected_replays_all_models_views_tasks']:
        raise ValueError('Unexpected number of test replays')
    env = {package: importlib.metadata.version(package) for package in
           ['torch', 'transformers', 'qwen-vl-utils', 'Pillow', 'numpy', 'pyarrow', 'PyYAML', 'accelerate']}
    identity = {'config': config, 'files': files, 'models': models, 'environment': env,
                'python': sys.version, 'labels_sha256': hashlib.sha256(label_text.encode()).hexdigest()}
    return {'fingerprint': digest(identity), 'identity': identity, 'config': config, 'tasks': tasks,
            'prompt_template': prompt_template,
            'labels': labels, 'trials': trials}


def task_time_limit_seconds(node, task):
    """An explicit task-wide allowance, never inferred from episode duration."""
    value = node.get('time_limit_seconds')
    if type(value) not in (int, float) or not np.isfinite(value) or value <= 0:
        raise ValueError(f'{task}: set time_limit_seconds to a positive finite number in the task YAML')
    return float(value)


def validate_prompt_inputs(config, tasks, template):
    if '{TIME_LIMIT}' in template:
        # This template explicitly promises a full observed prefix at 1 FPS.
        if not is_video(config) or any(
                model.get('video_fps') != 1.0 or video_history_seconds(config, name) is not None
                for name, model in config['models'].items()):
            raise ValueError('The task-deadline prompt requires full-history video at 1 FPS')
    for task, node in tasks['tasks'].items():
        if not isinstance(node.get('task_description'), str) or not node['task_description'].strip():
            raise ValueError(f'{task}: write a nonempty task_description in the task YAML')
        if '{TIME_LIMIT}' in template:
            task_time_limit_seconds(node, task)


def render_prompt(plan, task, model_name, seconds):
    node = plan['tasks']['tasks'][task]
    template = plan['prompt_template']
    decimals = plan['config']['inputs']['timestamp_display_decimals']
    values = {f't{i + 1}': f'{float(value):.{decimals}f}' for i, value in enumerate(seconds)}
    values.update(TASK_DESCRIPTION=node['task_description'],
                  t5_minus_t4=f'{float(seconds[-1] - seconds[-2]):.{decimals}f}' if len(seconds) > 1 else '0.000',
                  RESPONSE_INSTRUCTIONS=plan['config']['models'][model_name]['response_instructions'])
    if '{SUCCESS_CRITERIA}' in template:
        values['SUCCESS_CRITERIA'] = node['success_criteria']
    if '{TASK_SPECIFIC_DEFINITIONS}' in template:
        values['TASK_SPECIFIC_DEFINITIONS'] = plan['tasks']['shared_visual_rules'] + '\n' + '\n'.join(
            f'{name}: {text}' for name, text in node['failure_definitions'].items())
    if '{TIME_LIMIT}' in template:
        values['TIME_LIMIT'] = f'{task_time_limit_seconds(node, task):g}'
    if '{ELAPSED_TIME}' in template:
        if len(seconds) == 0 or not np.isfinite(seconds[-1]) or seconds[-1] < 0:
            raise ValueError('Elapsed time must be a finite, nonnegative observed wall timestamp')
        # The endpoint is captured before processing. No future recording length
        # or post-generation clock enters the prompt's assessment of this video.
        values['ELAPSED_TIME'] = f'{float(seconds[-1]):.{decimals}f}'
    if '{NORMAL_TASK_STAGES}' in template:
        # Required only by the stage-aware protocol; keep historical prompts unchanged.
        values['NORMAL_TASK_STAGES'] = node['normal_task_stages']
    if is_video(plan['config']):
        limit = video_history_seconds(plan['config'], model_name)
        values.update(frame_count=len(seconds),
                      video_fps=plan['config']['models'][model_name]['video_fps'],
                      history_seconds=limit,
                      history_description=('using all available history from recording time zero.' if limit is None else
                                           f'using at most the most recent {limit} seconds.\nAt the start of an episode, less history is available; do not expect a full buffer.'),
                      history_omission=('No earlier history is discarded.' if limit is None else
                                        'Older history outside this buffer is not provided.'),
                      start_time=f'{float(seconds[0]):.{decimals}f}',
                      end_time=f'{float(seconds[-1]):.{decimals}f}')
    return template.format(**values)


def generation_options(spec, checkpoint_config):
    from transformers import GenerationConfig
    # Do not inherit checkpoint sampling defaults or restrictive processors.
    # Transformers >=4.50 otherwise replaces fields equal to library defaults.
    generation = GenerationConfig(
        max_new_tokens=spec['max_new_tokens'], min_new_tokens=0,
        do_sample=False, num_beams=1, use_cache=True,
        temperature=None, top_p=None, top_k=None, repetition_penalty=1.0,
        eos_token_id=checkpoint_config.eos_token_id,
        pad_token_id=checkpoint_config.pad_token_id,
        return_dict_in_generate=True, output_scores=False)
    return {'generation_config': generation, 'use_model_defaults': False,
            'do_sample': False, 'num_beams': 1}


class DecisionError(RuntimeError):
    pass


class StreamClock:
    """Observe the existing replay clock without changing its scheduling."""
    def __init__(self):
        self.origin = None

    def __call__(self):
        now = time.perf_counter()
        if self.origin is None:
            self.origin = now
        return now

    def elapsed(self):
        return time.perf_counter() - self.origin if self.origin is not None else None


class VLMAdapter:
    def __init__(self, plan, model_name):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        self.plan, self.name = plan, model_name
        self.config = plan['config']
        self.spec = self.config['models'][model_name]
        self.device = self.config['inference']['device']
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is required; no CPU fallback')
        torch.set_num_threads(4)
        torch.manual_seed(self.config['inference']['seed'])
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        path = REPO / self.spec['path']
        self.processor = AutoProcessor.from_pretrained(path, local_files_only=True)
        started = time.perf_counter()
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            path, dtype=torch.bfloat16, device_map={'': self.device},
            attn_implementation='sdpa', local_files_only=True).eval()
        if any(p.device.type != 'cuda' or p.dtype != torch.bfloat16 for p in self.model.parameters()):
            raise RuntimeError('Unexpected precision or CPU offload')
        torch.cuda.synchronize(self.device)
        self.load_s = time.perf_counter() - started
        self.task = self.view = None
        self.stream_clock = None
        self.reset_episode()

    def reset_episode(self):
        self.records = []
        self.last_stage_times = {}

    def score(self, payload, window):
        import torch
        from PIL import Image
        from qwen_vl_utils import process_vision_info
        from transformers import StoppingCriteriaList
        started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats(self.device)
        video_mode = is_video(self.config)
        if video_mode:
            raw_ids, seconds = select_video_frames(payload['raw_wall'], window.raw_endpoint,
                                                   self.spec['video_fps'], video_history_seconds(self.config, self.name))
            raw_to_prepared = {int(raw): i for i, raw in enumerate(payload['raw'][:window.endpoint + 1])}
            positions = [raw_to_prepared.get(raw) for raw in raw_ids]
        else:
            positions = list(window.history) + [window.endpoint]
            raw_ids = [int(payload['raw'][i]) for i in positions]
            seconds = payload['raw_wall'][raw_ids] - payload['raw_wall'][0]
        record = {'prepared_frame_ids': positions, 'raw_frame_ids': raw_ids,
                  'frame_timestamps_s': seconds.tolist(),
                  'history_endpoint_gap_s': float(seconds[-1] - seconds[-2]) if len(seconds) > 1 else None,
                  'endpoint_source_time_s': window.source_time_s,
                  'raw_output': '', 'generated_token_ids': [], 'token_count': 0,
                  'status': 'runtime_error', 'final_label': None, 'stopping_reason': 'exception'}
        if self.stream_clock is not None:
            record['stream_started_s'] = self.stream_clock.elapsed()
        stop = None
        try:
            prompt = render_prompt(self.plan, self.task, self.name, seconds)
            record['rendered_prompt'] = prompt
            if '{TIME_LIMIT}' in self.plan['prompt_template']:
                record['prompt_time_limit_s'] = task_time_limit_seconds(self.plan['tasks']['tasks'][self.task], self.task)
                record['prompt_elapsed_time_s'] = float(seconds[-1])
            content = [{'type': 'text', 'text': prompt}]
            shapes = [list(payload['images'][self.view][raw_id].shape) for raw_id in raw_ids]
            if video_mode:
                content.append({'type': 'video'})
                record.update(input_modality=self.config['inputs']['input_modality'], target_fps=self.spec['video_fps'],
                              history_seconds=video_history_seconds(self.config, self.name),
                              original_shapes=shapes)
            for number, raw_id in enumerate(raw_ids if not video_mode else []):
                array = payload['images'][self.view][raw_id]
                content.extend([{'type': 'text', 'text': f'Frame {number + 1}: {seconds[number]:.3f} seconds'},
                                {'type': 'image', 'image': Image.fromarray(array),
                                 'min_pixels': self.config['preprocessing']['min_pixels_per_image'],
                                 'max_pixels': self.config['preprocessing']['max_pixels_per_image']}])
            messages = []
            if self.spec['system_message']:
                messages.append({'role': 'system', 'content': [{'type': 'text', 'text': self.spec['system_message']}]})
            messages.append({'role': 'user', 'content': content})
            if video_mode:
                inputs, details = prepare_video_inputs(self.processor, messages,
                    [payload['images'][self.view][i] for i in raw_ids], seconds, self.config['preprocessing'])
                record.update(details)
            else:
                text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                record['chat_template_text'] = text
                images, videos = process_vision_info(messages, image_patch_size=16)
                if videos is not None or len(images) != 5:
                    raise ValueError('Expected exactly five independent images')
                record['original_shapes'] = shapes
                record['processed_sizes_wh'] = [list(image.size) for image in images]
                if any(image.width * image.height > self.config['preprocessing']['max_pixels_per_image'] for image in images):
                    raise ValueError('Image pixel cap exceeded')
                inputs = self.processor(text=[text], images=images, do_resize=False, return_tensors='pt')
                record['image_grid_thw'] = inputs.image_grid_thw.tolist()
                merge = self.processor.image_processor.merge_size
                record['visual_tokens'] = int(sum(np.prod(grid) // merge**2 for grid in record['image_grid_thw']))
            record['input_tokens'] = int(inputs.input_ids.shape[1])
            prepared = time.perf_counter()
            inputs = inputs.to(self.device)
            torch.cuda.synchronize(self.device)
            transferred = time.perf_counter()
            eos = self.model.generation_config.eos_token_id
            eos = [eos] if isinstance(eos, int) else list(eos or [])
            options = generation_options(self.spec, self.model.generation_config)
            record['decoding'] = {'do_sample': False, 'num_beams': 1, 'use_model_defaults': False,
                                   'max_new_tokens': self.spec['max_new_tokens'], 'min_new_tokens': 0}
            stop = FinalAnswerStop(self.processor.tokenizer, inputs.input_ids.shape[1],
                                   require_think=self.spec['require_think'], started=started,
                                   timeout_s=self.config['generation']['decision_timeout_s'])
            with torch.inference_mode():
                output = self.model.generate(**inputs, **options,
                                             stopping_criteria=StoppingCriteriaList([stop]))
            torch.cuda.synchronize(self.device)
            generated = time.perf_counter()
            ids = output.sequences[0, inputs.input_ids.shape[1]:].tolist()
            decoded = self.processor.tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            record.update(raw_output=self.processor.tokenizer.decode(ids, skip_special_tokens=False,
                                                                     clean_up_tokenization_spaces=False),
                          parser_text=decoded, generated_token_ids=ids, token_count=len(ids))
            record.update(classify_completion(decoded, ids, require_think=self.spec['require_think'],
                                               max_new_tokens=self.spec['max_new_tokens'],
                                               eos_token_ids=eos, stopping_reason=stop.reason))
            record['stage_times'] = {'preprocessing_s': prepared - started,
                                     'host_to_device_s': transferred - prepared,
                                     'prefill_all_generation_and_stopping_s': generated - transferred,
                                     'final_decode_and_parse_s': time.perf_counter() - generated}
        except Exception as error:
            if stop is not None:
                record.update(raw_output=self.processor.tokenizer.decode(stop.token_ids, skip_special_tokens=False),
                              generated_token_ids=stop.token_ids, token_count=len(stop.token_ids))
            record.update(status='oom' if isinstance(error, torch.cuda.OutOfMemoryError) else 'runtime_error',
                          stopping_reason='oom' if isinstance(error, torch.cuda.OutOfMemoryError) else 'exception',
                          error=f'{type(error).__name__}: {error}')
            raise
        finally:
            record['peak_allocated_GiB'] = torch.cuda.max_memory_allocated(self.device) / 1024**3
            record['peak_reserved_GiB'] = torch.cuda.max_memory_reserved(self.device) / 1024**3
            record['decision_processing_s'] = time.perf_counter() - started
            if self.stream_clock is not None:
                record['stream_completed_s'] = self.stream_clock.elapsed()
            self.records.append(record)
            self.last_stage_times = record.get('stage_times', {})
        if record['status'] != 'valid':
            raise DecisionError(f'{record["status"]}: {record["stopping_reason"]}')
        # Binary values are only an interface to the unchanged strict-threshold
        # scheduler. They are NOT continuous scores or confidence estimates.
        return {self.view: 1.0 if record['final_label'] == 'FAIL' else 0.0}


def preload(plan, task, episode, view):
    return preload_episode(REPO / plan['tasks']['tasks'][task]['prepared_dataset'], episode, [view], method='vlm')


def window_at(payload, endpoint):
    # wm_ selects only the existing K/L index arithmetic, not any WM inference.
    return input_window(endpoint, payload['raw'], payload['elapsed'], 'wm_vlm_alignment', 4, 4)


def warmup(plan, adapter):
    task = next(iter(plan['tasks']['tasks']))
    adapter.task, adapter.view = task, 'fix'
    payload = preload(plan, task, plan['config']['development']['episode_index'], 'fix')
    adapter.reset_episode()
    for endpoint in range(7, 7 + plan['config']['streaming']['warmup_windows']):
        adapter.score(payload, window_at(payload, endpoint))
    return {'task': task, 'view': 'fix', 'episode': plan['config']['development']['episode_index'],
            'requests': list(adapter.records), 'model_load_s': adapter.load_s}


def development_cases(plan, model_name):
    config = plan['config']['development']
    extra = config.get('additional_last_endpoints', {}).get(model_name, {})
    for task in plan['tasks']['tasks']:
        for view in plan['config']['inputs']['views']:
            yield task, view, config['episode_index'], config['endpoints']
            for episode in extra.get(task, []):
                yield task, view, episode, ['last']


def smoke(plan, output):
    report = {'fingerprint': plan['fingerprint'], 'purpose': 'development_only_calibration_successes',
              'status': 'running', 'cases': [], 'warmup': {}, 'stream_checks': {}, 'hardware': hardware()}
    try:
        for name in plan['config']['models']:
            print(f'Loading {name} for development only', flush=True)
            adapter = VLMAdapter(plan, name)
            try:
                report['warmup'][name] = warmup(plan, adapter)
            except Exception:
                report['warmup'][name] = {'requests': adapter.records}
                raise
            for task, view, episode, requested_endpoints in development_cases(plan, name):
                payload = preload(plan, task, episode, view)
                adapter.task, adapter.view = task, view
                for requested_endpoint in requested_endpoints:
                    endpoint = len(payload['raw']) - 1 if requested_endpoint == 'last' else requested_endpoint
                    adapter.reset_episode()
                    try:
                        adapter.score(payload, window_at(payload, endpoint))
                    except DecisionError:
                        pass
                    finally:
                        case = {'model': name, 'task': task, 'view': view, 'episode': episode,
                                'endpoint': endpoint, 'requests': list(adapter.records)}
                        report['cases'].append(case)
                        write_json(output / 'smoke.json', report)
                    result = adapter.records[-1]
                    print(f'{name}/{task}/{view}/e{endpoint}: {result["status"]}; '
                          f'{result["token_count"]} tokens; {result["decision_processing_s"]:.2f}s; '
                          f'{result["peak_reserved_GiB"]:.2f} GiB reserved', flush=True)
            # Real scheduler checks use only calibration episodes, with all
            # views/tasks for the video protocol and the original one for V2/V3.
            stream_cases = [(next(iter(plan['tasks']['tasks'])), 'fix')]
            if plan['config']['development'].get('stream_all_tasks_views', False):
                stream_cases = [(task, view) for task in plan['tasks']['tasks']
                                for view in plan['config']['inputs']['views']]
            for task, view in stream_cases:
                episode = plan['config']['development']['episode_index']
                adapter.task, adapter.view = task, view
                payload = preload(plan, task, episode, view)
                adapter.stream_clock = StreamClock()
                replay, stream_error = None, None
                try:
                    replay = replay_episode(payload, adapter, lambda view, t: .5, [view],
                                            'wm_vlm_alignment', 4, 4, clock=adapter.stream_clock,
                                            stop_at_first_alarm=plan['config']['streaming']['stop_at_first_FAIL'])
                except DecisionError as error:
                    stream_error = str(error)
                key = f'{name}/{task}/{view}' if len(stream_cases) > 1 else name
                report['stream_checks'][key] = summarize_trial(plan,
                    {'model': name, 'task': task, 'view': view, 'episode': episode, 'outcome': 'success'},
                    payload, replay, list(adapter.records), stream_error)
                print(f'{key}: calibration replay {report["stream_checks"][key]["status"]}; '
                      f'{len(adapter.records)} requests; alarm={bool(report["stream_checks"][key]["alarm"])}', flush=True)
                write_json(output / 'smoke.json', report)
                adapter.stream_clock = None
            del adapter
            gc.collect()
            import torch
            torch.cuda.empty_cache()
        records = [row['requests'][0] for row in report['cases']]
        report['status'] = 'passed' if (all(r['status'] == 'valid' for r in records)
            and all(r['status'] == 'complete' for r in report['stream_checks'].values())) else 'failed'
        report['summary'] = {}
        for name in plan['config']['models']:
            rows = [row['requests'][0] for row in report['cases'] if row['model'] == name]
            report['summary'][name] = {'requests': len(rows), 'valid': sum(r['status'] == 'valid' for r in rows),
                'mean_processing_s': float(np.mean([r['decision_processing_s'] for r in rows])),
                'max_processing_s': max(r['decision_processing_s'] for r in rows),
                'token_count_range': [min(r['token_count'] for r in rows), max(r['token_count'] for r in rows)],
                'max_reserved_GiB': max(r['peak_reserved_GiB'] for r in rows)}
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        write_json(output / 'smoke.json', report)
    if report['status'] != 'passed':
        raise RuntimeError('Development completion check failed; full benchmark remains blocked')


def hardware():
    import torch
    return {'gpu': torch.cuda.get_device_name(), 'cuda': torch.version.cuda,
            'total_memory_bytes': torch.cuda.get_device_properties(0).total_memory}


def result_path(plan, output, trial):
    root = plan['config']['outputs']['per_task_root'].format(
        task=trial['task'], protocol_id=plan['config']['protocol_id'], run_id=output.name)
    return REPO / root / (trial['id'] + '.json')


def summarize_trial(plan, trial, payload, replay, requests, error=None):
    end = float(payload['raw_wall'][-1] - payload['raw_wall'][0])
    alarm = replay['alarms'].get(trial['view']) if replay else None
    onset = None
    if trial['outcome'] == 'failure':
        annotations = plan['labels']['tasks'][trial['task']].get('failure_onsets', {}).get(trial['episode'], {})
        view = annotation_views(annotations.get(plan['config']['evaluation']['primary_annotator_id'], {})).get(trial['view'])
        if view is not None:
            index = view['frame_index']
            if not 0 <= index < len(payload['raw_wall']):
                raise ValueError('Onset frame outside raw episode')
            from benchmark_consensus_onsets import elapsed_onset
            onset = elapsed_onset(view, payload['raw_wall'])
    events = replay['events'] if replay else []
    for record in requests:
        if record.get('stream_completed_s') is not None:
            record['late'] = record['stream_completed_s'] > end
    for record, event in zip(requests, events):
        record['stream_timing'] = event
        record['late'] = event['alarm_ready_s'] > end
    delay = alarm['alarm_time_s'] - onset if alarm and onset is not None else None
    result = {**trial, 'status': 'indeterminate' if error else 'complete', 'error': error,
            'raw_source_end_s': end, 'last_prepared_time_s': float(payload['elapsed'][-1]),
            'onset_s': onset, 'alarm': alarm, 'late_alarm': bool(alarm and alarm['alarm_time_s'] > end),
            'pre_onset_alarm': bool(delay is not None and delay < 0),
            'pre_onset_evidence': bool(alarm and onset is not None and alarm['source_time_s'] < onset),
            'post_onset_evidence_delay_s': delay if delay is not None and delay >= 0 and alarm['source_time_s'] >= onset else None,
            'preload_s': payload['preload_s'], 'replay': replay, 'requests': requests,
            'late_decisions': sum(r.get('late', False) for r in requests)}
    if plan['config'].get('evaluation_protocol'):
        from benchmark_frozen_protocol import load_protocol, alarm_run, allows_anticipation, recorded_curve, uses_post_onset
        spec = load_protocol(plan['config']['evaluation_protocol'])
        result.update(evaluation_protocol=spec['protocol_id'], any_time_alarm=alarm)
        if trial['outcome'] == 'failure':
            group = episode_labels(plan['labels']['tasks'][trial['task']])[trial['episode']]
            outcome = alarm_run(recorded_curve(events, trial['view'], 'episode'),
                index if onset is not None else None, onset,
                allow_anticipation=allows_anticipation(trial['task'], group['failure_origin']),
                first_post_onset=uses_post_onset(spec),
                complete=bool(replay and replay['stop_reason'] == 'source_exhausted' and not error))
            detected = None
            if outcome['post_alarm']:
                event = next(e for e in events if e['raw_frame'] == outcome['detection_raw_frame'])
                detected = {'raw_frame': event['raw_frame'], 'endpoint': event['endpoint'],
                            'source_time_s': event['source_time_s'], 'alarm_time_s': event['alarm_ready_s'],
                            'score': event['scores'][trial['view']], 'threshold': .5}
            result.update(frozen_onset_result=outcome, alarm=detected,
                post_onset_evidence_delay_s=outcome['detection_delay_s'],
                pre_onset_evidence=outcome['early_alarm'], pre_onset_alarm=outcome['early_completion'],
                late_alarm=bool(detected and detected['alarm_time_s'] > end))
    return result


def generate_report(plan, output):
    rows = []
    for trial in plan['trials']:
        path = result_path(plan, output, trial)
        if path.exists():
            result = json.loads(path.read_text())
            if result.get('fingerprint') != plan['fingerprint']:
                raise ValueError(f'Result provenance differs: {path}')
        else:
            interrupted = path.with_suffix('.started.json').exists()
            result = {**trial, 'status': 'indeterminate' if interrupted else 'unattempted',
                      'error': 'interrupted_or_in_progress' if interrupted else None}
        rows.append(result)
    summary = []
    for model in plan['config']['models']:
        for task in plan['tasks']['tasks']:
            for view in plan['config']['inputs']['views']:
                selected = [r for r in rows if (r['model'], r['task'], r['view']) == (model, task, view)]
                row = {'model': model, 'task': task, 'view': view, 'planned': len(selected),
                       'complete': sum(r['status'] == 'complete' for r in selected),
                       'indeterminate': sum(r['status'] == 'indeterminate' for r in selected),
                       'unattempted': sum(r['status'] == 'unattempted' for r in selected)}
                for outcome, metric in [('success', 'FAR'), ('failure', 'FDR')]:
                    subset = [r for r in selected if r['outcome'] == outcome]
                    unknown = sum(r['status'] != 'complete' for r in subset)
                    any_alarm = sum(bool(r.get('alarm')) for r in subset)
                    on_time = sum(bool(r.get('alarm')) and not r['late_alarm'] for r in subset)
                    row.update({f'{metric}_denominator': len(subset), f'{metric}_unknown': unknown,
                                f'{metric}_any_count': any_alarm, f'{metric}_on_time_count': on_time,
                                f'{metric}_any': any_alarm / len(subset) if not unknown else None,
                                f'{metric}_on_time': on_time / len(subset) if not unknown else None,
                                f'{metric}_any_lower_bound': any_alarm / len(subset),
                                f'{metric}_any_upper_bound': (any_alarm + unknown) / len(subset)})
                failures = [r for r in selected if r['outcome'] == 'failure']
                # Annotation membership is determined from the frozen labels even
                # for unattempted/error episodes; do not shrink the denominator.
                annotated = [r for r in failures if view in annotation_views(
                    plan['labels']['tasks'][task].get('failure_onsets', {}).get(r['episode'], {}).get(
                        plan['config']['evaluation']['primary_annotator_id'], {}))]
                delays = [r['post_onset_evidence_delay_s'] for r in annotated
                          if r.get('post_onset_evidence_delay_s') is not None and not r['late_alarm']]
                row.update(late_alarms=sum(r.get('late_alarm', False) for r in selected),
                           late_decisions=sum(r.get('late_decisions', 0) for r in selected),
                           pre_onset_alarms=sum(bool(r.get('pre_onset_alarm')) for r in failures),
                           pre_onset_evidence=sum(bool(r.get('pre_onset_evidence')) for r in failures),
                           valid_misses=sum(r['status'] == 'complete' and not r.get('alarm') for r in failures),
                           on_time_misses=sum(r['status'] == 'complete' and (not r.get('alarm') or r['late_alarm']) for r in failures),
                           annotated_failures=len(annotated),
                           annotated_unknown=sum(r['status'] != 'complete' for r in annotated),
                           conditional_delay_n=len(delays),
                           conditional_delay_mean_s=float(np.mean(delays)) if delays else None,
                           conditional_delay_sd_s=float(np.std(delays)) if delays else None)
                for deadline in plan['config']['evaluation']['deadline_seconds']:
                    count = sum(d <= deadline for d in delays)
                    row[f'detected_by_{deadline}s_count'] = count
                    row[f'detected_by_{deadline}s_denominator'] = len(annotated)
                    row[f'detected_by_{deadline}s_rate'] = count / len(annotated) if annotated and not row['annotated_unknown'] else None
                for status in ['malformed', 'truncated', 'timed_out', 'oom', 'runtime_error']:
                    row[status + '_requests'] = sum(q['status'] == status for r in selected for q in r.get('requests', []))
                summary.append(row)
    frozen_note = ''
    if plan['config'].get('evaluation_protocol'):
        from benchmark_frozen_protocol import load_protocol, uses_post_onset
        spec = load_protocol(plan['config']['evaluation_protocol'])
        frozen_note = (' FDR requires a post-onset alarm; early-only alarms are misses. '
                       + ('Latency uses the first positive post-onset-source decision and the exact annotated onset time; '
                          'sustained pre-onset runs are reported separately, without negative latency credit.'
                          if uses_post_onset(spec) else
                          'Persistent early runs can have signed delay except simulated induced failures.'))
    write_json(output / 'report.json', {'protocol_id': plan['config']['protocol_id'],
               'evaluation_protocol': plan['config'].get('evaluation_protocol'),
               'fingerprint': plan['fingerprint'], 'summary': summary,
               'note': 'Errors/unattempted episodes remain in denominators; rates are null with bounds until resolved. '
                       'Delays require post-onset source evidence and an on-time decision. Binary labels have no continuous AUROC.' + frozen_note})
    with (output / 'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    fields = ['model', 'task', 'view', 'complete', 'planned', 'indeterminate', 'unattempted',
              'FAR_any_count', 'FAR_denominator', 'FDR_on_time_count', 'FDR_denominator',
              'late_alarms', 'conditional_delay_n', 'conditional_delay_mean_s']
    lines = ['# VLM reasoning streaming benchmark', '',
             'On-time FDR excludes decisions returned after raw episode end. FAR column counts any alarm; '
             'see CSV/JSON for on-time FAR, late decisions, errors, deadline rates, and unknown bounds.' + frozen_note, '',
             '| ' + ' | '.join(fields) + ' |', '| ' + ' | '.join(['---'] * len(fields)) + ' |']
    lines.extend('| ' + ' | '.join(str(row.get(k)) for k in fields) + ' |' for row in summary)
    (output / 'report.md').write_text('\n'.join(lines) + '\n')
    table = '<table><tr>' + ''.join(f'<th>{html.escape(k)}</th>' for k in fields) + '</tr>'
    table += ''.join('<tr>' + ''.join(f'<td>{html.escape(str(row.get(k)))}</td>' for k in fields) + '</tr>' for row in summary)
    (output / 'index.html').write_text('<!doctype html><meta charset="utf-8"><title>VLM reasoning benchmark</title>'
        '<style>body{font:14px system-ui;margin:24px}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:6px}</style>'
        '<h1>VLM reasoning benchmark</h1><p>' + html.escape(lines[2]) + '</p>' + table + '</table>')


def run(plan, output):
    """One resident model, separate view streams; no in-episode retries."""
    try:
        for name in plan['config']['models']:
            pending = [t for t in plan['trials'] if t['model'] == name and not result_path(plan, output, t).exists()]
            if not pending:
                continue
            print(f'Loading {name}: {len(pending)} pending episodes', flush=True)
            adapter = VLMAdapter(plan, name)
            warmup_path = output / f'warmup_{name}_{time.time_ns()}.json'
            try:
                write_json(warmup_path, warmup(plan, adapter))
            except Exception:
                write_json(warmup_path, {'status': 'failed', 'requests': adapter.records})
                raise
            for number, trial in enumerate(pending, 1):
                path = result_path(plan, output, trial)
                path.parent.mkdir(parents=True, exist_ok=True)
                claim = path.with_suffix('.started.json')
                if claim.exists():
                    raise RuntimeError(f'Interrupted attempt must be reviewed; no automatic retry: {claim}')
                write_json(claim, {'trial': trial, 'fingerprint': plan['fingerprint'], 'started_unix': time.time()})
                adapter.task, adapter.view = trial['task'], trial['view']
                adapter.reset_episode()
                payload, replay, error, fatal = None, None, None, None
                try:
                    payload = preload(plan, trial['task'], trial['episode'], trial['view'])
                    adapter.stream_clock = StreamClock()
                    replay = replay_episode(payload, adapter, lambda view, t: 0.5, [trial['view']],
                                            'wm_vlm_alignment', horizon=4, history_len=4, clock=adapter.stream_clock,
                                            stop_at_first_alarm=plan['config']['streaming']['stop_at_first_FAIL'])
                except Exception as exc:
                    error = f'{type(exc).__name__}: {exc}'
                    if not isinstance(exc, DecisionError):
                        fatal = exc
                if payload is not None:
                    result = summarize_trial(plan, trial, payload, replay, list(adapter.records), error)
                else:
                    result = {**trial, 'status': 'indeterminate', 'error': error, 'requests': list(adapter.records)}
                result['fingerprint'] = plan['fingerprint']
                write_json(path, result)
                print(f'{name} {number}/{len(pending)} {trial["task"]}/{trial["view"]}/{trial["episode"]}: '
                      f'{result["status"]}; alarm={bool(result.get("alarm"))}; late={result.get("late_alarm")}', flush=True)
                if number % 20 == 0 or error:
                    generate_report(plan, output)
                if fatal is not None:
                    raise fatal
            del adapter
            gc.collect()
            import torch
            torch.cuda.empty_cache()
    finally:
        generate_report(plan, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'smoke', 'run', 'report'])
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--smoke-report', type=Path)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['HF_DATASETS_OFFLINE'] = '1'
    plan = make_plan(args.config.resolve())
    output = args.output.resolve()
    if args.mode == 'run':
        if args.smoke_report is None:
            parser.error('run requires --smoke-report from the frozen development check')
        check = json.loads(args.smoke_report.read_text())
        if check.get('status') != 'passed' or check.get('fingerprint') != plan['fingerprint']:
            raise ValueError('Development check failed, or code/settings/data identity changed; freeze and validate again')
        if check['hardware'] != hardware():
            raise ValueError('GPU/CUDA differs from development check')
    if output.exists():
        if not (args.resume or args.mode == 'report'):
            raise FileExistsError(f'Choose a fresh output folder, or explicitly --resume: {output}')
        saved = json.loads((output / 'plan.json').read_text())
        if saved['fingerprint'] != plan['fingerprint']:
            raise ValueError('Frozen plan differs; existing results are preserved')
    else:
        if args.mode == 'report' or args.resume:
            raise FileNotFoundError(output)
        output.mkdir(parents=True)
        write_json(output / 'plan.json', plan)
        snapshots = output / 'frozen_sources'
        for relative in plan['identity']['files']:
            path = snapshots / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((REPO / relative).read_bytes())
    with (output / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.mode in {'prepare', 'run', 'report'}:
            # Bind task-local output roots to this plan; disallow collisions with
            # a different report folder that happens to share the same basename.
            for task in plan['tasks']['tasks']:
                trial = next(t for t in plan['trials'] if t['task'] == task)
                owner = result_path(plan, output, trial).parent / 'owner.json'
                identity = {'report_root': str(output), 'fingerprint': plan['fingerprint']}
                if owner.exists() and json.loads(owner.read_text()) != identity:
                    raise FileExistsError(f'Result directory belongs to another run: {owner}')
                if not owner.exists():
                    write_json(owner, identity)
            generate_report(plan, output)
        if args.mode == 'smoke':
            if (output / 'smoke.json').exists():
                raise FileExistsError('Development reports are immutable; choose a fresh folder')
            smoke(plan, output)
        elif args.mode == 'run':
            run(plan, output)
        print(f'{args.mode} finished: {output}', flush=True)


if __name__ == '__main__':
    main()
