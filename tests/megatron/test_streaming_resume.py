import ast
import copy
import importlib.util
import json
import os
import pytest
import random
import sys
import torch
import types
from datetime import timedelta
from pathlib import Path
from torch.utils.data import DataLoader, IterableDataset
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(os.environ.get('SWIFT_STREAMING_BASE_SOURCE', ROOT / 'swift' / 'megatron' / 'trainers' / 'base.py'))
UTILS = ROOT / 'swift' / 'megatron' / 'trainers' / 'utils.py'
RUN_GLOO = os.environ.get('SWIFT_RUN_GLOO_TEST') == '1'


def _extract_function(path, class_name, function_name, namespace):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    owner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    function = copy.deepcopy(
        next(node for node in owner.body if isinstance(node, ast.FunctionDef) and node.name == function_name))
    function.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    exec(compile(module, str(path), 'exec'), namespace)
    return namespace[function_name]


class _Logger:

    def info(self, *args, **kwargs):
        return None


class _Mpu:

    data_parallel_size = 1

    @classmethod
    def get_data_parallel_world_size(cls):
        return cls.data_parallel_size

    @staticmethod
    def get_data_parallel_group():
        return torch.distributed.group.WORLD


def _load_dispatcher():
    path = ROOT / 'swift' / 'dataloader' / 'dispatcher.py'
    module_name = '_streaming_resume_dispatcher'
    saved_swift = sys.modules.get('swift')
    saved_utils = sys.modules.get('swift.utils')
    try:
        if saved_swift is None:
            swift = types.ModuleType('swift')
            swift.__path__ = [str(ROOT / 'swift')]
            sys.modules['swift'] = swift
        if saved_utils is None:
            utils = types.ModuleType('swift.utils')
            utils.to_device = lambda data, *args, **kwargs: data
            sys.modules['swift.utils'] = utils
        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module.DataLoaderDispatcher
    finally:
        if saved_utils is None:
            sys.modules.pop('swift.utils', None)
        if saved_swift is None:
            sys.modules.pop('swift', None)


DataLoaderDispatcher = _load_dispatcher()


def _load_megatron_loader_helpers():
    tree = ast.parse(UTILS.read_text(encoding='utf-8'))
    class_node = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'MegatronDataLoaderDispatcher')
    builder_node = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'build_streaming_dataloader')
    namespace = {
        'DataLoaderDispatcher': DataLoaderDispatcher,
        'mpu': _Mpu,
        'torch': torch,
    }
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(class_node)], type_ignores=[])), str(UTILS),
            'exec'), namespace)
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(builder_node)], type_ignores=[])), str(UTILS),
            'exec'), namespace)
    return namespace['MegatronDataLoaderDispatcher'], namespace['build_streaming_dataloader']


MegatronDataLoaderDispatcher, build_streaming_dataloader = _load_megatron_loader_helpers()

_METHOD_NAMESPACE = {'logger': _Logger(), 'mpu': _Mpu}
cyclic_iter = _extract_function(BASE, 'BaseMegatronTrainer', 'cyclic_iter', _METHOD_NAMESPACE)
prepare_data_iterator = _extract_function(BASE, 'BaseMegatronTrainer', '_prepare_data_iterator', _METHOD_NAMESPACE)


class _EpochStream(IterableDataset):

    def __init__(self, size=8, seed=19):
        self.size = size
        self.seed = seed
        self.epoch = 0
        self.seen_epochs = []

    def set_epoch(self, epoch):
        self.epoch = epoch
        self.seen_epochs.append(epoch)

    def __iter__(self):
        order = list(range(self.size))
        random.Random(self.seed + self.epoch).shuffle(order)
        yield from (self.epoch * 1000 + index for index in order)


def _args(**overrides):
    values = dict(
        streaming=True,
        rlhf_type='sft',
        micro_batch_size=2,
        num_microbatches=1,
        global_batch_size=2,
        num_generations=1,
        steps_per_generation=1,
        num_train_epochs=None,
        train_iters=0,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        dataloader_prefetch_factor=None,
        dataloader_persistent_workers=False,
        dataloader_multiprocessing_context=None,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _loader(dataset, args, skip_batches=0):
    loader = build_streaming_dataloader(args, dataset, lambda rows: tuple(rows))
    loader.skip_batches = skip_batches
    return loader


class _TrainerHarness:

    cyclic_iter = cyclic_iter
    _prepare_data_iterator = prepare_data_iterator

    def __init__(self, args, consumed=0, iteration=0, dataset_size=8, val_dataset=None, prepositioned_batches=0):
        self.args = args
        self.state = SimpleNamespace(consumed_train_samples=consumed, iteration=iteration)
        self.unwrapped_models = [SimpleNamespace(training=True)]
        self.dataset = _EpochStream(dataset_size)
        self.val_dataset = val_dataset
        self.prepositioned_batches = prepositioned_batches
        self.prepare_calls = 0

    def _prepare_dataloader(self, train_dataset, val_dataset=None):
        self.prepare_calls += 1
        train_dataloader = _loader(self.dataset, self.args, self.prepositioned_batches)
        val_dataloader = _loader(val_dataset, self.args) if val_dataset is not None else None
        return train_dataloader, val_dataloader


def _batches(iterator, count):
    return [tuple(next(iterator)) for _ in range(count)]


def _stream(args=None, consumed=0, iteration=0, dataset_size=8, val_dataset=None, prepositioned_batches=0):
    trainer = _TrainerHarness(args or _args(), consumed, iteration, dataset_size, val_dataset, prepositioned_batches)
    train_iterator, val_iterator = trainer._prepare_data_iterator(trainer.dataset, val_dataset)
    return trainer, train_iterator, val_iterator


@pytest.fixture(autouse=True)
def _reset_data_parallel_size():
    _Mpu.data_parallel_size = 1
    yield
    _Mpu.data_parallel_size = 1


def test_streaming_builder_uses_real_dataloader_and_megatron_dispatcher():
    args = _args(micro_batch_size=2)
    dataset = _EpochStream(size=4)
    loader = _loader(dataset, args)

    assert isinstance(loader, MegatronDataLoaderDispatcher)
    assert isinstance(loader.base_dataloader, DataLoader)
    batches = _batches(iter(loader), 2)
    assert sorted(item for batch in batches for item in batch) == list(range(4))


@pytest.mark.parametrize('prefix_batches', [1, 2, 5])
def test_resume_matches_uninterrupted_stream_across_epoch_positions(prefix_batches):
    args = _args(micro_batch_size=2)
    _, fresh_iterator, _ = _stream(args, dataset_size=4)
    uninterrupted = _batches(fresh_iterator, prefix_batches + 3)

    _, resumed_iterator, _ = _stream(args, consumed=prefix_batches * args.micro_batch_size, dataset_size=4)
    resumed = _batches(resumed_iterator, 3)

    assert resumed == uninterrupted[prefix_batches:]


def test_resume_preserves_deterministic_set_epoch_order_and_does_not_double_skip_finite_cycles():
    args = _args(micro_batch_size=2, num_train_epochs=2)
    fresh_trainer, fresh_iterator, _ = _stream(args, dataset_size=4)
    uninterrupted = _batches(fresh_iterator, 8)

    resumed_trainer, resumed_iterator, _ = _stream(args, consumed=args.micro_batch_size, dataset_size=4)
    resumed = _batches(resumed_iterator, 7)

    assert resumed == uninterrupted[1:]
    assert fresh_trainer.dataset.seen_epochs[:3] == [0, 1, 2]
    assert resumed_trainer.dataset.seen_epochs[:3] == [0, 1, 2]


def test_two_independent_main_iterators_share_the_same_resume_offset():
    args = _args(micro_batch_size=2, global_batch_size=2)
    _, first_iterator, _ = _stream(args, consumed=4, dataset_size=8)
    _, second_iterator, _ = _stream(args, consumed=4, dataset_size=8)

    assert next(first_iterator) == next(second_iterator)


def test_zero_resume_validation_and_origin_resample_start_fresh():
    _Mpu.data_parallel_size = 1
    args = _args(micro_batch_size=2)
    validation = _EpochStream(size=4, seed=31)
    fresh_trainer, fresh_iterator, _ = _stream(args, dataset_size=4)
    fresh_first = next(fresh_iterator)

    resumed_trainer, resumed_iterator, validation_iterator = _stream(
        args, consumed=4, iteration=1, dataset_size=4, val_dataset=validation)
    normal_first = next(resumed_iterator)
    validation_first = next(validation_iterator)
    expected_validation = next(iter(_loader(_EpochStream(size=4, seed=31), args)))

    origin_trainer = _TrainerHarness(args, consumed=4, iteration=1, dataset_size=4)
    origin_iterator, _ = origin_trainer._prepare_data_iterator(origin_trainer.dataset, use_origin_cyclic=True)

    assert fresh_trainer.dataset.seen_epochs[0] == 0
    assert normal_first != fresh_first
    assert validation_first == expected_validation
    assert next(origin_iterator) == fresh_first
    assert resumed_trainer.prepare_calls == 1


@pytest.mark.parametrize('data_parallel_size', [2, 4])
@pytest.mark.parametrize('num_microbatches', [1, 2])
def test_sft_skip_uses_micro_batch_and_data_parallel_width_not_gradient_accumulation(
        data_parallel_size, num_microbatches):
    _Mpu.data_parallel_size = data_parallel_size
    args = _args(
        micro_batch_size=2,
        num_microbatches=num_microbatches,
        global_batch_size=2 * data_parallel_size * num_microbatches,
    )
    checkpoint_iterations = 2
    skip_batches = args.num_microbatches * checkpoint_iterations
    consumed = args.global_batch_size * checkpoint_iterations
    _, expected_iterator, _ = _stream(args, dataset_size=16)
    expected = _batches(expected_iterator, skip_batches + 1)[-1]
    _, resumed_iterator, _ = _stream(args, consumed=consumed, dataset_size=16)

    assert next(resumed_iterator) == expected


@pytest.mark.parametrize(('rlhf_type', 'num_generations'), [('grpo', 3), ('gkd', 1)])
def test_rollout_skip_uses_num_generations_without_steps_multiplier(rlhf_type, num_generations):
    _Mpu.data_parallel_size = 2
    args = _args(
        rlhf_type=rlhf_type,
        micro_batch_size=2,
        num_generations=num_generations,
        steps_per_generation=4,
        global_batch_size=2 * 2 * num_generations,
    )
    checkpoint_iteration = 4
    consumed = args.global_batch_size * checkpoint_iteration
    _, expected_iterator, _ = _stream(args, dataset_size=8)
    expected = _batches(expected_iterator, checkpoint_iteration + 1)[-1]
    _, resumed_iterator, _ = _stream(args, consumed=consumed, iteration=checkpoint_iteration, dataset_size=8)

    assert next(resumed_iterator) == expected


@pytest.mark.parametrize(
    ('rlhf_type', 'consumed', 'data_parallel_size'),
    [('sft', 1, 1), ('grpo', 11, 2), ('gkd', 13, 2)],
)
def test_nondivisible_streaming_resume_is_rejected_before_dataloader(rlhf_type, consumed, data_parallel_size):
    _Mpu.data_parallel_size = data_parallel_size
    args = _args(
        rlhf_type=rlhf_type,
        micro_batch_size=2,
        num_generations=3,
        global_batch_size=2 * data_parallel_size * (3 if rlhf_type == 'grpo' else 1),
    )
    trainer = _TrainerHarness(args, consumed=consumed, dataset_size=8)

    with pytest.raises(RuntimeError, match='divisible'):
        trainer._prepare_data_iterator(trainer.dataset)

    assert trainer.prepare_calls == 0


@pytest.mark.parametrize(('rlhf_type', 'num_generations'), [('grpo', 3), ('gkd', 1)])
def test_mid_rollout_streaming_resume_is_rejected(rlhf_type, num_generations):
    _Mpu.data_parallel_size = 2
    args = _args(
        rlhf_type=rlhf_type,
        micro_batch_size=2,
        num_generations=num_generations,
        steps_per_generation=4,
        global_batch_size=2 * 2 * num_generations,
    )
    trainer = _TrainerHarness(args, consumed=args.global_batch_size * 3, iteration=3, dataset_size=8)

    with pytest.raises(RuntimeError, match='middle of a rollout buffer'):
        trainer._prepare_data_iterator(trainer.dataset)

    assert trainer.prepare_calls == 0


def test_nonstreaming_prepositioned_sampler_is_not_skipped_again():
    # The finite sampler is already positioned by its own cursor; streaming resume logic must stay off.
    args = _args(streaming=False, micro_batch_size=2, global_batch_size=2)
    _, fresh_iterator, _ = _stream(args, dataset_size=8)
    expected = _batches(fresh_iterator, 2)[-1]
    _, resumed_iterator, _ = _stream(args, consumed=args.global_batch_size, dataset_size=8, prepositioned_batches=1)

    assert next(resumed_iterator) == expected


def _gloo_resume_worker(rank, init_method, result_path):
    torch.distributed.init_process_group(
        'gloo', rank=rank, world_size=2, init_method=init_method, timeout=timedelta(seconds=30))
    try:
        _Mpu.data_parallel_size = 2
        args = _args(micro_batch_size=1, global_batch_size=2)
        _, fresh_iterator, _ = _stream(args, dataset_size=8)
        uninterrupted = _batches(fresh_iterator, 8)
        _, resumed_iterator, _ = _stream(args, consumed=args.global_batch_size * 5, dataset_size=8)
        resumed = _batches(resumed_iterator, 3)
        gathered = [None, None]
        payload = {'rank': rank, 'uninterrupted': uninterrupted, 'resumed': resumed}
        torch.distributed.all_gather_object(gathered, payload)
        if rank == 0:
            Path(result_path).write_text(json.dumps(gathered), encoding='utf-8')
        torch.distributed.barrier()
    finally:
        torch.distributed.destroy_process_group()


def _assert_gloo_payload(gathered):
    gathered.sort(key=lambda item: item['rank'])
    for payload in gathered:
        assert payload['resumed'] == payload['uninterrupted'][5:]
    assert gathered[0]['uninterrupted'] != gathered[1]['uninterrupted']


@pytest.mark.skipif(
    not RUN_GLOO or not torch.distributed.is_available() or not torch.distributed.is_gloo_available(),
    reason='Gloo integration is an explicit server-side CPU harness run on Windows',
)
def test_two_rank_gloo_dispatcher_scatter_and_epoch_restart(tmp_path):
    """Run the patched resume trajectory on both ranks through real Gloo scatter."""
    init_file = tmp_path / 'gloo_init'
    result_file = tmp_path / 'gloo_result.json'
    torch.multiprocessing.spawn(
        _gloo_resume_worker,
        args=(init_file.as_uri(), str(result_file)),
        nprocs=2,
        join=True,
    )
    _assert_gloo_payload(json.loads(result_file.read_text(encoding='utf-8')))
