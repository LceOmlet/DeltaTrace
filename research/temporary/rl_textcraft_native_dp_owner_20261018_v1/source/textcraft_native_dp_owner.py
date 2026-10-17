"""Call the unchanged AgentGym episode owner inside an existing VERL DP rank.

The trainer still opens/closes the original collective rollout context. Only
the placement of the existing owner loop changes; every local model call goes
through the original ActorRolloutRefWorker.generate_sequences implementation.
"""
from types import SimpleNamespace


def collect_native_textcraft(worker, data):
    from omegaconf import OmegaConf
    from hydra.utils import instantiate
    from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector

    manager = worker.rollout_sharding_manager
    engine = worker.rollout.inference_engine.llm_engine
    if manager.tp_size != 1 or engine.dp_group is not None:
        raise ValueError('DP-local native episodes require independent native TP1 engines')
    if not getattr(worker, '_owner_rollout_context_open', False):
        raise RuntimeError('The original complete-rollout context must already be open')

    metadata = dict(data.meta_info)
    configuration = OmegaConf.create(metadata.pop('native_textcraft_configuration'))
    is_train = metadata.pop('native_textcraft_is_train')
    data.meta_info = metadata
    cache = getattr(worker, '_native_textcraft_episode_owners', None)
    if cache is None:
        train, evaluation = instantiate(configuration.env.factory,
            configuration=configuration, tokenizer=worker.tokenizer, _recursive_=False)
        collector = TrajectoryCollector(configuration, worker.tokenizer,
            processor=getattr(worker, 'processor', None))
        cache = (train, evaluation, collector)
        worker._native_textcraft_episode_owners = cache
    train, evaluation, collector = cache
    owner = train if is_train else evaluation
    local = SimpleNamespace(world_size=1, generate_sequences=worker.generate_sequences)
    return owner._collect_native_trajectories(collector, data, local, is_train)
