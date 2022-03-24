import asyncio
import contextlib
from enum import Enum
from threading import local
from typing import Any, Iterable, Optional, Sequence

import torch

from hivemind.averaging.allreduce import AveragingMode
from hivemind.averaging.control import StepControl
from hivemind.averaging.group_info import GroupInfo
from hivemind.averaging.load_balancing import load_balance_peers
from hivemind.averaging.matchmaking import MatchmakingException
from hivemind.dht import DHT
from hivemind.optim.power_sgd_averager import AllReducePhases, PowerSGDGradientAverager
from hivemind.utils import get_logger
from hivemind.utils.asyncio import enter_asynchronously
from hivemind.utils.math import get_flatten_greedy_dims, orthogonalize_

GatheredData = Any
logger = get_logger(__name__)

class PowerEFGradientAverager(PowerSGDGradientAverager):
    """
    A new and powerful way to average gradients

    :note: due to the above rule, PowerEF is *not* shape-invariant. For instance, a
     matrix of shape (256, 256) be compressed differently if you .reshape it to (32, 32, 32).

    :param parameters: pytorch parameters for which to aggregate gradients
    :param averager_rank: rank of compressed gradients
    :param dht: a DHT isntance connected to the rest of the swarm. See hivemind.DHT docs
    :param prefix: a unique DHT key used for matchmaking. E.g. this can be your experiment name with optional suffixes
    :param reuse_grad_buffers: if True, use model's .grad buffers for accumulating gradients over multiple steps.
      This is more memory efficient, but it requires that the user does *not* call zero_grad or clip_by_whatever at all
    :param accumulate_grads_on: if specified, accumulate gradients on this device. By default, this will use the same
      device as model parameters. One can specify a different device (e.g. 'cpu' vs 'cuda') to save device memory at
      the cost of extra time per step. If reuse_grad_buffers is True, this parameter has no effect.
    :param client_mode: if False, this averager will accept incoming requests from other peers.
      if True, the averager will only join existing groups where at least one peer has client_mode=False.
      By default, this flag is copied from DHTNode inside the ``dht`` instance.
    :param warn: if True, warn when the averager did not reset accumulators after use or did not use averaging results
    :param min_compression_ratio: apply PowerEF to a tensor only if it reduces communication by at least this factor,
      otherwise aggregate tensors as is
    :param grad_averaging_interval: how often average gradients directly
    :param averaged_grads: if provided, it will be used as a set of averagable gradients
    """

    def __init__(
        self,
        parameters: Iterable[torch.nn.Parameter],
        averager_rank: int,
        *,
        dht: DHT,
        prefix: str,
        reuse_grad_buffers: bool = False,
        accumulate_grads_on: Optional[torch.device] = None,
        client_mode: bool = None,
        warn: bool = True,
        min_compression_ratio: float = 0.5,
        averaged_grads: Optional[Sequence[torch.Tensor]] = None,
        ms: Optional[Sequence[torch.Tensor]] = None,
        **kwargs,
    ):
        super().__init__(
            parameters=parameters,
            averager_rank=averager_rank,
            dht=dht,
            prefix=prefix,
            reuse_grad_buffers=reuse_grad_buffers,
            accumulate_grads_on=accumulate_grads_on,
            client_mode=client_mode,
            warn=warn,
            min_compression_ratio=min_compression_ratio,
            averaged_grads=averaged_grads,
            ms=ms,
            **kwargs,
        )

    async def _aggregate_with_group(self, group_info: GroupInfo, min_vector_size: int, **kwargs) -> GatheredData:
        """Run aggregation in a given group and update tensors in place, return gathered metadata"""
        try:
            bandwidths, mode_ids, user_gathered_bytes = zip(*map(self.serializer.loads, group_info.gathered))
            user_gathered = dict(zip(group_info.peer_ids, map(self.serializer.loads, user_gathered_bytes)))
            modes = tuple(map(AveragingMode, mode_ids))

            download_bandwidths = [
                thr if mode != AveragingMode.CLIENT else 0.0 for thr, mode in zip(bandwidths, modes)
            ]
            peer_fractions = await asyncio.get_event_loop().run_in_executor(
                None, load_balance_peers, self.total_size, download_bandwidths, min_vector_size
            )

            async with enter_asynchronously(self.get_tensors()) as averaged_grads:
                averaged_grads_via_sgd = [
                    grad for idx, grad in enumerate(averaged_grads) if idx not in self._uncompressed_gradients_indexes
                ]

                ps = [torch.zeros((get_flatten_greedy_dims(grad)[0], self.rank), device="cpu") for grad in averaged_grads_via_sgd]
                for p, q, m, grad in zip(ps, self._qs, self._ms, averaged_grads_via_sgd):
                    # we use reshape for all matrixes because PowerEF works only with 2d tensors
                    c = (grad - m).reshape(-1, q.size(0))
                    torch.matmul(c, q, out=p)

                p_group_id = group_info.group_id + AllReducePhases.PHASE_P.name.encode()
                q_groud_id = group_info.group_id + AllReducePhases.PHASE_Q.name.encode()

                await self._run_allreduce_inplace_(ps, group_info, p_group_id, peer_fractions=peer_fractions, **kwargs)

                for p in ps:
                    orthogonalize_(p)

                for p, q, m, grad in zip(ps, self._qs, self._ms, averaged_grads_via_sgd):
                    c = (grad - m).reshape(-1, q.size(0))
                    torch.matmul(c.t(), p, out=q)
                    grad.zero_()

                phase_q_tensors = self._qs + [
                    grad for idx, grad in enumerate(averaged_grads) if idx in self._uncompressed_gradients_indexes
                ]

                await self._run_allreduce_inplace_(
                    phase_q_tensors, group_info, q_groud_id, peer_fractions=peer_fractions, **kwargs
                )

                for p, q, m in zip(ps, self._qs, self._ms):
                    c = (p @ q.t()).reshape(m.shape)
                    torch.add(m, c, out=m)

                return user_gathered
        except BaseException as e:
            logger.exception(e)
            raise MatchmakingException(f"Unable to run All-Reduce: {e}")

    @contextlib.contextmanager
    @torch.no_grad()
    def use_averaged_gradients(self):
        """Substitute model's main gradients with averaged gradients (does not respect device placement)"""
        self._new_averaged_grads = False
        with self.get_tensors() as averaged_grads:
            assert len(averaged_grads) == len(self.parameters)
            try:
                old_grads = [param.grad for param in self.parameters]

                param_grads_old_way = [
                    param for idx, param in enumerate(self.parameters) if idx in self._uncompressed_gradients_indexes
                ]
                averaged_grads_old_way = [
                    grad for idx, grad in enumerate(averaged_grads) if idx in self._uncompressed_gradients_indexes
                ]
                for param, new_grad in zip(param_grads_old_way, averaged_grads_old_way):
                    param.grad = new_grad

                param_grads_via_sgd = [
                    param for idx, param in enumerate(self.parameters) if idx not in self._uncompressed_gradients_indexes
                ]
                for param, m in zip(param_grads_via_sgd, self._ms):
                    param.grad = m
                yield averaged_grads
            finally:
                for param, old_grad in zip(self.parameters, old_grads):
                    param.grad = old_grad
