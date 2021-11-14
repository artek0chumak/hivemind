import asyncio
import contextlib
import math
import torch
import multiprocessing as mp

from typing import Any, Iterable, Optional, Sequence

import hivemind
from hivemind.averaging.allreduce import AllreduceException, AllReduceRunner, AveragingMode, GroupID
from hivemind.averaging.control import AveragingStage, StepControl
from hivemind.averaging.group_info import GroupInfo
from hivemind.averaging.load_balancing import load_balance_peers
from hivemind.averaging.matchmaking import Matchmaking, MatchmakingException
from hivemind.averaging.partition import DEFAULT_PART_SIZE_BYTES
from hivemind.compression import (
    CompressionBase,
    CompressionInfo,
    NoCompression,
    deserialize_torch_tensor,
    serialize_torch_tensor,
)
from hivemind.dht import DHT, DHTID
from hivemind.p2p import P2P, P2PContext, P2PHandlerError, PeerID, ServicerBase
from hivemind.proto import averaging_pb2
from hivemind.utils import MPFuture, TensorDescriptor, get_logger
from hivemind.utils.asyncio import (
    achain,
    aiter_with_timeout,
    anext,
    as_aiter,
    azip,
    enter_asynchronously,
    switch_to_uvloop,
)
from hivemind.utils.grpc import combine_from_streaming, split_for_streaming
from hivemind.utils.serializer import MSGPackSerializer, SerializerBase
from hivemind.utils.timed_storage import DHTExpiration, ValueWithExpiration, get_dht_time

from .factorized_averager import FactorizedGradientAverager

GatheredData = Any
logger = get_logger(__name__)


class PowerSGDAverager(FactorizedGradientAverager):
    def __init__(
        self,
        parameters: Iterable[torch.nn.Parameter],
        rank: int,
        *,
        dht: hivemind.DHT,
        prefix: str,
        reuse_grad_buffers: bool = False,
        accumulate_grads_on: Optional[torch.device] = None,
        client_mode: bool = None,
        warn: bool = True,
        **kwargs,
    ):
        self._parameters = tuple(parameters)
        self._tensor_buffer = tuple(torch.zeros_like(grad, device=accumulate_grads_on) for grad in self._grads_from_parameters())
        self.lock_tensor_buffer = mp.Lock()
        for tensor in self._tensor_buffer:
            assert tensor.grad_fn is None, "averaged_tensors must be either parameters or leaf tensors"
            tensor.share_memory_()

        super().__init__(
            self._parameters,
            rank=rank,
            dht=dht,
            prefix=prefix,
            reuse_grad_buffers=reuse_grad_buffers,
            accumulate_grads_on=accumulate_grads_on,
            client_mode=client_mode,
            warn=warn,
            **kwargs
        )

    @contextlib.contextmanager
    def get_buffer_tensors(self) -> Sequence[torch.Tensor]:
        with self.lock_tensor_buffer:
            yield self._tensor_buffer

    async def _step(self, *, step: StepControl, future_for_trigger: MPFuture):
        try:
            trigger = MPFuture()
            step.attach_trigger(trigger)
            future_for_trigger.set_result(trigger)

            while not step.done():
                try:
                    self._pending_group_assembled.clear()
                    step.stage = AveragingStage.LOOKING_FOR_GROUP
                    group_info = await self._matchmaking.look_for_group(step)
                    if group_info is None:
                        raise AllreduceException("Averaging step failed: could not find a group.")

                    if not step.triggered:
                        step.stage = AveragingStage.AWAITING_TRIGGER
                        await step.wait_for_trigger()

                    step.stage = AveragingStage.RUNNING_ALLREDUCE

                    with self.get_subsets_tensors() as averaged_PQ, self.get_tensors() as averaged_grad, self.get_buffer_tensors() as buffers:
                        for grad_acc, buffer, (av_p, av_q) in zip(averaged_grad, buffers, zip(*averaged_PQ)):
                            buffer.add_(grad_acc)
                            if len(tuple(buffer.size())) > 1:
                                p = buffer.reshape((av_p.size(0), av_q.size(0))) @ av_q
                                av_p.copy_(p)
                            else:
                                av_p.copy_(grad_acc.reshape(av_p.size()))

                    # P averaging
                    p_result = await asyncio.wait_for(
                        self._run_allreduce_subsets(
                            0, False, group_info, tensor_infos=self.tensor_infos, weight=step.weight, **self.allreduce_kwargs
                        ),
                        timeout=self._allreduce_timeout,
                    )
                    with self.get_subsets_tensors() as averaged_PQ, self.get_buffer_tensors() as buffers:
                        for buffer, (av_p, av_q) in zip(buffers, zip(*averaged_PQ)):
                            if len(tuple(buffer.size())) > 1:
                                orthogonalize(av_p)
                                q = buffer.reshape((av_p.size(0), av_q.size(0))).T @ av_p
                                av_q.copy_(q)
                            else:
                                av_q.copy_(torch.eye(1))

                    # Q averaging
                    q_result = await asyncio.wait_for(
                        self._run_allreduce_subsets(
                            1, True, group_info, tensor_infos=self.tensor_infos, weight=step.weight, **self.allreduce_kwargs
                        ),
                        timeout=self._allreduce_timeout,
                    )
                    with self.get_subsets_tensors() as averaged_PQ, self.get_buffer_tensors() as buffers:
                        for buffer, (av_p, av_q) in zip(buffers, zip(*averaged_PQ)):
                            if len(tuple(buffer.size())) > 1:
                                new_buffer = av_p @ av_q.t()
                                buffer.copy_(buffer - new_buffer.reshape(buffer.size()))

                    step.set_result(q_result)
                    # averaging is finished, loop will now exit

                except (
                    AllreduceException,
                    MatchmakingException,
                    AssertionError,
                    StopAsyncIteration,
                    asyncio.CancelledError,
                    asyncio.InvalidStateError,
                    P2PHandlerError,
                ) as e:
                    if not step.allow_retries or get_dht_time() >= step.deadline:
                        logger.exception(e)
                        step.set_exception(e)
                    else:
                        logger.warning(f"{self.__class__.__name__} caught {repr(e)}, retrying")

        except BaseException as e:
            if not step.done():
                step.set_exception(e)
            raise
        finally:
            step.stage = AveragingStage.FINISHED
            if not step.done():
                step.set_exception(
                    RuntimeError(
                        "Internal sanity check failed: averager.step left future pending."
                        " Please report this to hivemind issues."
                    )
                )


@torch.jit.script
def orthogonalize(matrix, eps=torch.tensor(1e-8)):
    n, m = matrix.shape
    for i in range(m):
        col = matrix[:, i: i + 1]
        col /= torch.sqrt(torch.sum(col ** 2)) + eps
        if i + 1 < m:
            rest = matrix[:, i + 1:]
            rest -= torch.sum(col * rest, dim=0) * col
