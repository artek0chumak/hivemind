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

from .grad_averager import GradientAverager

GatheredData = Any
logger = get_logger(__name__)


class FactorizedGradientAverager(GradientAverager):
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
        self._averaged_tensors_subsets = [
            tuple(
                # if grad.size() == 1 => p = grad, q = 1
                # if grad.size() > 1 => p, q: grad.reshape() = p @ q.T
                torch.zeros(
                    (grad.reshape((grad.size(0), -1)).size(idx), rank if len(tuple(grad.size())) > 1 else 1),
                    device=accumulate_grads_on
                )
                for grad in self._grads_from_parameters()
            )
            for idx in range(2)
        ]
        for subsets in self._averaged_tensors_subsets:
            for tensor in subsets:
                assert tensor.grad_fn is None, "averaged_tensors must be either parameters or leaf tensors"
                tensor.share_memory_()
        self.lock_averaged_tensors_subsets = mp.Lock()
        self.rank = rank

        super().__init__(
            self._parameters,
            dht=dht,
            prefix=prefix,
            reuse_grad_buffers=reuse_grad_buffers,
            accumulate_grads_on=accumulate_grads_on,
            client_mode=client_mode,
            warn=warn,
            **kwargs
        )

    @contextlib.contextmanager
    def get_subsets_tensors(self) -> Sequence[torch.Tensor]:
        with self.lock_averaged_tensors_subsets:
            yield self._averaged_tensors_subsets

    async def _run_allreduce_subsets(self, subset_idx: int, update_state: bool, group_info: GroupInfo, min_vector_size: int, **kwargs) -> GatheredData:
        try:
            bandwidths, mode_ids, user_gathered_bytes = zip(*map(self.serializer.loads, group_info.gathered))
            user_gathered = dict(zip(group_info.peer_ids, map(self.serializer.loads, user_gathered_bytes)))
            modes = tuple(map(AveragingMode, mode_ids))

            # compute optimal part sizes from peer bandwidths; TODO: replace with proper load balancing
            download_bandwidths = [
                thr if mode != AveragingMode.CLIENT else 0.0 for thr, mode in zip(bandwidths, modes)
            ]
            peer_fractions = await asyncio.get_event_loop().run_in_executor(
                None, load_balance_peers, self.total_size, download_bandwidths, min_vector_size
            )

            async with enter_asynchronously(self.get_subsets_tensors()) as subsets_local_tensors:
                local_tensors  = subsets_local_tensors[subset_idx]
                allreduce = AllReduceRunner(
                    p2p=self._p2p,
                    servicer_type=type(self),
                    prefix=self.prefix,
                    group_id=group_info.group_id,
                    tensors=local_tensors,
                    ordered_peer_ids=group_info.peer_ids,
                    peer_fractions=peer_fractions,
                    gathered=user_gathered,
                    modes=modes,
                    **kwargs,
                )

                with self.register_allreduce_group(group_info.group_id, allreduce):
                    if modes[group_info.peer_ids.index(self.peer_id)] != AveragingMode.AUX:
                        async for tensor, update in azip(as_aiter(*local_tensors), allreduce):
                            # all-reduce is performed asynchronously while iterating
                            tensor.add_(update, alpha=self._averaging_alpha)
                            if update_state:
                                self.last_updated = get_dht_time()
                                self._state_updated.set()
                    else:
                        async for _ in allreduce:  # trigger all-reduce by iterating
                            raise ValueError("aux peers should not receive averaged tensors")

                return allreduce.gathered
        except BaseException as e:
            logger.exception(e)
            raise MatchmakingException(f"Unable to run All-Reduce: {e}")

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

                    with self.get_subsets_tensors() as averaged_PQ, self.get_tensors() as averaged_grad:
                        averaged_P, averaged_Q = averaged_PQ
                        for grad_acc, av_p, av_q in zip(averaged_grad, averaged_P, averaged_Q):
                            try:
                                if len(tuple(grad_acc.size())) > 1:
                                    grad_acc = grad_acc.reshape(av_p.size(0), av_q.size(0))
                                    u, s, v = torch.linalg.svd(grad_acc)
                                    p = (u @ torch.diag(s))[:, :self.rank]
                                    q = v.t()[:, :self.rank]
                                    assert (p @ q.t()).size() == grad_acc.size()
                                else:
                                    p = grad_acc.reshape(av_p.size())
                                    q = torch.eye(1)
                                av_p.copy_(p)
                                av_q.copy_(q)
                            except:
                                av_p.copy_(torch.zeros(p.size()))
                                av_q.copy_(torch.zeros(q.size()))

                    # P averaging
                    p_result = await asyncio.wait_for(
                        self._run_allreduce_subsets(
                            0, False, group_info, tensor_infos=self.tensor_infos, weight=step.weight, **self.allreduce_kwargs
                        ),
                        timeout=self._allreduce_timeout,
                    )
                    # Q averaging
                    q_result = await asyncio.wait_for(
                        self._run_allreduce_subsets(
                            1, True, group_info, tensor_infos=self.tensor_infos, weight=step.weight, **self.allreduce_kwargs
                        ),
                        timeout=self._allreduce_timeout,
                    )

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

    @contextlib.contextmanager
    @torch.no_grad()
    def use_averaged_gradients(self):
        self._new_averaged_grads = False
        with self.get_subsets_tensors() as averaged_PQ, self.get_tensors() as averaged_tensors:
            try:
                averaged_P, averaged_Q = averaged_PQ
                assert len(averaged_P) == len(self._parameters), f"averaged_P: {len(averaged_P)}, parameters: {len(self._parameters)}"
                assert len(averaged_Q) == len(self._parameters), f"averaged_Q: {len(averaged_Q)}, parameters: {len(self._parameters)}"
                old_grads = [param.grad for param in self._parameters]
                for param, p, q in zip(self._parameters, averaged_P, averaged_Q):
                    new_grad = p @ q.t()
                    param.grad = new_grad.reshape(param.size())
                yield
            finally:
                for param, old_grad in zip(self._parameters, old_grads):
                    param.grad = old_grad
