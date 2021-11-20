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


class PowerEFGradientAverager(GradientAverager):
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
        self.rank = rank
        self._parameters = tuple(parameters)
        self._uncompressed_gradients = set(i for i, grad in enumerate(self._grads_from_parameters()) if len(tuple(grad.size())) == 1)
        self._gs = list(
            torch.zeros_like(grad, device=accumulate_grads_on)
            for idx, grad in enumerate(self._grads_from_parameters()) if idx not in self._uncompressed_gradients
        )
        self._qs = list(
            torch.rand((grad.reshape((grad.size(0), -1)).size(1), self.rank), device=accumulate_grads_on)
            for idx, grad in enumerate(self._grads_from_parameters()) if idx not in self._uncompressed_gradients
        )
        for tensor in self._qs:
            if tensor is not None:
                assert tensor.grad_fn is None, "averaged_tensors must be either parameters or leaf tensors"
                tensor.share_memory_()
        

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

    async def _run_allreduce(self, group_info: GroupInfo, min_vector_size: int, **kwargs) -> GatheredData:
        """Run All-Reduce in a given group and update tensors in place, return gathered metadata"""
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

            async with enter_asynchronously(self.get_tensors()) as local_tensors:
                compressed_tensors = [lt for idx, lt in enumerate(local_tensors) if idx not in self._uncompressed_gradients]
                cs = [torch.zeros_like(grad, device="cpu") for grad in compressed_tensors]
                for c, g, cg in zip(cs, self._gs, compressed_tensors):
                    torch.sub(cg, g, out=c)

                ps = [torch.zeros((grad.size(0), self.rank), device="cpu") for grad in compressed_tensors]
                for p, q, c in zip(ps, self._qs, cs):
                    torch.matmul(c.reshape(-1, q.size(0)), q, out=p)
                first_all_reduced = ps + [lt for idx, lt in enumerate(local_tensors) if idx in self._uncompressed_gradients]
                allreduce = AllReduceRunner(
                    p2p=self._p2p,
                    servicer_type=type(self),
                    prefix=self.prefix,
                    group_id=group_info.group_id,
                    tensors=first_all_reduced,
                    ordered_peer_ids=group_info.peer_ids,
                    peer_fractions=peer_fractions,
                    gathered=user_gathered,
                    modes=modes,
                    **kwargs,
                )

                with self.register_allreduce_group(group_info.group_id, allreduce):
                    if modes[group_info.peer_ids.index(self.peer_id)] != AveragingMode.AUX:
                        async for tensor, update in azip(as_aiter(*first_all_reduced), allreduce):
                            # all-reduce is performed asynchronously while iterating
                            tensor.add_(update, alpha=self._averaging_alpha)
                    else:
                        async for _ in allreduce:  # trigger all-reduce by iterating
                            raise ValueError("aux peers should not receive averaged tensors")

                # orth ps
                for p in ps:
                    orthogonalize(p)

                # compute qs
                for p, q, c in zip(ps, self._qs, cs):
                    torch.matmul(c.reshape(-1, q.size(0)).t(), p, out=q)

                allreduce = AllReduceRunner(
                    p2p=self._p2p,
                    servicer_type=type(self),
                    prefix=self.prefix,
                    group_id=group_info.group_id,
                    tensors=self._qs,
                    ordered_peer_ids=group_info.peer_ids,
                    peer_fractions=peer_fractions,
                    gathered=user_gathered,
                    modes=modes,
                    **kwargs,
                )

                with self.register_allreduce_group(group_info.group_id, allreduce):
                    if modes[group_info.peer_ids.index(self.peer_id)] != AveragingMode.AUX:
                        async for tensor, update in azip(as_aiter(*self._qs), allreduce):
                            # all-reduce is performed asynchronously while iterating
                            tensor.add_(update, alpha=self._averaging_alpha)
                            self.last_updated = get_dht_time()
                            self._state_updated.set()
                    else:
                        async for _ in allreduce:  # trigger all-reduce by iterating
                            raise ValueError("aux peers should not receive averaged tensors")

                # recompute grads
                for p, q, c in zip(ps, self._qs, cs):
                    new_c = torch.matmul(p, q.t())
                    c.copy_(new_c.reshape(c.size()))

                for c, g, cg in zip(cs, self._gs, compressed_tensors):
                    torch.add(g, c, out=g)
                    cg.copy_(g)

                return allreduce.gathered
        except BaseException as e:
            logger.exception(e)
            raise MatchmakingException(f"Unable to run All-Reduce: {e}")


@torch.jit.script
def orthogonalize(matrix, eps=torch.tensor(1e-8)):
    n, m = matrix.shape
    for i in range(m):
        col = matrix[:, i: i + 1]
        col /= torch.sqrt(torch.sum(col ** 2)) + eps
        if i + 1 < m:
            rest = matrix[:, i + 1:]
            rest -= torch.sum(col * rest, dim=0) * col
