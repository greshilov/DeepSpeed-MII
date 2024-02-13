# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import asyncio
import grpc
import requests
from typing import Awaitable, AsyncGenerator, Dict, Any, Callable, List, Union

from mii.batching.data_classes import Response
from mii.config import MIIConfig
from mii.constants import GRPC_MAX_MSG_SIZE
from mii.grpc_related.proto import modelresponse_pb2, modelresponse_pb2_grpc
from mii.grpc_related.task_methods import TASK_METHODS_DICT

MiiClientResponse = Union[None, List[Response]]


def create_channel(host, port):
    return grpc.aio.insecure_channel(
        f"{host}:{port}",
        options=[
            ("grpc.max_send_message_length",
             GRPC_MAX_MSG_SIZE),
            ("grpc.max_receive_message_length",
             GRPC_MAX_MSG_SIZE),
        ],
    )


class MIIClient:
    """
    Client to send queries to a single endpoint.
    """
    def __init__(self, mii_config: MIIConfig, host: str = "localhost") -> None:
        self.mii_config = mii_config
        self.task = mii_config.model_config.task
        self.port = mii_config.port_number
        self.asyncio_loop = asyncio.get_event_loop()
        channel = create_channel(host, self.port)
        self.stub = modelresponse_pb2_grpc.ModelResponseStub(channel)

    def __call__(self, *args, **kwargs) -> List[Response]:
        return self.generate(*args, **kwargs)

    async def _request_async_response(
            self,
            prompts: Union[str,
                           List[str]],
            **query_kwargs: Dict[str,
                                 Any]) -> Awaitable[MiiClientResponse]:
        task_methods = TASK_METHODS_DICT[self.task]
        proto_request = task_methods.pack_request_to_proto(prompts, **query_kwargs)

        proto_response = await getattr(self.stub, task_methods.method)(proto_request)
        return task_methods.unpack_response_from_proto(proto_response)

    async def _request_async_response_stream(
            self,
            prompts: Union[str,
                           List[str]],
            **query_kwargs: Dict[str,
                                 Any]) -> AsyncGenerator[MiiClientResponse,
                                                         None]:

        if len(prompts) > 1:
            raise RuntimeError(
                "MII client streaming only supports a single prompt input.")

        task_methods = TASK_METHODS_DICT[self.task]
        proto_request = task_methods.pack_request_to_proto(prompts, **query_kwargs)

        assert hasattr(task_methods, "method_stream_out"), f"{self.task} does not support streaming response"
        async for response in getattr(self.stub,
                                      task_methods.method_stream_out)(proto_request):
            yield task_methods.unpack_response_from_proto(response)

    def agenerate(
        self,
        prompts: Union[str,
                       List[str]],
        **query_kwargs: Dict[str,
                             Any]
    ) -> Union[Awaitable[MiiClientResponse],
               AsyncGenerator[MiiClientResponse,
                              None]]:

        if isinstance(prompts, str):
            prompts = [prompts]

        if query_kwargs.get("stream", False):
            return self._request_async_response_stream(prompts, **query_kwargs)

        return self._request_async_response(prompts, **query_kwargs)

    def generate(self,
                 prompts: Union[str,
                                List[str]],
                 streaming_fn: Callable = None,
                 **query_kwargs: Dict[str,
                                      Any]) -> MiiClientResponse:
        if streaming_fn is not None:
            query_kwargs["stream"] = True

            async def read_stream():
                async for response in self.agenerate(prompts, **query_kwargs):
                    streaming_fn(response)

            return self.asyncio_loop.run_until_complete(read_stream())

        return self.asyncio_loop.run_until_complete(
            self.agenerate(prompts,
                           **query_kwargs))

    async def terminate_async(self) -> None:
        await self.stub.Terminate(
            modelresponse_pb2.google_dot_protobuf_dot_empty__pb2.Empty())

    def terminate_server(self) -> None:
        self.asyncio_loop.run_until_complete(self.terminate_async())
        if self.mii_config.enable_restful_api:
            requests.get(
                f"http://localhost:{self.mii_config.restful_api_port}/terminate")
