# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import mii.grpc_related.proto.modelresponse_pb2 as modelresponse__pb2


class ModelResponseStub(object):
    """The greeting service definition.
    """
    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.StringReply = channel.unary_unary(
            '/modelresponse.ModelResponse/StringReply',
            request_serializer=modelresponse__pb2.RequestString.SerializeToString,
            response_deserializer=modelresponse__pb2.ReplyString.FromString,
        )


class ModelResponseServicer(object):
    """The greeting service definition.
    """
    def StringReply(self, request, context):
        """Sends a greeting
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_ModelResponseServicer_to_server(servicer, server):
    rpc_method_handlers = {
        'StringReply':
        grpc.unary_unary_rpc_method_handler(
            servicer.StringReply,
            request_deserializer=modelresponse__pb2.RequestString.FromString,
            response_serializer=modelresponse__pb2.ReplyString.SerializeToString,
        ),
    }
    generic_handler = grpc.method_handlers_generic_handler('modelresponse.ModelResponse',
                                                           rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler, ))


# This class is part of an EXPERIMENTAL API.
class ModelResponse(object):
    """The greeting service definition.
    """
    @staticmethod
    def StringReply(request,
                    target,
                    options=(),
                    channel_credentials=None,
                    call_credentials=None,
                    insecure=False,
                    compression=None,
                    wait_for_ready=None,
                    timeout=None,
                    metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/modelresponse.ModelResponse/StringReply',
            modelresponse__pb2.RequestString.SerializeToString,
            modelresponse__pb2.ReplyString.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata)
