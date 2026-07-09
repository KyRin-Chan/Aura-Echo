import msgspec
from socketio import packet


class MsgPackPacket(packet.Packet):
    uses_binary_events = False

    def encode(self):
        """Encode the packet for transmission."""
        return msgspec.msgpack.encode(self._to_dict())

    def decode(self, encoded_packet):
        """Decode a transmitted package."""
        if isinstance(encoded_packet, str):
            return super().decode(encoded_packet)

        try:
            decoded = msgspec.msgpack.decode(encoded_packet)
            self.packet_type = decoded['type']
            self.data = decoded.get('data')
            self.id = decoded.get('id')
            self.namespace = decoded['nsp']
        except Exception:
            if isinstance(encoded_packet, bytes):
                try:
                    return super().decode(encoded_packet.decode('utf-8'))
                except Exception:
                    pass
            raise
