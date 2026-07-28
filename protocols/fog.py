from .partial_observation import ProtocolEvent


def fog_event(camera: str, start_frame: int, duration: int, severity: float) -> ProtocolEvent:
    return ProtocolEvent(start_frame, start_frame + duration - 1, fog={camera: severity})
