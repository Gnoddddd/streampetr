from .partial_observation import ProtocolEvent


def motion_blur_event(camera: str, start_frame: int, duration: int, severity: float) -> ProtocolEvent:
    return ProtocolEvent(start_frame, start_frame + duration - 1, motion_blur={camera: severity})
