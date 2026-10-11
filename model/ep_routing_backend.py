"""Attach MiniMind EP routing using the installed operator package."""
def attach_ep_routing(model, *, library=None, mode="compact", decode_capacity_tokens=32):
    from .ep.routing import MiniMindEPRouting
    return MiniMindEPRouting(model, library, mode, decode_capacity_tokens=decode_capacity_tokens)
