"""Test doubles for the Anthropic client — script exactly what 'Claude' does
so the agent loop can be tested deterministically and offline."""

from types import SimpleNamespace


def text(t):
    return SimpleNamespace(type="text", text=t)


def tool_use(name, input_, id_=None):
    tool_use.counter += 1
    return SimpleNamespace(type="tool_use", name=name, input=input_, id=id_ or f"toolu_{tool_use.counter}")


tool_use.counter = 0


def response(*blocks, in_tok=100, out_tok=20):
    stop = "tool_use" if any(b.type == "tool_use" for b in blocks) else "end_turn"
    return SimpleNamespace(
        stop_reason=stop,
        content=list(blocks),
        usage=SimpleNamespace(input_tokens=in_tok, output_tokens=out_tok),
    )


class ScriptedClient:
    """Returns pre-written responses in order and records every request."""

    def __init__(self, *responses):
        self._responses = list(responses)
        self.requests = []
        self.messages = self

    def create(self, **kwargs):
        # Snapshot messages: the agent mutates the list after the call.
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})
        if not self._responses:
            raise AssertionError("agent made more model calls than the script allows")
        return self._responses.pop(0)

    def offered_tools(self, i=0):
        return {t["name"] for t in self.requests[i]["tools"]}
