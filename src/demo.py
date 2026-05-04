"""Command-line FunctionGemma physical-AI demo for the SL2619 Coral Dev Board.

Usage:
    python3 demo.py --prompt "Turn the lights red and beep twice"

By default the WLED Neopixel ring is OFF. Pass ``--wled-port /dev/ttyACM0``
to drive an Adafruit Mini Sparkle Motion + WS2812B ring over USB-CDC.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dispatcher import Dispatcher
from hardware import HardwareDevice
from llamacpp import FunctionGemmaModel
from wled import WLEDSerialClient


DEFAULT_MODEL = (
    Path(__file__).resolve().parent.parent
    / "models" / "functiongemma-physical-ai-Q4_K_M.gguf"
)


def main() -> None:
    p = argparse.ArgumentParser(
        description="FunctionGemma physical-AI demo on SL2619",
    )
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL,
                   help=f"Path to the GGUF model. Default: {DEFAULT_MODEL}")
    p.add_argument("--prompt", required=True,
                   help="Natural-language prompt to send to the model")
    p.add_argument(
        "--wled-port",
        help="USB-CDC port for the Mini Sparkle Motion (e.g. /dev/ttyACM0). "
             "Omit to run without the Neopixel ring.",
    )
    p.add_argument("--wled-baud", type=int, default=115200,
                   help="WLED serial baud rate (default 115200)")
    args = p.parse_args()

    print(f"[1/3] Loading model from {args.model}")
    model = FunctionGemmaModel(str(args.model))

    wled = WLEDSerialClient(port=args.wled_port, baud=args.wled_baud) \
        if args.wled_port else None
    dispatcher = Dispatcher(HardwareDevice(wled=wled))

    print(f"[2/3] Running inference on prompt: {args.prompt!r}")
    result = model.generate(args.prompt)
    names = [c.name for c in result.tool_calls]
    print(f"  -> {len(result.tool_calls)} tool call(s) in {result.latency_ms:.0f} ms: {names}")

    print(f"[3/3] Dispatching to hardware")
    for call_result in dispatcher.dispatch_all(result.tool_calls):
        print(f"  - {call_result}")


if __name__ == "__main__":
    main()
