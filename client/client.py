import argparse
import asyncio
import threading
from typing import Callable, Iterable

from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_bundle_builder import OscBundleBuilder
from pythonosc.osc_server import AsyncIOOSCUDPServer, ThreadingOSCUDPServer
from pythonosc.udp_client import OscBundle, OscMessageBuilder, SimpleUDPClient

REMOTE_PORT = 11000
LOCAL_PORT = 11001

# --------------------------------------------------------------------------------
# An Ableton Live tick is 100ms. This constant is typically used for timeouts,
# and factors in some extra time for processing overhead.
# --------------------------------------------------------------------------------
TICK_DURATION = 0.150


class AbletonOSCClient:
    def __init__(self, hostname="127.0.0.1", port=REMOTE_PORT, client_port=LOCAL_PORT):
        """
        Create a client to connect to an Ableton OSC instance.
        Args:
            hostname: The remote host to connect to.
            port: The remote port to connect to. Defaults to 11000, the default AbletonOSC port.
            client_port: The local port to bind to. Defaults to 11001, the default AbletonOSC reply port.
        """
        dispatcher = Dispatcher()
        dispatcher.set_default_handler(self.handle_osc)
        self.server = ThreadingOSCUDPServer(("0.0.0.0", client_port), dispatcher)
        self.server_thread = threading.Thread(target=self.server.serve_forever)
        self.server_thread.daemon = True
        self.server_thread.start()
        self.address_handlers = {}
        self.client = SimpleUDPClient(hostname, port)
        self.verbose = False

    def handle_osc(self, address, *params):
        # print("Received OSC: %s %s" % (address, params))
        if address in self.address_handlers:
            self.address_handlers[address](address, params)
        if self.verbose:
            print(address, params)

    def stop(self):
        self.server.shutdown()
        self.server_thread.join()
        self.server = None

    def send_bundle(self, messages: list[tuple[str, tuple]]):
        import time

        now = int(time.time())
        bundle_builder = OscBundleBuilder(now)
        for address, params in messages:
            builder = OscMessageBuilder(address=address)
            for param in params:
                builder.add_arg(param)
            msg = builder.build()
            bundle_builder.add_content(msg)
        bundle = bundle_builder.build()
        self.client.send(bundle)

    def send_message(self, address: str, params: Iterable = ()):
        """
        Send a message to the given OSC address on the server.

        Args:
            address (str): The OSC address to send to (e.g. /live/song/set/tempo)
            params (Iterable): Optional list of arguments to pass to the OSC message.
        """
        self.client.send_message(address, params)

    def set_handler(self, address: str, fn: Callable = None):
        """
        Set the handler for the specified OSC message.

        Args:
            address (str): The OSC address to listen for (e.g. /live/song/get/tempo)
            fn (Callable): The function to trigger when a message received.
                           Must accept a two arguments:
                            - str: the OSC address
                            - tuple: the OSC parameters
        """
        self.address_handlers[address] = fn

    def remove_handler(self, address: str):
        """
        Remove the handler for the specified OSC message.

        Args:
            address (str): The OSC address whose handler to remove.
        """
        del self.address_handlers[address]

    def await_message(self, address: str, timeout: float = TICK_DURATION):
        """
        Awaits a reply from the given `address`, and optionally asserts that the function `fn`
        returns True when called with the returned OSC parameters.

        Args:
            address: OSC query (and reply) address
            fn: Optional assertion function
            timeout: Maximum number of seconds to wait for a successful reply

        Returns:
            True if the reply is received within the timeout period and the assertion succeeds,
            False otherwise

        """
        rv = None
        _event = threading.Event()

        def received_response(address, params):
            print("Received response: %s %s" % (address, str(params)))
            nonlocal rv
            nonlocal _event
            rv = params
            _event.set()

        self.set_handler(address, received_response)
        _event.wait(timeout)
        self.remove_handler(address)
        if not _event.is_set():
            raise RuntimeError("No response received to query: %s" % address)
        return rv

    def query(self, address: str, params: tuple = (), timeout: float = TICK_DURATION):
        rv = None
        _event = threading.Event()

        def received_response(address, params):
            nonlocal rv
            nonlocal _event
            rv = params
            _event.set()

        self.set_handler(address, received_response)
        self.send_message(address, params)
        _event.wait(timeout)
        self.remove_handler(address)
        if not _event.is_set():
            raise RuntimeError("No response received to query: %s" % address)
        return rv


class AsyncAbletonOSCClient:
    """Async AbletonOSC client using asyncio's event loop for UDP listening.

    Unlike AbletonOSCClient, this uses AsyncIOOSCUDPServer so all OSC message
    handling runs directly on the asyncio event loop thread. Benefits:
    - No thread synchronization or locks needed
    - asyncio.Event can be used directly (no call_soon_threadsafe)
    - query() is a proper coroutine that does not block the event loop
    - Responses are matched to queries by param-prefix, allowing concurrent
      queries to the same OSC address with different parameters

    Must call await start() before using query().
    """

    def __init__(
        self,
        hostname: str = "127.0.0.1",
        port: int = REMOTE_PORT,
        client_port: int = LOCAL_PORT,
    ) -> None:
        self._client_port = client_port
        self.client = SimpleUDPClient(hostname, port)
        self.address_handlers: dict[str, Callable] = {}
        # Keyed by (address, query_params_tuple); value is (event, result_holder)
        self.query_handlers: dict[tuple[str, tuple], tuple[asyncio.Event, list]] = {}
        self._transport: asyncio.BaseTransport | None = None
        self.verbose = False

    async def start(self) -> None:
        """Initialize the asyncio UDP server. Safe to call multiple times."""
        if self._transport is not None:
            return
        dispatcher = Dispatcher()
        dispatcher.set_default_handler(self.handle_osc)
        loop = asyncio.get_running_loop()
        server = AsyncIOOSCUDPServer(("0.0.0.0", self._client_port), dispatcher, loop)
        self._transport, _ = await server.create_serve_endpoint()

    def stop(self) -> None:
        """Close the UDP transport."""
        if self._transport is not None:
            self._transport.close()
            self._transport = None

    # Song-level bulk-query endpoints whose responses don't echo back the query params.
    # For these, match by address only. All other endpoints echo identifying params
    # (track_id, device_id, etc.) as a prefix, so they use param-prefix matching.
    _ADDRESS_ONLY_ENDPOINTS = {"/live/song/get/track_data"}

    def handle_osc(self, address: str, *params) -> None:
        """Dispatch an incoming OSC message. Always runs on the event loop thread.

        Checks pending query handlers first. Most endpoints echo the query params back as
        a prefix in the response (e.g. track_id, device_id), so concurrent queries to the
        same address can be distinguished. Song bulk-query endpoints like
        /live/song/get/track_data don't echo params back, so those match by address only.

        Falls through to push-subscription handlers (address_handlers) afterward.
        """
        print("received OSC response at:", address)
        # print("received params:", params)
        for (addr, query_params), (event, rv) in self.query_handlers.items():
            if addr == address:
                # print("address in query_handlers dict:", addr)
                # print("query params in dict:", query_params)
                # print("address to handle:", address)
                # print("query params to handle:", params)
                if addr not in self._ADDRESS_ONLY_ENDPOINTS:
                    n = len(query_params)
                    if params[:n] != query_params:
                        continue
                rv[0] = params
                event.set()
                break

        if address in self.address_handlers:
            self.address_handlers[address](address, params)

    async def query(
        self, address: str, params: tuple | list = (), timeout: float = 5.0
    ) -> tuple:
        """Send an OSC message and await the matching response.

        Concurrent calls with different params are safe and will resolve
        independently via param-prefix matching in handle_osc.
        """
        event = asyncio.Event()
        rv: list = [None]
        key = (address, tuple(params))
        self.query_handlers[key] = (event, rv)
        self.send_message(address, params)
        print("waiting for response to:", key)
        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            raise RuntimeError(f"No response received to query: {address}")
        finally:
            self.query_handlers.pop(key, None)
        return rv[0]

    def send_message(self, address: str, params: Iterable = ()) -> None:
        self.client.send_message(address, params)

    def set_handler(self, address: str, fn: Callable) -> None:
        self.address_handlers[address] = fn

    def remove_handler(self, address: str) -> None:
        del self.address_handlers[address]


def main(args):
    client = AbletonOSCClient(args.hostname, args.port)
    client.send_message("/live/song/set/tempo", [125.0])
    tempo = client.query("/live/song/get/tempo")
    print("Got song tempo: %.1f" % tempo[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Client for AbletonOSC")
    parser.add_argument("--hostname", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=str, default=11000)
    args = parser.parse_args()
    main(args)
