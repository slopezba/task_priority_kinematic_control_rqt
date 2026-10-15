"""Marker client ROS I/O, independent of the panel's Qt-driven ROS spinner."""
from concurrent.futures import Future
from queue import Empty, Queue
from threading import Event, Thread
import uuid

from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy


class MarkerTransport:
    """Own a node and executor; callbacks must marshal UI work to the Qt thread."""

    def __init__(self, context, api, state_type, endpoint, on_state):
        self._stop = Event()
        self._commands = Queue()
        self._pending = set()
        self._node = Node('planning_target_client_' + uuid.uuid4().hex,
                          context=context, use_global_arguments=False)
        self._client = self._node.create_client(api, endpoint)
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                         reliability=ReliabilityPolicy.RELIABLE)
        self._subscription = self._node.create_subscription(
            state_type, endpoint[:-len('/update')] + '/state', on_state, qos)
        self._guard = self._node.create_guard_condition(self._drain)
        self._executor = SingleThreadedExecutor(context=context)
        self._executor.add_node(self._node)
        self._thread = Thread(target=self._run, name='planning-target-client', daemon=True)
        self._thread.start()

    def service_is_ready(self):
        return not self._stop.is_set() and self._client.service_is_ready()

    def call_async(self, request):
        future = Future()
        if self._stop.is_set():
            future.set_exception(RuntimeError('Planning target client is closed'))
            return future
        self._commands.put((request, future))
        self._guard.trigger()
        return future

    def _drain(self):
        while not self._stop.is_set():
            try:
                request, future = self._commands.get_nowait()
            except Empty:
                return
            self._pending.add(future)
            try:
                ros_future = self._client.call_async(request)
                ros_future.add_done_callback(lambda reply, result=future: self._complete(reply, result))
            except Exception as error:
                self._pending.discard(future)
                future.set_exception(error)

    def _complete(self, reply, future):
        self._pending.discard(future)
        if future.done():
            return
        try:
            future.set_result(reply.result())
        except Exception as error:
            future.set_exception(error)

    def _run(self):
        try:
            while not self._stop.is_set() and self._node.context.ok():
                self._executor.spin_once(timeout_sec=0.1)
        except ExternalShutdownException:
            pass
        finally:
            self._stop.set()
            while True:
                try:
                    _, future = self._commands.get_nowait()
                    self._pending.add(future)
                except Empty:
                    break
            for future in self._pending:
                if not future.done():
                    future.set_exception(RuntimeError('Planning target client stopped'))
            self._pending.clear()
            self._executor.shutdown()
            self._node.destroy_node()

    def close(self):
        self._stop.set()
        self._executor.wake()
        self._thread.join(timeout=1.0)
