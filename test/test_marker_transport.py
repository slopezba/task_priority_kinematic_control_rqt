"""Real DDS marker-client traffic must progress without the panel spinner."""
import os
import tempfile
from threading import Event, Thread
import unittest
from unittest.mock import patch
import uuid

import rclpy
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from std_srvs.srv import Trigger

from task_priority_kinematic_control_rqt.marker_transport import MarkerTransport


class MarkerTransportIntegrationTest(unittest.TestCase):
    def test_replies_and_states_progress_without_spinning_the_panel_node(self):
        # Standard wire types exercise transport without requiring interface builds.
        with tempfile.TemporaryDirectory(prefix='marker_client_ros_') as log_dir, patch.dict(os.environ, {
                'ROS_DOMAIN_ID': str(150 + os.getpid() % 50), 'ROS_LOCALHOST_ONLY': '1',
                'ROS_LOG_DIR': log_dir}):
            context = Context()
            rclpy.init(args=[], context=context)
            server = Node('marker_transport_test_server', context=context)
            panel = Node('marker_transport_unspun_panel', context=context)
            executor = SingleThreadedExecutor(context=context)
            executor.add_node(server)
            namespace = '/transport_test_' + uuid.uuid4().hex
            def respond(request, response):
                response.success, response.message = True, 'target acknowledged'
                return response
            server.create_service(Trigger, namespace + '/update', respond)
            qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
            publisher = server.create_publisher(String, namespace + '/state', qos)
            publisher.publish(String(data='confirmed target'))
            received = Event()
            states = []
            def on_state(state):
                states.append(state.data)
                received.set()
            stop = Event()
            def spin_server():
                while not stop.is_set():
                    executor.spin_once(timeout_sec=0.1)
            worker = Thread(target=spin_server)
            worker.start()
            transport = None
            try:
                transport = MarkerTransport(panel.context, Trigger, String, namespace + '/update', on_state)
                self.assertTrue(received.wait(4.0), 'Transient target state was not received')
                self.assertEqual(states[-1], 'confirmed target')
                self.assertTrue(transport.service_is_ready())
                for _ in range(3):
                    # Neither Qt events nor the panel's ROS executor run here.
                    response = transport.call_async(Trigger.Request()).result(timeout=3.0)
                    self.assertTrue(response.success)
                    self.assertEqual(response.message, 'target acknowledged')
                transport.close()
                self.assertFalse(transport._thread.is_alive())
                self.assertIsNone(panel.executor)
                with self.assertRaisesRegex(RuntimeError, 'closed'):
                    transport.call_async(Trigger.Request()).result(timeout=1.0)
            finally:
                if transport is not None:
                    transport.close()
                stop.set()
                executor.wake()
                worker.join(timeout=2.0)
                executor.shutdown()
                panel.destroy_node()
                server.destroy_node()
                context.shutdown()
