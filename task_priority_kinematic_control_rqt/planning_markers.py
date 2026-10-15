"""RQT client for the independent planning marker authority (no marker servers)."""
from copy import deepcopy
import time
import uuid

from python_qt_binding.QtCore import QObject, Qt, QTimer, Signal
from python_qt_binding.QtWidgets import QCheckBox, QLabel, QLineEdit
from rclpy.validate_full_topic_name import validate_full_topic_name
from .marker_transport import MarkerTransport

from .marker_geometry import (  # Re-export pose helpers used by the Cartesian form.
    pose_components, pose_from_components, pose_from_transform, transform_pose,
    normalized_pose, relative_pose,
)


def target_interfaces():
    from sura_manipulation_actions.msg import PlanningTargetState
    from sura_manipulation_actions.srv import UpdatePlanningTarget
    return PlanningTargetState, UpdatePlanningTarget


class PlanningMarkerEditor(QObject):
    # Worker callbacks emit only data/callables. All widget changes and goal
    # dispatch remain on the Qt thread, even when a reply is already complete.
    _state_received = Signal(object)
    _reply_received = Signal(object)

    def __init__(self, tab, form, bimanual=False):
        super().__init__(tab)
        self._connection_generation = 0
        self._state_received.connect(self._deliver_state, Qt.QueuedConnection)
        self._reply_received.connect(lambda callback: callback(), Qt.QueuedConnection)
        self.tab, self.bimanual = tab, bimanual
        mode = 'bimanual' if bimanual else 'cartesian'
        suffix = 'bimanual_cartesian_goal_markers' if bimanual else 'cartesian_goal_markers'
        self.show = QCheckBox('Mostrar objetivos en RViz')
        self.show.setChecked(True)
        self.namespace = QLineEdit('/cirtesub/manipulation/' + suffix)
        self.endpoint = QLineEdit(f'/cirtesub/manipulation/planning_targets/{mode}/update')
        self.status = QLabel('Waiting for the independent planning marker server')
        self.status.setWordWrap(True)
        form.addRow(self.show)
        form.addRow('Planning marker namespace', self.namespace)
        form.addRow('Planning target service', self.endpoint)
        form.addRow('RViz target', self.status)
        if bimanual:
            note = QLabel('Markers define relative separation/orientation only. Their absolute poses '
                          'are illustrative; the base is fixed.')
            note.setWordWrap(True)
            form.addRow(note)
        tab._inputs.extend([self.show, self.namespace, self.endpoint])
        self.client_id = uuid.uuid4().hex
        self.client = None
        self.confirmed = None
        self.closed, self.editable = False, True
        self.poses, self.frame, self.selection = [], '', ()
        self._inflight = None
        self._queued = None
        self._heartbeat = None
        self._state_type = self._api = None
        self._applying = False
        self._selection_pending = False
        self._last_fields = self._field_text()
        try:
            self._state_type, self._api = target_interfaces()
        except ImportError:
            self.status.setText('Rebuild sura_manipulation_actions for the planning target interfaces. '
                                'Disable markers to use manual goals.')
        self.endpoint.editingFinished.connect(self._connect)
        self.namespace.editingFinished.connect(self.commit_fields)
        self.show.toggled.connect(self._visibility_changed)
        for edit in (tab.xyz_edit, tab.quaternion_edit):
            edit.editingFinished.connect(self.commit_fields)
        if bimanual:
            for combo in (tab.arm_a_combo, tab.arm_b_combo):
                combo.currentTextChanged.connect(self._schedule_selection)
            for edit in (tab.arm_a_tip_edit, tab.arm_b_tip_edit, tab.reference_frame_edit):
                edit.editingFinished.connect(self._selection_changed)
        else:
            tab.arm_combo.currentTextChanged.connect(self._schedule_selection)
            tab.tool_frame_edit.editingFinished.connect(self._selection_changed)
        self.timer = QTimer(self)
        self.timer.setInterval(500)
        self.timer.timeout.connect(self._tick)
        self.timer.start()
        self._connect()

    def _connect(self):
        if self.closed or self._api is None:
            return
        if self._inflight is not None:
            self.status.setText('Wait for the current target request before changing endpoints')
            return
        endpoint = self.endpoint.text().strip()
        if not endpoint.endswith('/update'):
            self.status.setText('Planning target service must end in /update')
            return
        try:
            validate_full_topic_name(endpoint)
        except ValueError as error:
            self.status.setText(str(error))
            return
        if self.client is not None:
            self.client.close()
        self._connection_generation += 1
        generation = self._connection_generation
        self.confirmed = None
        self.client = MarkerTransport(
            self.tab._node.context, self._api, self._state_type, endpoint,
            lambda state: self._state_received.emit((generation, state)))

    def _deliver_state(self, update):
        generation, state = update
        if generation == self._connection_generation:
            self._on_state(state)

    def _field_text(self):
        return self.tab.xyz_edit.text(), self.tab.quaternion_edit.text()

    def _field_pose(self):
        p, q = ([float(v.strip()) for v in text.split(',')] for text in self._field_text())
        return pose_from_components(p, q)

    def _selection(self):
        tab = self.tab
        if self.bimanual:
            return ((tab.arm_a_combo.currentText().strip(), tab.arm_a_tip_edit.text().strip()),
                    (tab.arm_b_combo.currentText().strip(), tab.arm_b_tip_edit.text().strip()))
        return ((tab.arm_combo.currentText().strip(), tab.tool_frame_edit.text().strip()),)

    def _write_fields(self, pose):
        for edit, values in zip((self.tab.xyz_edit, self.tab.quaternion_edit), pose_components(pose)):
            previous = edit.blockSignals(True)
            edit.setText(','.join(f'{v:.12g}' for v in values))
            edit.blockSignals(previous)
        self._last_fields = self._field_text()

    def _apply_state(self, state):
        self._applying = True
        try:
            self.confirmed = deepcopy(state)
            self.poses, self.frame = deepcopy(state.tool_poses), state.frame_id
            self.selection = tuple(zip(state.arm_names, state.tool_frames))
            widgets = [(self.namespace, state.marker_namespace), (self.tab.reference_frame_edit, state.frame_id)]
            if self.bimanual:
                widgets += [(self.tab.arm_a_combo, state.arm_names[0]),
                            (self.tab.arm_b_combo, state.arm_names[1]),
                            (self.tab.arm_a_tip_edit, state.tool_frames[0]),
                            (self.tab.arm_b_tip_edit, state.tool_frames[1])]
            else:
                widgets += [(self.tab.arm_combo, state.arm_names[0]),
                            (self.tab.tool_frame_edit, state.tool_frames[0])]
                self.tab._target_frame = state.frame_id
            old_arm = self.tab.arm_combo.currentText() if not self.bimanual else None
            for widget, text in widgets:
                previous = widget.blockSignals(True)
                if hasattr(widget, 'setCurrentText'):
                    widget.setCurrentText(text)
                else:
                    widget.setText(text)
                widget.blockSignals(previous)
            previous = self.show.blockSignals(True)
            self.show.setChecked(state.visible)
            self.show.blockSignals(previous)
            if state.tool_poses:
                self._write_fields(state.target_pose.pose)
            if old_arm is not None and old_arm != state.arm_names[0]:
                self.tab._refresh_joint_catalog()
            self.status.setText(state.status)
            foreign_lock = state.locked and state.owner_id != self.client_id
            if foreign_lock:
                for widget in self.tab._inputs:
                    widget.setEnabled(False)
                self.tab.send_button.setEnabled(False)
        finally:
            self._applying = False

    def _on_state(self, state):
        if self.closed or state.bimanual != self.bimanual:
            return
        same_server = self.confirmed is not None and state.server_id == self.confirmed.server_id
        if same_server and state.revision < self.confirmed.revision:
            return
        # Keep in-progress numeric edits intact until their edit request is sent.
        dirty = self._field_text() != self._last_fields
        if self._inflight is not None or dirty:
            self.confirmed = deepcopy(state)
            return
        was_foreign = self.confirmed is not None and self.confirmed.locked and self.confirmed.owner_id != self.client_id
        self._apply_state(state)
        if state.locked and state.owner_id == self.client_id and not self.tab._busy and self._inflight is None:
            self._request(self._api.Request.RELEASE)
        if was_foreign and not state.locked:
            self.tab._update_controls()

    def _draft(self):
        state = deepcopy(self.confirmed) if self.confirmed is not None else self._state_type()
        state.bimanual = self.bimanual
        state.marker_namespace = self.namespace.text().strip().rstrip('/')
        state.frame_id = self.tab.reference_frame_edit.text().strip()
        state.arm_names = [arm for arm, _ in self._selection()]
        state.tool_frames = [tool for _, tool in self._selection()]
        state.target_pose.header.frame_id = state.frame_id
        state.target_pose.pose = self._field_pose()
        state.tool_poses = deepcopy(self.confirmed.tool_poses) if self.confirmed is not None else deepcopy(self.poses)
        state.visible = self.show.isChecked()
        state.action_name = self.tab.action_edit.text().strip()
        return state

    def _request(self, operation, draft=None, done=None, failed=None):
        if self.closed:
            return
        if self._api is None or self.client is None or not self.client.service_is_ready() or self.confirmed is None:
            message = 'Independent marker server unavailable; start planning_markers.launch.py or disable markers'
            self.status.setText(message)
            if failed:
                failed(message)
            return
        if self._inflight is not None:
            # Keep the last draft while one acknowledged edit is outstanding.
            self._queued = (operation, draft, done, failed)
            return
        request = self._api.Request()
        request.operation, request.client_id = operation, self.client_id
        request.expected_revision = self.confirmed.revision
        request.target = draft if draft is not None else deepcopy(self.confirmed)
        future = self.client.call_async(request)
        self._inflight = (future, time.monotonic(), failed)
        def completed(reply):
            if self.closed or self._inflight is None or self._inflight[0] is not reply:
                return
            self._inflight = None
            succeeded, failure_message = False, 'Target update failed'
            try:
                response = reply.result()
                succeeded, failure_message = response.success, response.message
                latest = self.confirmed
                state = latest if latest is not None and latest.server_id == response.state.server_id and \
                    latest.revision > response.state.revision else response.state
                self._apply_state(state)
                if response.success:
                    if operation == self._api.Request.LOCK and (not state.locked or state.owner_id != self.client_id):
                        raise ValueError('Target reservation expired before dispatch; retry planning')
                    if done:
                        done(state)
                else:
                    self.status.setText(response.message)
                    if failed:
                        failed(response.message)
            except Exception as error:
                succeeded, failure_message = False, str(error)
                self.status.setText(str(error))
                if failed:
                    failed(str(error))
            queued, self._queued = self._queued, None
            if queued is not None:
                if not succeeded and queued[0] == self._api.Request.LOCK:
                    if queued[3]:
                        queued[3](failure_message)
                else:
                    self._request(*queued)
        future.add_done_callback(lambda reply: self._reply_received.emit(lambda: completed(reply)))

    def set_poses(self, poses, frame):
        # Used by the legacy manual TF read when markers are disabled.
        self.poses, self.frame, self.selection = [normalized_pose(p) for p in poses], frame, self._selection()
        self._write_fields(relative_pose(*self.poses) if self.bimanual else self.poses[0])
        self.commit_fields()

    def read_pose(self):
        if not self.show.isChecked():
            return False
        if self._api is None:
            self.status.setText('Rebuild the planning target interfaces or disable markers for manual TF reads')
            return True
        try:
            self._request(self._api.Request.READ_TF, self._draft())
        except ValueError as error:
            self.status.setText(str(error))
        return True

    def commit_fields(self):
        if self.closed or not self.editable or self._applying or not self.show.isChecked() or self._api is None:
            return
        try:
            self._request(self._api.Request.EDIT, self._draft())
        except ValueError as error:
            self.status.setText(str(error))

    def _schedule_selection(self, *_):
        if not self._selection_pending and not self._applying:
            self._selection_pending = True
            QTimer.singleShot(0, self._selection_changed)

    def _selection_changed(self):
        self._selection_pending = False
        self.commit_fields()

    def _visibility_changed(self, *_):
        if self._applying or self._api is None or not self.editable:
            return
        try:
            self._request(self._api.Request.EDIT, self._draft())
        except ValueError as error:
            self.status.setText(str(error))

    def prepare_goal(self, goal, done, failed):
        if not self.show.isChecked():
            done(goal)
            return
        if self._api is None:
            failed('Rebuild the planning target interfaces or disable markers for manual goals')
            return
        try:
            draft = self._draft()
            def locked(state):
                if self.bimanual:
                    goal.arm_a, goal.arm_b = state.arm_names
                    goal.relative_target = deepcopy(state.target_pose)
                else:
                    goal.arm = state.arm_names[0]
                    goal.target_pose = deepcopy(state.target_pose)
                done(goal)
            self._request(self._api.Request.LOCK, draft, locked, failed)
        except ValueError as error:
            failed(str(error))

    def bind_goal(self, handle):
        if self.confirmed is not None and self.confirmed.locked and self.confirmed.owner_id == self.client_id:
            draft = deepcopy(self.confirmed)
            # Fixed ROS uint8 arrays yield numpy scalars; message setters require
            # native Python ints when assigning a sequence.
            draft.goal_id = [int(value) for value in handle.goal_id.uuid]
            self._request(self._api.Request.BIND_GOAL, draft)

    def set_editable(self, editable):
        if self.closed:
            return
        changed = self.editable != editable
        self.editable = editable
        if changed and editable and self.confirmed is not None and self.confirmed.locked and self.confirmed.owner_id == self.client_id:
            self._request(self._api.Request.RELEASE)
        if self.confirmed is not None and self.confirmed.locked and self.confirmed.owner_id != self.client_id:
            for widget in self.tab._inputs:
                widget.setEnabled(False)
            self.tab.send_button.setEnabled(False)

    def _tick(self):
        if self.closed:
            return
        if self._inflight is not None and not self._inflight[0].done() and time.monotonic() - self._inflight[1] > 3.0:
            _, _, failed = self._inflight
            queued = self._queued
            self._inflight = None
            self._queued = None
            message = 'Planning target request timed out; check the independent marker node'
            self.status.setText(message)
            if failed:
                failed(message)
            if queued is not None and queued[3]:
                queued[3](message)
        if self.tab._busy and self.confirmed is not None and self.confirmed.locked and self.confirmed.owner_id == self.client_id and self._api is not None:
            if self._heartbeat is None or self._heartbeat.done():
                request = self._api.Request()
                request.operation = self._api.Request.HEARTBEAT
                request.client_id = self.client_id
                self._heartbeat = self.client.call_async(request)

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.timer.stop()
        # Do not unlock an executing goal: the independent server observes its
        # action status and remains available after the panel disappears.
        if self.client is not None:
            self.client.close()
            self.client = None
