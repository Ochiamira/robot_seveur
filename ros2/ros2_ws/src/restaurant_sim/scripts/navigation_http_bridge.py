#!/usr/bin/env python3
"""HTTP boundary between the Windows vision pipeline and ROS 2 Nav2.

Only fixed, reviewed restaurant table identifiers are accepted.  Pixel
coordinates produced by the vision pipeline are deliberately ignored: they
are not map coordinates and must never be sent directly to Nav2.
"""

import json
import math
import os
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node


# Safe approach poses in the saved restaurant map (x, y, yaw in radians).
# Each pose keeps the robot out of the furniture footprint and oriented
# toward the customer. Table 4 uses a diagonal approach so the chair does
# not hide the seated customer from the robot camera.
TABLE_GOALS = {
    1: (0.90, -1.50, -math.pi / 2.0),
    2: (0.90, 1.50, math.pi / 2.0),
    3: (4.00, -1.50, -math.pi / 2.0),
    # Client T4 assis sur la chaise ouest. Cette pose se trouve au
    # nord-ouest du client : elle évite le dossier de sa chaise ainsi que
    # la chaise nord, tout en gardant environ 1.6 m de recul caméra.
    4: (2.50, 1.50, 0.35),
}

# Pose de retour dans le repere ``map``. Elle correspond au point de depart
# valide dans RViz / AMCL, et non aux coordonnees brutes du monde Gazebo.
HOME_GOAL = (0.0, 0.0, 0.0)

ACTIVE_STATES = {
    "queued", "waiting_nav2", "sending", "navigating", "canceling",
}


class NavigationBridge(Node):
    def __init__(self) -> None:
        super().__init__("restaurant_navigation_bridge")
        self._client = ActionClient(self, NavigateToPose, "navigate_to_pose")
        self._commands = queue.Queue()
        self._lock = threading.Lock()
        self._goal_handle = None
        self._state = {
            "state": "idle",
            "nav2_ready": False,
            "target": None,
            "table_id": None,
            "canonical_id": None,
            "order_id": None,
            "distance_remaining": None,
            "message": "",
            "updated_at": time.time(),
        }
        self.create_timer(0.10, self._tick)

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self._state)

    def _update(self, **values) -> None:
        with self._lock:
            self._state.update(values)
            self._state["updated_at"] = time.time()

    def enqueue_navigation(self, payload: dict):
        try:
            table_id = int(payload.get("table_id"))
            canonical_id = int(payload.get("canonical_id"))
        except (TypeError, ValueError):
            return False, "table_id et canonical_id doivent etre des entiers", 400

        if table_id not in TABLE_GOALS:
            return False, f"table inconnue: {table_id}", 400

        with self._lock:
            if self._state["state"] in ACTIVE_STATES:
                return False, "une navigation est deja active", 409
            self._state.update({
                "state": "queued",
                "target": f"table_{table_id}",
                "table_id": table_id,
                "canonical_id": canonical_id,
                "order_id": None,
                "distance_remaining": None,
                "message": "objectif mis en file",
                "updated_at": time.time(),
            })

        self._commands.put(
            ("navigate", table_id, canonical_id, None, time.monotonic())
        )
        return True, "objectif accepte", 202

    def enqueue_return_home(self, payload: dict):
        """Met en file une mission de retour apres une commande confirmee."""
        try:
            canonical_id = int(payload.get("canonical_id"))
        except (TypeError, ValueError):
            return False, "canonical_id doit etre un entier", 400

        raw_order_id = payload.get("order_id")
        order_id = str(raw_order_id) if raw_order_id is not None else None

        with self._lock:
            if self._state["state"] in ACTIVE_STATES:
                return False, "une navigation est deja active", 409
            self._state.update({
                "state": "queued",
                "target": "home",
                "table_id": None,
                "canonical_id": canonical_id,
                "order_id": order_id,
                "distance_remaining": None,
                "message": "retour home mis en file",
                "updated_at": time.time(),
            })

        self._commands.put(
            ("return_home", None, canonical_id, order_id, time.monotonic())
        )
        return True, "retour home accepte", 202

    def enqueue_delivery(self, payload: dict):
        """Envoie le robot livrer une commande marquee prete par le staff."""
        try:
            table_id = int(payload.get("table_id"))
        except (TypeError, ValueError):
            return False, "table_id doit etre un entier", 400

        if table_id not in TABLE_GOALS:
            return False, f"table inconnue: {table_id}", 400

        raw_order_id = payload.get("order_id")
        if raw_order_id is None or not str(raw_order_id).strip():
            return False, "order_id est obligatoire", 400
        order_id = str(raw_order_id).strip()

        with self._lock:
            if self._state["state"] in ACTIVE_STATES:
                return False, "une navigation est deja active", 409
            self._state.update({
                "state": "queued",
                "target": f"delivery_table_{table_id}",
                "table_id": table_id,
                "canonical_id": None,
                "order_id": order_id,
                "distance_remaining": None,
                "message": "livraison mise en file",
                "updated_at": time.time(),
            })

        self._commands.put(
            ("deliver_order", table_id, None, order_id, time.monotonic())
        )
        return True, "livraison acceptee", 202

    def enqueue_cancel(self):
        self._commands.put(("cancel",))
        return True, "annulation demandee", 202

    def _tick(self) -> None:
        ready = self._client.server_is_ready()
        self._update(nav2_ready=ready)
        try:
            command = self._commands.get_nowait()
        except queue.Empty:
            return

        if command[0] == "cancel":
            if self._goal_handle is None:
                self._update(state="idle", message="aucun objectif actif")
            else:
                self._goal_handle.cancel_goal_async()
                self._update(state="canceling", message="annulation en cours")
            return

        command_type, table_id, canonical_id, order_id, queued_at = command
        if not ready:
            if time.monotonic() - queued_at < 15.0:
                self._update(state="waiting_nav2", message="attente du serveur Nav2")
                self._commands.put(command)
            else:
                self._update(state="error", message="serveur Nav2 indisponible")
            return

        if command_type == "return_home":
            x, y, yaw = HOME_GOAL
            target = "home"
            sending_message = "envoi vers le point home"
            log_message = (
                f"Objectif retour client #{canonical_id}: home, "
                f"pose map=({x:.2f}, {y:.2f}, yaw={yaw:.2f})"
            )
        elif command_type == "deliver_order":
            x, y, yaw = TABLE_GOALS[table_id]
            target = f"delivery_table_{table_id}"
            sending_message = f"livraison vers table {table_id}"
            log_message = (
                f"Livraison commande {order_id}: table {table_id}, "
                f"pose map=({x:.2f}, {y:.2f}, yaw={yaw:.2f})"
            )
        else:
            x, y, yaw = TABLE_GOALS[table_id]
            target = f"table_{table_id}"
            sending_message = f"envoi vers table {table_id}"
            log_message = (
                f"Objectif client #{canonical_id}: table {table_id}, "
                f"pose map=({x:.2f}, {y:.2f}, yaw={yaw:.2f})"
            )
        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.orientation.z = math.sin(yaw / 2.0)
        pose.pose.orientation.w = math.cos(yaw / 2.0)

        goal = NavigateToPose.Goal()
        goal.pose = pose
        self._update(state="sending", target=target, message=sending_message)
        future = self._client.send_goal_async(
            goal, feedback_callback=self._feedback_callback)
        future.add_done_callback(self._goal_response_callback)
        self.get_logger().info(log_message)

    def _goal_response_callback(self, future) -> None:
        try:
            goal_handle = future.result()
        except Exception as exc:
            self._update(state="error", message=f"erreur envoi: {exc}")
            return
        if not goal_handle.accepted:
            self._update(state="rejected", message="objectif refuse par Nav2")
            return
        self._goal_handle = goal_handle
        self._update(state="navigating", message="objectif accepte par Nav2")
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._result_callback)

    def _feedback_callback(self, feedback_msg) -> None:
        distance = float(feedback_msg.feedback.distance_remaining)
        self._update(state="navigating", distance_remaining=round(distance, 3))

    def _result_callback(self, future) -> None:
        try:
            status = future.result().status
        except Exception as exc:
            self._update(state="error", message=f"erreur resultat: {exc}")
            self._goal_handle = None
            return

        labels = {
            GoalStatus.STATUS_SUCCEEDED: "succeeded",
            GoalStatus.STATUS_CANCELED: "canceled",
            GoalStatus.STATUS_ABORTED: "aborted",
        }
        state = labels.get(status, f"finished_{status}")
        target = self.snapshot().get("target")
        if state == "succeeded":
            if target == "home":
                message = "robot arrive au point home"
            elif str(target).startswith("delivery_table_"):
                message = "robot arrive pour livrer la commande"
            else:
                message = "robot arrive a la table"
        else:
            message = f"navigation vers {target or 'destination'} terminee: {state}"
        self._update(state=state, distance_remaining=0.0, message=message)
        self._goal_handle = None
        self.get_logger().info(message)


class NavigationHttpHandler(BaseHTTPRequestHandler):
    bridge = None

    def log_message(self, _format, *args):
        return

    def _reply(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _json_body(self):
        try:
            size = int(self.headers.get("Content-Length", "0"))
            return json.loads(self.rfile.read(size) or b"{}")
        except (ValueError, json.JSONDecodeError):
            return None

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/health", "/status"):
            self._reply(200, self.bridge.snapshot())
        else:
            self._reply(404, {"error": "route inconnue"})

    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/navigate":
            payload = self._json_body()
            if payload is None:
                self._reply(400, {"error": "JSON invalide"})
                return
            ok, message, status = self.bridge.enqueue_navigation(payload)
            self._reply(status, {"accepted": ok, "message": message,
                                 **self.bridge.snapshot()})
        elif path == "/return-home":
            payload = self._json_body()
            if payload is None:
                self._reply(400, {"error": "JSON invalide"})
                return
            ok, message, status = self.bridge.enqueue_return_home(payload)
            self._reply(status, {"accepted": ok, "message": message,
                                 **self.bridge.snapshot()})
        elif path == "/deliver-order":
            payload = self._json_body()
            if payload is None:
                self._reply(400, {"error": "JSON invalide"})
                return
            ok, message, status = self.bridge.enqueue_delivery(payload)
            self._reply(status, {"accepted": ok, "message": message,
                                 **self.bridge.snapshot()})
        elif path == "/cancel":
            ok, message, status = self.bridge.enqueue_cancel()
            self._reply(status, {"accepted": ok, "message": message})
        else:
            self._reply(404, {"error": "route inconnue"})


def main(args=None) -> None:
    rclpy.init(args=args)
    bridge = NavigationBridge()
    NavigationHttpHandler.bridge = bridge
    port = int(os.environ.get("NEXOR_NAV_PORT", "8090"))
    server = ThreadingHTTPServer(("0.0.0.0", port), NavigationHttpHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    bridge.get_logger().info(f"Passerelle navigation HTTP sur 0.0.0.0:{port}")

    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(bridge)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        executor.shutdown()
        bridge.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
