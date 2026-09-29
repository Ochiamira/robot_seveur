#!/usr/bin/env python3
"""Spawn the restaurant customer relative to the current Gazebo clock.

Gazebo actor ``delay_start`` values are placed on the world's simulation
timeline.  A model spawned after the world has been running for a while must
therefore use ``current_sim_time + requested_wait`` instead of a fixed value.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rosgraph_msgs.msg import Clock


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crée le client avec un départ relatif à l'horloge Gazebo."
    )
    parser.add_argument("--wait", type=float, default=3.0,
                        help="Attente en secondes simulées avant le mouvement (défaut: 3).")
    parser.add_argument("--clock-timeout", type=float, default=10.0,
                        help="Délai maximal d'attente du topic /clock.")
    parser.add_argument("--world", default="restaurant")
    parser.add_argument("--name", default="customer_actor_1")
    parser.add_argument("--x", type=float, default=0.0)
    parser.add_argument("--y", type=float, default=-4.6)
    parser.add_argument("--z", type=float, default=0.0)
    parser.add_argument("--yaw", type=float, default=0.0)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Dossier restaurant_actor à utiliser (défaut: modèle installé).",
    )
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Ne tente pas de supprimer un client portant déjà le même nom.",
    )
    return parser.parse_args()


def current_sim_time(timeout_s: float) -> float:
    received: list[float] = []
    qos = QoSProfile(
        depth=1,
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
    )

    rclpy.init()
    node = rclpy.create_node("customer_scenario_clock_reader")

    def on_clock(msg: Clock) -> None:
        received.append(msg.clock.sec + msg.clock.nanosec / 1_000_000_000.0)

    subscription = node.create_subscription(Clock, "/clock", on_clock, qos)
    deadline = time.monotonic() + timeout_s
    try:
        while not received and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.2)
    finally:
        node.destroy_subscription(subscription)
        node.destroy_node()
        rclpy.shutdown()

    if not received:
        raise RuntimeError(
            "Aucun message reçu sur /clock. Vérifie Gazebo et ros_gz_bridge."
        )
    return received[-1]


def runtime_model(model_dir: Path, delay_start: float) -> Path:
    source_sdf = model_dir / "model.sdf"
    if not source_sdf.is_file():
        raise FileNotFoundError(f"Modèle acteur introuvable: {source_sdf}")

    runtime_dir = Path(tempfile.mkdtemp(prefix="nexor_actor_"))
    shutil.copytree(model_dir, runtime_dir, dirs_exist_ok=True)
    runtime_sdf = runtime_dir / "model.sdf"
    content = runtime_sdf.read_text(encoding="utf-8")
    content, replacements = re.subn(
        r"<delay_start>[^<]+</delay_start>",
        f"<delay_start>{delay_start:.3f}</delay_start>",
        content,
        count=1,
    )
    if replacements != 1:
        raise RuntimeError("Élément <delay_start> absent ou ambigu dans model.sdf")
    runtime_sdf.write_text(content, encoding="utf-8")
    return runtime_sdf


def remove_existing(world: str, name: str) -> None:
    command = [
        "gz", "service",
        "-s", f"/world/{world}/remove/blocking",
        "--reqtype", "gz.msgs.Entity",
        "--reptype", "gz.msgs.Boolean",
        "--timeout", "5000",
        "--req", f'name: "{name}", type: ACTOR',
    ]
    result = subprocess.run(command, check=False, text=True, capture_output=True)
    output = "\n".join(part.strip() for part in (result.stdout, result.stderr) if part.strip())
    if output:
        print(f"[CLIENT] Suppression précédente: {output}")


def spawn(args: argparse.Namespace, runtime_sdf: Path) -> int:
    command = [
        "ros2", "run", "ros_gz_sim", "create",
        "-world", args.world,
        "-name", args.name,
        "-file", str(runtime_sdf),
        "-x", str(args.x),
        "-y", str(args.y),
        "-z", str(args.z),
        "-Y", str(args.yaw),
    ]
    return subprocess.run(command, check=False).returncode


def main() -> int:
    args = parse_args()
    if args.wait < 0:
        print("[CLIENT] --wait doit être positif.", file=sys.stderr)
        return 2

    if args.model_dir is None:
        package_share = Path(get_package_share_directory("restaurant_sim"))
        model_dir = package_share / "models" / "restaurant_actor"
    else:
        model_dir = args.model_dir.expanduser().resolve()

    try:
        sim_time = current_sim_time(args.clock_timeout)
        delay_start = sim_time + args.wait
        runtime_sdf = runtime_model(model_dir, delay_start)
    except (FileNotFoundError, RuntimeError, OSError) as exc:
        print(f"[CLIENT] Erreur: {exc}", file=sys.stderr)
        return 2

    print(f"[CLIENT] Horloge Gazebo : {sim_time:.3f} s")
    print(f"[CLIENT] Attente        : {args.wait:.3f} s simulées")
    print(f"[CLIENT] Départ script  : {delay_start:.3f} s")
    print(f"[CLIENT] Modèle runtime : {runtime_sdf}")

    if not args.keep_existing:
        remove_existing(args.world, args.name)

    return spawn(args, runtime_sdf)


if __name__ == "__main__":
    raise SystemExit(main())
