# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""사용자가 만든 rb_hand.usd 를 불러와 조인트 움직임을 테스트하는 스크립트.

이 USD 는 RB5 암 + Inspire Hand 가 하나의 articulation(18 DOF: 암 6 + 손 12, body 20개)
으로 결합되어 있다. 이 스크립트는 암과 손 조인트를 사인파로 구동하면서 각 조인트의
현재 위치를 출력해 모든 조인트가 정상 동작하는지 확인한다.

.. code-block:: bash

    # 암 + 손 동시
    ./isaaclab.sh -p scripts/tutorials/01_assets/run_rb_hand_test.py
    # 손가락만
    ./isaaclab.sh -p scripts/tutorials/01_assets/run_rb_hand_test.py --hand_only
    # 조인트를 하나씩 순차적으로
    ./isaaclab.sh -p scripts/tutorials/01_assets/run_rb_hand_test.py --sequential
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="사용자 rb_hand.usd 조인트 움직임 테스트.")
parser.add_argument("--sequential", action="store_true", help="조인트를 하나씩 순차적으로 움직여 확인합니다.")
parser.add_argument("--hand_only", action="store_true", help="암은 고정하고 손가락만 움직입니다.")
parser.add_argument("--amplitude", type=float, default=0.4, help="조인트 구동 사인파의 진폭 (rad).")
parser.add_argument("--frequency", type=float, default=0.25, help="조인트 구동 사인파의 주파수 (Hz).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import math
import os

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.sim import SimulationContext

# 사용자가 만든 결합 USD
RB_HAND_USD = "/home/joy4mj/rb_hand.usd"
# USD 를 /World/RbHand 아래에 스폰했을 때, 결합 articulation 의 prim 경로
ROBOT_PRIM = "/World/RbHand/rb5_850e/rb5_850e"

# 암 조인트 이름(나머지는 손으로 분류)
ARM_JOINTS = ["base", "shoulder", "elbow", "wrist1", "wrist2", "wrist3"]


def design_scene() -> Articulation:
    """바닥/조명/로봇을 배치하고 결합 articulation 을 생성."""
    sim_utils.GroundPlaneCfg().func("/World/defaultGroundPlane", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)).func(
        "/World/Light", sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # 사용자 USD 를 한 번 스폰 (내부에 결합 로봇이 들어있음)
    sim_utils.UsdFileCfg(usd_path=RB_HAND_USD).func("/World/RbHand", sim_utils.UsdFileCfg(usd_path=RB_HAND_USD))

    # 결합 로봇을 단일 articulation 으로 래핑
    # (right_thumb_2_joint 는 limit 하한 0.035 라 기본값 0.0 이 범위를 벗어남 -> 초기값 지정)
    robot = Articulation(
        ArticulationCfg(
            prim_path=ROBOT_PRIM,
            spawn=None,  # 이미 스폰된 prim 사용
            init_state=ArticulationCfg.InitialStateCfg(joint_pos={"right_thumb_2_joint": 0.1}),
            actuators={
                "arm": ImplicitActuatorCfg(
                    joint_names_expr=ARM_JOINTS,
                    effort_limit_sim=200.0,
                    velocity_limit_sim=100.0,
                    stiffness=2000.0,
                    damping=200.0,
                ),
                "hand": ImplicitActuatorCfg(
                    joint_names_expr=["right_.*_joint"],
                    effort_limit_sim=10.0,
                    velocity_limit_sim=100.0,
                    stiffness=50.0,
                    damping=5.0,
                ),
            },
        )
    )
    return robot


def run_simulator(sim: SimulationContext, robot: Articulation):
    """결합 로봇의 조인트를 움직이고 위치를 출력."""
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    joint_names = robot.data.joint_names
    num_joints = len(joint_names)
    arm_idx = [i for i, n in enumerate(joint_names) if n in ARM_JOINTS]
    hand_idx = [i for i in range(num_joints) if i not in arm_idx]

    print("=" * 78)
    print(f"[INFO] 결합 로봇 로드 완료. 총 조인트: {num_joints} (암 {len(arm_idx)} + 손 {len(hand_idx)})")
    print(f"   [암 ] {[joint_names[i] for i in arm_idx]}")
    print(f"   [손 ] {[joint_names[i] for i in hand_idx]}")
    print("=" * 78)

    default_joint_pos = robot.data.default_joint_pos.clone()

    while simulation_app.is_running():
        if count % 1000 == 0:
            count = 0
            sim_time = 0.0
            robot.write_joint_state_to_sim(
                robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
            )
            robot.reset()
            print("\n[INFO] 상태 초기화...\n")

        target = default_joint_pos.clone()
        phase = 2.0 * math.pi * args_cli.frequency * sim_time

        if args_cli.sequential:
            active = (count // 150) % num_joints
            target[:, active] = default_joint_pos[:, active] + args_cli.amplitude * math.sin(phase)
        else:
            if not args_cli.hand_only:
                for k, j in enumerate(arm_idx):
                    target[:, j] = default_joint_pos[:, j] + args_cli.amplitude * math.sin(
                        phase + k * (math.pi / max(len(arm_idx), 1))
                    )
            grip = 0.5 * (1.0 - math.cos(phase)) * args_cli.amplitude
            for j in hand_idx:
                target[:, j] = default_joint_pos[:, j] + grip

        robot.set_joint_position_target(target)
        robot.write_data_to_sim()
        sim.step()
        sim_time += sim_dt
        count += 1
        robot.update(sim_dt)

        if count % 30 == 0:
            pos = robot.data.joint_pos[0]
            if args_cli.sequential:
                active = (count // 150) % num_joints
                grp = "암" if active in arm_idx else "손"
                print(f"[t={sim_time:6.2f}s] 움직이는 조인트: [{grp}] {joint_names[active]}")
            arm_str = " ".join(f"{joint_names[i]}={pos[i].item():+.2f}" for i in arm_idx)
            hand_str = " ".join(f"{joint_names[i]}={pos[i].item():+.2f}" for i in hand_idx)
            print(f"[t={sim_time:6.2f}s] 암 : {arm_str}")
            print(f"[t={sim_time:6.2f}s] 손 : {hand_str}")


def main():
    if not os.path.exists(RB_HAND_USD):
        raise FileNotFoundError(f"USD 를 찾을 수 없습니다: {RB_HAND_USD}")
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([1.8, 1.8, 1.4], [0.0, 0.0, 0.4])
    robot = design_scene()
    sim.reset()
    print("[INFO] 셋업 완료...")
    run_simulator(sim, robot)


if __name__ == "__main__":
    main()
    simulation_app.close()
