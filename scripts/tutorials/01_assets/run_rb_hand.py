# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RB5-850E 암 + Inspire Hand 결합 로봇을 불러와 각 조인트가 움직이는지 확인하는 스크립트.

먼저 build_rb_hand_usd.py 로 결합 USD( /home/joy4mj/rb5_with_inspire_hand.usd )를 생성해야 합니다.

    ./isaaclab.sh -p scripts/tutorials/01_assets/build_rb_hand_usd.py

그 다음 이 스크립트를 실행하면 암(6 DOF)과 손(12 DOF)을 사인파로 구동하면서
각 조인트의 현재 위치를 출력해 모든 조인트가 정상 동작하는지 확인할 수 있습니다.

.. code-block:: bash

    # 암 + 손 동시에 움직이며 확인
    ./isaaclab.sh -p scripts/tutorials/01_assets/run_rb_hand.py

    # 손가락만 쥐었다 폈다 하며 손 동작만 확인
    ./isaaclab.sh -p scripts/tutorials/01_assets/run_rb_hand.py --hand_only

    # 조인트를 하나씩 순차적으로 움직여 개별 확인
    ./isaaclab.sh -p scripts/tutorials/01_assets/run_rb_hand.py --sequential
"""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="RB5-850E + Inspire Hand 결합 로봇의 조인트 움직임 확인.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--sequential", action="store_true", help="조인트를 하나씩 순차적으로 움직여 확인합니다.")
parser.add_argument("--hand_only", action="store_true", help="암은 고정하고 손가락만 움직입니다.")
parser.add_argument("--amplitude", type=float, default=0.4, help="조인트 구동 사인파의 진폭 (rad).")
parser.add_argument("--frequency", type=float, default=0.25, help="조인트 구동 사인파의 주파수 (Hz).")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import math
import os

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg

# build_rb_hand_usd.py 로 생성한 결합 USD 경로
COMBINED_USD_PATH = "/home/joy4mj/rb5_with_inspire_hand.usd"

# 암 조인트 이름 (이 이름들에 해당하면 "암", 나머지는 "손"으로 분류)
ARM_JOINT_NAMES = ["base", "shoulder", "elbow", "wrist1", "wrist2", "wrist3"]

RB_HAND_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=COMBINED_USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        # right_thumb_2_joint 는 limit 하한이 0.035 라 기본값 0.0 이 범위를 벗어남 -> 범위 안으로 지정
        joint_pos={"right_thumb_2_joint": 0.1},
    ),
    actuators={
        # 암 조인트: 비교적 강한 게인
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["base", "shoulder", "elbow", "wrist1", "wrist2", "wrist3"],
            effort_limit_sim=200.0,
            velocity_limit_sim=100.0,
            stiffness=2000.0,
            damping=200.0,
        ),
        # 손가락 조인트: 작고 가벼우므로 낮은 게인
        "hand": ImplicitActuatorCfg(
            joint_names_expr=["right_.*_joint"],
            effort_limit_sim=10.0,
            velocity_limit_sim=100.0,
            stiffness=50.0,
            damping=5.0,
        ),
    },
)


class RbHandSceneCfg(InteractiveSceneCfg):
    """RB5 + Inspire Hand 확인용 씬."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )
    robot = RB_HAND_CONFIG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """시뮬레이션 루프: 암/손 조인트를 움직이고 위치를 출력."""
    robot = scene["robot"]
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    joint_names = robot.data.joint_names
    num_joints = len(joint_names)
    # 암 / 손 조인트 인덱스 분류
    arm_idx = [i for i, n in enumerate(joint_names) if n in ARM_JOINT_NAMES]
    hand_idx = [i for i, n in enumerate(joint_names) if i not in arm_idx]

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
            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            robot.write_root_pose_to_sim(root_state[:, :7])
            robot.write_root_velocity_to_sim(root_state[:, 7:])
            robot.write_joint_state_to_sim(
                robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
            )
            scene.reset()
            print("\n[INFO] 로봇 상태 초기화...\n")

        target = default_joint_pos.clone()
        phase = 2.0 * math.pi * args_cli.frequency * sim_time

        if args_cli.sequential:
            # 한 번에 하나의 조인트만 움직임
            active = (count // 150) % num_joints
            target[:, active] = default_joint_pos[:, active] + args_cli.amplitude * math.sin(phase)
        else:
            # 암: 위상차를 둔 웨이브 (hand_only 면 정지)
            if not args_cli.hand_only:
                for k, j in enumerate(arm_idx):
                    target[:, j] = default_joint_pos[:, j] + args_cli.amplitude * math.sin(
                        phase + k * (math.pi / max(len(arm_idx), 1))
                    )
            # 손: 모든 손가락을 함께 쥐었다 폈다 (0 ~ amplitude 범위)
            grip = 0.5 * (1.0 - math.cos(phase)) * args_cli.amplitude
            for j in hand_idx:
                target[:, j] = default_joint_pos[:, j] + grip

        robot.set_joint_position_target(target)
        scene.write_data_to_sim()
        sim.step()
        sim_time += sim_dt
        count += 1
        scene.update(sim_dt)

        # 주기적으로 조인트 위치 출력
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
    if not os.path.exists(COMBINED_USD_PATH):
        raise FileNotFoundError(
            f"결합 USD 를 찾을 수 없습니다: {COMBINED_USD_PATH}\n"
            "먼저 다음을 실행하세요: ./isaaclab.sh -p scripts/tutorials/01_assets/build_rb_hand_usd.py"
        )
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([1.8, 1.8, 1.4], [0.0, 0.0, 0.4])
    scene_cfg = RbHandSceneCfg(args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    print("[INFO] 셋업 완료...")
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
