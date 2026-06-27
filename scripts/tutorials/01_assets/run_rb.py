# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RB5-850E 로봇암을 불러와서 각 조인트가 움직이는지 확인하는 스크립트.

각 조인트를 사인파로 순차/동시 구동하면서 조인트 이름과 현재 위치(rad)를
주기적으로 출력해 모든 조인트가 정상적으로 움직이는지 확인할 수 있습니다.

.. code-block:: bash

    # 사용법
    ./isaaclab.sh -p scripts/tutorials/01_assets/run_rb.py

    # 조인트를 하나씩 순차적으로 움직여 확인하고 싶을 때
    ./isaaclab.sh -p scripts/tutorials/01_assets/run_rb.py --sequential
"""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="RB5-850E 로봇의 각 조인트 움직임을 확인하는 스크립트.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument(
    "--sequential",
    action="store_true",
    help="설정하면 조인트를 동시에 흔드는 대신 하나씩 순차적으로 움직여 확인합니다.",
)
parser.add_argument("--amplitude", type=float, default=0.5, help="조인트 구동 사인파의 진폭 (rad).")
parser.add_argument("--frequency", type=float, default=0.25, help="조인트 구동 사인파의 주파수 (Hz).")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import math

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg

# RB5-850E USD 파일 경로
RB_USD_PATH = "/home/joy4mj/rb5_850e.usd"

RB_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=RB_USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
    ),
    # 모든 조인트(.*)에 위치 제어 액추에이터를 적용
    actuators={
        "all_joints": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit_sim=100.0,
            velocity_limit_sim=100.0,
            stiffness=1000.0,
            damping=100.0,
        ),
    },
)


class RbSceneCfg(InteractiveSceneCfg):
    """RB 로봇 확인용 씬 구성."""

    # 바닥
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # 조명
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # 로봇
    robot = RB_CONFIG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """시뮬레이션 루프: 각 조인트를 움직이고 위치를 출력."""
    robot = scene["robot"]
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    # 불러온 조인트 정보 출력
    joint_names = robot.data.joint_names
    num_joints = len(joint_names)
    print("=" * 70)
    print(f"[INFO] RB 로봇을 불러왔습니다. 총 조인트 개수: {num_joints}")
    for i, name in enumerate(joint_names):
        print(f"   [{i}] {name}")
    print("=" * 70)

    # 기본 조인트 위치 (출력의 기준점)
    default_joint_pos = robot.data.default_joint_pos.clone()

    while simulation_app.is_running():
        # 일정 주기마다 초기 상태로 리셋
        if count % 1000 == 0:
            count = 0
            sim_time = 0.0
            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            robot.write_root_pose_to_sim(root_state[:, :7])
            robot.write_root_velocity_to_sim(root_state[:, 7:])
            joint_pos = robot.data.default_joint_pos.clone()
            joint_vel = robot.data.default_joint_vel.clone()
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            scene.reset()
            print("\n[INFO] 로봇 상태를 초기화했습니다...\n")

        # 목표 조인트 위치 계산 (기본 위치 + 사인파)
        target = default_joint_pos.clone()
        phase = 2.0 * math.pi * args_cli.frequency * sim_time

        if args_cli.sequential:
            # 한 번에 하나의 조인트만 움직여서 개별 확인
            active = (count // 150) % num_joints
            target[:, active] = default_joint_pos[:, active] + args_cli.amplitude * math.sin(phase)
        else:
            # 모든 조인트를 위상 차이를 두고 동시에 움직임
            for j in range(num_joints):
                joint_phase = phase + j * (math.pi / max(num_joints, 1))
                target[:, j] = default_joint_pos[:, j] + args_cli.amplitude * math.sin(joint_phase)

        robot.set_joint_position_target(target)

        scene.write_data_to_sim()
        sim.step()
        sim_time += sim_dt
        count += 1
        scene.update(sim_dt)

        # 주기적으로 각 조인트의 현재 위치 출력 (움직임 확인)
        if count % 25 == 0:
            pos = robot.data.joint_pos[0]  # 첫 번째 환경
            if args_cli.sequential:
                active = (count // 150) % num_joints
                print(f"[t={sim_time:6.2f}s] 움직이는 조인트: [{active}] {joint_names[active]}")
            pos_str = "  ".join(f"{joint_names[i]}={pos[i].item():+.3f}" for i in range(num_joints))
            print(f"[t={sim_time:6.2f}s] {pos_str}")


def main():
    """메인 함수."""
    # 시뮬레이션 컨텍스트 초기화
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([2.5, 2.5, 2.0], [0.0, 0.0, 0.5])
    # 씬 구성
    scene_cfg = RbSceneCfg(args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # 시뮬레이터 재생
    sim.reset()
    print("[INFO] 셋업 완료...")
    # 시뮬레이션 루프 실행
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
