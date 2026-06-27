# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RB5-850E 암 끝(link6)에 Inspire Hand를 mount한 결합 USD를 생성하는 스크립트.

두 로봇은 각각 별도의 USD(서로 다른 articulation)이므로, 그대로는 한 articulation으로
제어할 수 없습니다. 이 스크립트는 다음을 수행해 둘을 하나의 articulation으로 결합합니다.

  1. 새 stage에 /World/Robot (ArticulationRoot) 컨테이너 생성
  2. 암 USD( /World/rb5_850e )와 핸드 USD( /World/a_ )를 reference 로 가져옴
  3. 두 USD가 각자 갖고 있던 world-고정 root_joint 를 비활성화
  4. base_fix_joint : world  -> 암 link0  (고정 베이스)
     hand_mount_joint: 암 link6 -> 핸드 right_hand_base_link (손목에 손 장착)

결과물: /home/joy4mj/rb5_with_inspire_hand.usd

.. code-block:: bash

    ./isaaclab.sh -p scripts/tutorials/01_assets/build_rb_hand_usd.py
"""

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="RB5 + Inspire Hand 결합 USD 생성기")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------
ARM_USD = "/home/joy4mj/rb5_850e.usd"
HAND_USD = "/home/joy4mj/inspire_hand.usd"
OUT_USD = "/home/joy4mj/rb5_with_inspire_hand.usd"

# 결합 stage 내부 경로 (default prim 은 pseudo-root 직속이어야 하므로 최상위 /Robot 사용)
ROBOT_PATH = "/Robot"
ARM_PATH = "/Robot/rb5_850e"
HAND_PATH = "/Robot/inspire_hand"

# 소스 USD의 서브트리 경로
ARM_SRC = "/World/rb5_850e"
HAND_SRC = "/World/a_"

# 암 끝단(손이 붙을 링크)과 손 베이스 링크
ARM_EE_LINK = f"{ARM_PATH}/link6"
ARM_TCP = f"{ARM_PATH}/link6/tcp"
HAND_BASE = f"{HAND_PATH}/right_hand_base_link"

# 마운트 미세조정 파라미터.
#  - 손의 "손목 장착면"이 암 link6 의 "플랜지 면"에 밀착되도록 자동 정렬한다.
#  - 플랜지 면 / 손 장착면 위치는 각 메시의 바운딩박스에서 자동으로 계산한다(공구축 = link6 -Y).
#  - 아래 값으로만 추가 미세조정한다.
MOUNT_GAP = 0.0  # (m) +면 손을 플랜지에서 공구축 방향으로 더 띄움(틈), -면 더 파고듦
MOUNT_ROLL_DEG = 0.0  # (deg) 공구축(손목) 기준 손 회전 — 손바닥 방향 조정용


def main():
    # 소스 stage 열기 (단위 / tcp 변환 참고용)
    arm_stage = Usd.Stage.Open(ARM_USD)
    meters_per_unit = UsdGeom.GetStageMetersPerUnit(arm_stage)

    # 새 결합 stage 생성
    stage = Usd.Stage.CreateNew(OUT_USD)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, meters_per_unit)

    # 최상위 /Robot 컨테이너 (= default prim 후보)
    robot_xform = UsdGeom.Xform.Define(stage, ROBOT_PATH)
    robot_prim = robot_xform.GetPrim()

    # 결합 로봇을 단일 articulation 으로 만들기 위한 ArticulationRoot
    UsdPhysics.ArticulationRootAPI.Apply(robot_prim)

    # 스폰 시 이 prim 이 배치되도록 default prim 지정
    stage.SetDefaultPrim(robot_prim)

    # 암 / 핸드 서브트리를 reference 로 가져오기 (메시 등 외부 에셋 경로 보존)
    arm_prim = stage.OverridePrim(ARM_PATH)
    arm_prim.GetReferences().AddReference(ARM_USD, ARM_SRC)
    hand_prim = stage.OverridePrim(HAND_PATH)
    hand_prim.GetReferences().AddReference(HAND_USD, HAND_SRC)

    # 각 소스가 갖고 있던 world-고정 root_joint 비활성화
    for rj in (f"{ARM_PATH}/root_joint", f"{HAND_PATH}/root_joint"):
        p = stage.OverridePrim(rj)
        p.SetActive(False)

    # --- 플랜지 면 / 손 장착면 위치를 바운딩박스로 자동 계산 ---
    # 공구축은 link6 의 -Y (tcp 가 link6 -Y 방향에 있음).
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])

    # 암 link6 의 플랜지 면 = link6 메시에서 가장 -Y 인 면
    arm_link6 = arm_stage.GetPrimAtPath(f"{ARM_SRC}/link6")
    link6_rng = bbox_cache.ComputeUntransformedBound(arm_link6).ComputeAlignedRange()
    flange_y = float(link6_rng.GetMin()[1])  # 가장 -Y (공구축 끝 = 플랜지 면)

    # 손 base 의 손목 장착면 = 손 base 메시에서 가장 +Y 인 면(손가락은 -Y 로 뻗으므로 손목쪽은 +Y)
    hand_stage = Usd.Stage.Open(HAND_USD)
    hand_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    hand_base = hand_stage.GetPrimAtPath(f"{HAND_SRC}/right_hand_base_link")
    hand_rng = hand_cache.ComputeUntransformedBound(hand_base).ComputeAlignedRange()
    hand_mount_y = float(hand_rng.GetMax()[1])  # 가장 +Y (손목 장착면)

    # 공구축(-Y) 으로 MOUNT_GAP 만큼 띄운 지점에 손목면을 맞춘다.
    # fixed joint: localPose0(link6) 와 localPose1(손) 프레임이 일치하도록 구속된다.
    #   link6 의 부착점 = (0, flange_y - MOUNT_GAP, 0)   (-Y 방향으로 gap 만큼 이동)
    #   손  의 부착점 = (0, hand_mount_y, 0)             (손목 장착면)
    # 두 프레임 회전을 정렬(identity)하면 손의 -Y(손가락)가 link6 의 -Y(공구축)와 일치한다.
    link6_attach = Gf.Vec3f(0.0, flange_y - MOUNT_GAP, 0.0)
    hand_attach = Gf.Vec3f(0.0, hand_mount_y, 0.0)

    # 공구축(Y) 기준 roll 회전 (손바닥 방향 미세조정)
    half = math.radians(MOUNT_ROLL_DEG) * 0.5
    mount_rot0 = Gf.Quatf(math.cos(half), 0.0, math.sin(half), 0.0)  # Y축 회전

    # 조인트 컨테이너
    UsdGeom.Scope.Define(stage, f"{ROBOT_PATH}/mount_joints")

    # 1) 베이스를 world 에 고정 (fixed-base articulation)
    base_fix = UsdPhysics.FixedJoint.Define(stage, f"{ROBOT_PATH}/mount_joints/base_fix_joint")
    base_fix.CreateBody1Rel().SetTargets([Sdf.Path(f"{ARM_PATH}/link0")])
    base_fix.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    base_fix.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    base_fix.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    base_fix.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

    # 2) 손목(link6) 에 손(right_hand_base_link) 고정 — 손목면을 플랜지 면에 밀착
    hand_joint = UsdPhysics.FixedJoint.Define(stage, f"{ROBOT_PATH}/mount_joints/hand_mount_joint")
    hand_joint.CreateBody0Rel().SetTargets([Sdf.Path(ARM_EE_LINK)])
    hand_joint.CreateBody1Rel().SetTargets([Sdf.Path(HAND_BASE)])
    hand_joint.CreateLocalPos0Attr().Set(link6_attach)
    hand_joint.CreateLocalRot0Attr().Set(mount_rot0)
    hand_joint.CreateLocalPos1Attr().Set(hand_attach)
    hand_joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

    # 저장
    stage.GetRootLayer().Save()
    print("=" * 70)
    print(f"[OK] 결합 USD 생성 완료: {OUT_USD}")
    print(f"     - ArticulationRoot   : {ROBOT_PATH}")
    print(f"     - 플랜지 면(link6 -Y) : y={flange_y:.4f}")
    print(f"     - 손 손목 장착면      : y={hand_mount_y:.4f}")
    print(f"     - link6 부착점        : {tuple(round(v, 4) for v in link6_attach)}")
    print(f"     - 손   부착점         : {tuple(round(v, 4) for v in hand_attach)}")
    print(f"     - MOUNT_GAP={MOUNT_GAP}  MOUNT_ROLL_DEG={MOUNT_ROLL_DEG}")
    print(f"     - meters_per_unit     : {meters_per_unit}")
    print("=" * 70)


if __name__ == "__main__":
    main()
    simulation_app.close()
