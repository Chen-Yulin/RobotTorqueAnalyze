#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
import numpy as np


class GravityTorqueNode(Node):
    def __init__(self):
        super().__init__("gravity_torque_node")

        # === 加载 URDF ===
        urdf_path = self.declare_parameter(
            "urdf_path", "/home/jetson/xclimb_ws/src/xclimbot_model/urdf/X-Climbot.urdf"
        ).value

        self.robot = RobotWrapper.BuildFromURDF(urdf_path)
        self.model = self.robot.model
        self.data = self.robot.data
        self.model.gravity.linear = np.array([0, -9.81, 0])

        # === 关节名称 ===
        self.joint_names = ["j1", "j2", "j3", "j4", "j5", "j6"]

        # === 预计算关节索引映射 ===
        # 注意：continuous 关节在配置空间中占用 nq=2 的空间
        # 但 idx_qs 给出的是起始索引，idx_vs 给出速度空间的索引
        self.joint_q_indices = []
        self.joint_v_indices = []

        for jname in self.joint_names:
            joint_id = self.model.getJointId(jname)
            q_idx = self.model.idx_qs[joint_id]  # 配置空间起始索引
            v_idx = self.model.idx_vs[joint_id]  # 速度空间索引
            nq = self.model.nqs[joint_id]  # 配置空间维度
            nv = self.model.nvs[joint_id]  # 速度空间维度

            self.joint_q_indices.append(q_idx)
            self.joint_v_indices.append(v_idx)

            self.get_logger().info(
                f"{jname}: joint_id={joint_id}, q_idx={q_idx}, v_idx={v_idx}, "
                f"nq={nq}, nv={nv}"
            )

        # === 打印模型信息 ===
        self.get_logger().info("=" * 70)
        self.get_logger().info(f"Loaded: {urdf_path}")
        self.get_logger().info(f"Robot: {self.model.name}")
        self.get_logger().info(f"Configuration space dim (nq): {self.model.nq}")
        self.get_logger().info(f"Velocity space dim (nv): {self.model.nv}")

        total_mass = sum(
            self.model.inertias[i].mass for i in range(1, self.model.njoints)
        )
        self.get_logger().info(f"Total mass: {total_mass:.3f} kg")
        self.get_logger().info("=" * 70)

        # === 订阅 JointState ===
        self.subscription = self.create_subscription(
            JointState, "/joint_states", self.joint_state_callback, 10
        )

        self.get_logger().info("Ready. Listening to /joint_states...\n")

    def joint_state_callback(self, msg: JointState):
        """计算并显示重力力矩"""

        if len(msg.position) < 6:
            self.get_logger().warn(f"Expected 6 joints, got {len(msg.position)}")
            return

        # === 方法1：使用 pin.neutral() 然后填充（适用于 continuous 关节）===
        q_full = pin.neutral(self.model)  # 初始化为中性配置

        # 对于 continuous 关节，我们需要正确设置配置空间的值
        # Pinocchio 使用 (cos, sin) 表示，但我们可以用角度来设置
        for i, jname in enumerate(self.joint_names):
            joint_id = self.model.getJointId(jname)
            q_idx = self.model.idx_qs[joint_id]

            # 对于 continuous 关节（nq=2），使用角度表示
            angle = msg.position[i]
            # Pinocchio 的 continuous 关节表示: [cos(θ), sin(θ)]
            q_full[q_idx] = np.cos(angle)
            q_full[q_idx + 1] = np.sin(angle)

        # === 计算重力力矩 ===
        pin.computeGeneralizedGravity(self.model, self.data, q_full)

        # === 提取关节力矩 ===
        # 注意：重力力矩在速度空间中，所以使用 v_idx
        tau_joints = [self.data.g[v_idx] for v_idx in self.joint_v_indices]

        # === 输出 ===
        self.get_logger().info("Joint States:")
        for i, (jname, q, tau) in enumerate(
            zip(self.joint_names, msg.position[:6], tau_joints)
        ):
            self.get_logger().info(
                f"  {jname}: q={q:7.3f} rad ({np.degrees(q):7.2f}°), τ_g={tau:8.3f} Nm"
            )

        max_tau = max(abs(t) for t in tau_joints)
        self.get_logger().info(f"Max |τ_g|: {max_tau:.3f} Nm")

        # 合理性检查
        if max_tau > 50:
            self.get_logger().warn(f"⚠️  High torque detected!")

        self.get_logger().info("-" * 70 + "\n")


def main():
    rclpy.init()
    node = GravityTorqueNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
