#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
import numpy as np
from collections import deque
import threading

# Plotly Dash
from dash import Dash, dcc, html
from dash.dependencies import Output, Input
import plotly.graph_objs as go


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

        # 数据存储
        self.angle_history = [deque(maxlen=200) for _ in self.joint_names]
        self.torque_history = [deque(maxlen=200) for _ in self.joint_names]
        self.max_torque = np.zeros(len(self.joint_names))

        self.get_logger().info(f"Loaded URDF: {urdf_path}, nq = {self.model.nq}")

        # === JointState 订阅 ===
        self.subscription = self.create_subscription(
            JointState, "/joint_states", self.joint_state_callback, 10
        )

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

        for i in range(len(self.joint_names)):
            angle = msg.position[i]
            torque = tau_joints[i]  # ⚠️ 直接用 0~5 索引
            self.angle_history[i].append(angle)
            self.torque_history[i].append(torque)
            self.max_torque[i] = max(self.max_torque[i], abs(torque))


# === Dash App ===
def run_dash(node):
    app = Dash(__name__)
    app.layout = html.Div(
        [
            html.H1("Gravity Torque Monitor", style={"textAlign": "center"}),
            # 使用 grid 布局，每行3列
            html.Div(
                id="graphs-container",
                style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(3, 1fr)",  # 3列，每列等宽
                    "gap": "20px",  # 图表之间的间距
                    "padding": "20px",
                },
            ),
            # 最大力矩显示
            html.Div(
                id="max-torques",
                style={
                    "padding": "20px",
                    "backgroundColor": "#f0f0f0",
                    "borderRadius": "10px",
                    "margin": "20px",
                },
            ),
            # 定时更新
            dcc.Interval(id="interval", interval=1000, n_intervals=0),
        ]
    )

    @app.callback(
        Output("graphs-container", "children"),
        Output("max-torques", "children"),
        Input("interval", "n_intervals"),
    )
    def update_graph(_):
        graph_components = []
        max_texts = []

        for i, jname in enumerate(node.joint_names):
            angles = list(node.angle_history[i])
            torques = list(node.torque_history[i])

            fig = go.Figure()

            # 添加角度曲线（使用左侧Y轴）
            fig.add_trace(
                go.Scatter(
                    y=angles,
                    mode="lines+markers",
                    name="angle (rad)",
                    line=dict(color="blue", width=2),
                    marker=dict(size=4),
                    yaxis="y1",
                )
            )

            # 添加力矩曲线（使用右侧Y轴）
            fig.add_trace(
                go.Scatter(
                    y=torques,
                    mode="lines+markers",
                    name="gravity torque (Nm)",
                    line=dict(color="red", width=2),
                    marker=dict(size=4),
                    yaxis="y2",
                )
            )

            # 设置双Y轴，分别固定范围
            fig.update_layout(
                yaxis=dict(
                    title="Angle (rad)",
                    range=[-5, 5],  # 角度范围 ±5 rad
                    fixedrange=False,
                    side="left",
                ),
                yaxis2=dict(
                    title="Torque (Nm)",
                    range=[
                        -node.max_torque[i],
                        node.max_torque[i],
                    ],  # 固定力矩范围为 ±100 Nm
                    fixedrange=False,
                    overlaying="y",
                    side="right",
                ),
                xaxis=dict(title="Time step"),
                title=dict(text=f"{jname}", font=dict(size=16, color="#333")),
                height=350,
                margin=dict(l=60, r=60, t=60, b=50),
                legend=dict(
                    x=0.5,
                    y=1.15,
                    xanchor="center",
                    orientation="h",
                    bgcolor="rgba(255,255,255,0.8)",
                ),
                plot_bgcolor="#fafafa",
                paper_bgcolor="white",
            )

            # 创建每个图表的容器
            graph_components.append(
                dcc.Graph(figure=fig, style={"width": "100%", "height": "100%"})
            )

            max_texts.append(f"{jname}: {node.max_torque[i]:.2f} Nm")

        # 最大力矩文本显示
        max_torque_display = html.Div(
            [
                html.H3("Maximum Torques:", style={"marginBottom": "10px"}),
                html.Div(
                    [
                        html.Span(
                            text,
                            style={
                                "display": "inline-block",
                                "margin": "5px 15px",
                                "padding": "8px 15px",
                                "backgroundColor": "#e8f4f8",
                                "borderRadius": "5px",
                                "fontWeight": "bold",
                            },
                        )
                        for text in max_texts
                    ]
                ),
            ]
        )

        return graph_components, max_torque_display

    app.run(debug=False, port=8050)


# === Main Thread ===
def main():
    rclpy.init()
    node = GravityTorqueNode()

    # Dash 在后台线程运行
    dash_thread = threading.Thread(target=run_dash, args=(node,), daemon=True)
    dash_thread.start()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
