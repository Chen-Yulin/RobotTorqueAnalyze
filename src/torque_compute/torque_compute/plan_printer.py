#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from moveit_msgs.msg import DisplayTrajectory


class PlanListener(Node):
    def __init__(self):
        super().__init__("plan_listener")

        self.subscription = self.create_subscription(
            DisplayTrajectory, "/display_planned_path", self.trajectory_callback, 10
        )

        self.get_logger().info("Listening to /display_planned_path...")

    def trajectory_callback(self, msg: DisplayTrajectory):
        # 每次 MoveIt 规划完成都会发布一次
        if not msg.trajectory:
            self.get_logger().warn("Received empty trajectory.")
            return

        traj = msg.trajectory[0].joint_trajectory
        self.get_logger().info(f"Received trajectory with {len(traj.points)} points.")

        for i, point in enumerate(traj.points):
            positions = [round(p, 3) for p in point.positions]
            time_from_start = (
                point.time_from_start.sec + point.time_from_start.nanosec * 1e-9
            )
            self.get_logger().info(
                f"Point {i}: positions={positions}, time={time_from_start:.3f}s"
            )


def main(args=None):
    rclpy.init(args=args)
    node = PlanListener()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
