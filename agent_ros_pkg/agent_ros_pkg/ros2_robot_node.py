#!/src/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, Image
from geometry_msgs.msg import Twist
from scipy.spatial.transform import Rotation as R
from nats.aio.client import Client as NATS
import asyncio
import json
import subprocess
import cv2
import numpy as np
from cv_bridge import CvBridge


class PostureBridge(Node):
    def __init__(self):
        super().__init__('posture_bridge')
        self.orientation = None
        self.latest_image_msg = None
        self.bridge = CvBridge()


        self.subscription = self.create_subscription(Imu, '/imu', self.imu_callback, 10)
        self.image_subscription = self.create_subscription(Image, '/camera_sensor/image_raw', self.image_callback, 10)
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)

        self.get_logger().info("PostureBridge node initialized.")

    def imu_callback(self, msg):
        self.orientation = msg.orientation

    def image_callback(self, msg):
        self.latest_image_msg = msg

    def get_posture(self):
        if self.orientation is None:
            return "Unknown"

        q = self.orientation
        r = R.from_quat([q.x, q.y, q.z, q.w])
        roll, pitch, _ = r.as_euler('xyz', degrees=True)

        roll = (roll + 180) % 360 - 180
        pitch = (pitch + 180) % 360 - 180

        if abs(roll) < 20 and abs(pitch) < 20:
            return "Upright"
        elif abs(roll) > 160 or abs(pitch) > 160:
            return "Upside Down"
        else:
            return "Tilted"

    async def execute_command(self, data: dict):
        if "Move" in data:
            direction = data["Move"].get("direction", "forward")
            distance = data["Move"].get("distance", 0.0)
            duration = distance / 0.25
            twist = Twist()
            twist.linear.x = 0.1 if direction == "forward" else -0.1
            self.get_logger().info(f"Moving {direction} for {duration:.1f}s")
            await self._publish_for_duration(twist, duration)

        if "Rotate" in data:
            direction = data["Rotate"].get("direction", "left")
            angle = data["Rotate"].get("angle", 0.0)
            duration = angle / 30.0
            twist = Twist()
            twist.angular.z = 0.5236 if direction == "left" else -0.5236
            self.get_logger().info(f"Rotating {direction} for {duration:.1f}s")
            await self._publish_for_duration(twist, duration)

    async def _publish_for_duration(self, twist_msg, duration):
        rate_hz = 10
        ticks = int(duration * rate_hz)
        interval = 1.0 / rate_hz
        for _ in range(ticks):
            self.cmd_vel_publisher.publish(twist_msg)
            await asyncio.sleep(interval)
        self.cmd_vel_publisher.publish(Twist())  # stop

    def get_system_report(self):
        try:
            result = subprocess.run(['inxi', '-Fxz'], capture_output=True, text=True, check=True)
            return result.stdout
        except FileNotFoundError:
            return "Error: 'inxi' is not installed. Install it using 'sudo apt install inxi'."
        except subprocess.CalledProcessError as e:
            return f"Command failed:\n{e.stderr}"

    def get_image_bytes(self):
        if self.latest_image_msg is None:
            return None
        try:
            # Convert ROS Image message to OpenCV image in BGR format
            cv_image = self.bridge.imgmsg_to_cv2(self.latest_image_msg, desired_encoding="bgr8")
            success, buffer = cv2.imencode(".jpg", cv_image)
            return buffer.tobytes() if success else None
        except Exception as e:
            self.get_logger().error(f"Image conversion failed: {e}")
            return None


async def bridge_main():
    rclpy.init()
    node = PostureBridge()
    nc = NATS()
    await nc.connect()

    async def status_handler(msg):
        await nc.publish(msg.reply, node.get_posture().encode())

    await nc.subscribe("robot.status.request", cb=status_handler)

    async def command_handler(msg):
        try:
            data = json.loads(msg.data.decode())
            await node.execute_command(data)
        except Exception as e:
            print(f"[ROS2] Command error: {e}")

    await nc.subscribe("robot.command.execute", cb=command_handler)

    async def report_handler(msg):
        report = node.get_system_report()
        await nc.publish(msg.reply, report.encode())

    await nc.subscribe("robot.system.report", cb=report_handler)

    async def capture_handler(msg):
        print("[ROS2] Capture image requested")
        for _ in range(20):  # wait up to 2 seconds
            if node.latest_image_msg:
                break
            await asyncio.sleep(0.1)

        img_bytes = node.get_image_bytes()
        if img_bytes:
            await nc.publish(msg.reply, img_bytes)
        else:
            await nc.publish(msg.reply, b"Error: No image available")

    await nc.subscribe("robot.camera.capture", cb=capture_handler)

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            await asyncio.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        await nc.close()
        node.destroy_node()
        rclpy.shutdown()


def main():
    asyncio.run(bridge_main())