import os
import asyncio
import json
from typing import Optional, Literal
import base64
from nats.aio.client import Client as NATS
from pydantic_ai.tools import Tool
from groq import Groq
from pathlib import Path

os.environ["GROQ_API_KEY"] = "...."  # # place the API key

# Initialize Groq client
client = Groq()

# -----------------------------------------------------------
# Tool 1: Get robot posture
# -----------------------------------------------------------

async def get_robot_posture(query: Optional[str] = "Get the current posture of the robot.") -> dict:
    try:
        nc = NATS()
        await nc.connect()
        response = await nc.request("robot.status.request", b"status?", timeout=1)
        await nc.close()
        return {"status": "success", "message": response.data.decode()}
    except Exception as e:
        return {"status": "error", "message": f"Failed to retrieve robot status: {e}"}

get_robot_posture_tool = Tool(
    name="get_robot_posture",
    description="Retrieve the robot's current posture from the robot simulator.",
    function=get_robot_posture
)

# -----------------------------------------------------------
# Tool 2: Move robot
# -----------------------------------------------------------

async def move_robot(
    move_direction: Literal["forward", "backward"],
    move_distance: float,
    rotate_direction: Literal["left", "right"],
    rotate_angle: float
) -> dict:
    command = {
        "Move": {"direction": move_direction, "distance": move_distance},
        "Rotate": {"direction": rotate_direction, "angle": rotate_angle}
    }

    try:
        nc = NATS()
        await nc.connect()
        await nc.publish("robot.command.execute", json.dumps(command).encode())
        await nc.flush()
        await nc.close()
        return {"status": "success", "message": f"Command sent: {command}"}
    except Exception as e:
        return {"status": "error", "message": f"Failed to send move command: {e}"}

move_robot_tool = Tool(
    name="move_robot",
    description="Send a movement command to the robot (forward/backward + rotation).",
    function=move_robot
)

# -----------------------------------------------------------
# Tool 3: Get system report
# -----------------------------------------------------------

async def get_system_report(note: Optional[str] = "Read the system report and explain briefly.") -> dict:
    try:
        nc = NATS()
        await nc.connect()
        response = await nc.request("robot.system.report", b"report?", timeout=5)
        await nc.close()
        return {"status": "success", "report": response.data.decode()}
    except Exception as e:
        return {"status": "error", "report": f"Failed to get system report: {e}"}

get_system_report_tool = Tool(
    name="get_system_report",
    description="Get a full system report (battery, CPU, health, temperature, etc.).",
    function=get_system_report
)

# -----------------------------------------------------------
# Tool 4: Capture camera image (Groq llama-4 vision)
# -----------------------------------------------------------

async def capture_camera_image(query: Optional[str] = "What do you see in this image?") -> dict:
    try:
        nc = NATS()
        await nc.connect()
        response = await nc.request("robot.camera.capture", b"image?", timeout=5)
        await nc.close()

        image_bytes = response.data
        image_path = "captured_image.jpg"
        with open(image_path, "wb") as f:
            f.write(image_bytes)

        img_path = Path(image_path)
        if not img_path.exists():
            return {"status": "error", "message": "Image file not found"}

        encoded_image = base64.b64encode(img_path.read_bytes()).decode("utf-8")

        # Groq vision model
        chat_response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": query},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded_image}"
                        }}
                    ]
                }
            ]
        )

        return {"status": "success", "report": chat_response.choices[0].message.content}

    except Exception as e:
        return {"status": "error", "message": f"Failed to analyze image: {e}"}

capture_camera_image_tool = Tool(
    name="capture_camera_image",
    description="Capture an image from the robot's camera and analyze it using vision model.",
    function=capture_camera_image
)