import os
from pydantic_ai import Agent
from pydantic_ai.models.groq import GroqModel
from pydantic import BaseModel
from typing import Union
from tools import (
    get_robot_posture_tool,
    move_robot_tool,
    get_system_report_tool,
    capture_camera_image_tool
)

class RobotStatusResult(BaseModel):
    status: str
    message: str

class RobotCommandResult(BaseModel):
    status: str
    message: str

class SystemReportResult(BaseModel):
    status: str
    report: str

class ImageResult(BaseModel):
    status: str
    report: str

AgentOutput = Union[
    RobotStatusResult,
    RobotCommandResult,
    SystemReportResult,
    ImageResult
]

os.environ["GROQ_API_KEY"] = "......"  # place the API key

groq_model = GroqModel(model_name="llama-3.3-70b-versatile")

agent = Agent(
    model=groq_model,
    output_type=AgentOutput,
    tools=[
        get_robot_posture_tool,
        move_robot_tool,
        get_system_report_tool,
        capture_camera_image_tool
    ],
    system_prompt=(
        "You are a helpful robot assistant.:\n"
        "- If you are asked to change the posture, e.g., fall down, immediately call robot posture tool and respond whether you need to change the posture\n"
        "- If you are asked about system report, immediately read the full system report first, then briefly describe the system report with, especially CPU use, Memory, temperature and overal health).\n\n"
        "The system report is the following:\n"
        "CPU Status: \n"
        "Memory Status: \n"
        "Battery Status: \n"
        "Overwall Health: \n"
        "- If you are asked to move or rotate, immediately, send the motion command\n"
        "- When the user asks anything related to an image, call the `capture_camera_image_tool` tool immediately.\n\n"
        "Describe the relative distance and orentation of the objects you observe in the image with respec to the robot, e.g.,\n"
        "An object x is two meter away and at positive 30 degrees from the robot, ..."
    )
)

print("\n🤖 Robot Assistant Ready. Type 'exit' to quit.\n")
while True:
    user_input = input("🧑 You: ").strip()
    if user_input.lower() in {"exit", "quit"}:
        print("👋 Goodbye!")
        break
    result = agent.run_sync(user_input)
    output = result.output
    print("🤖 Robot:", output)