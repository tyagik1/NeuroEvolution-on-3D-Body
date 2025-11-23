import pyrosim.pyrosim as pyrosim

# Start creating a new URDF file
pyrosim.Start_URDF("robot.urdf")

# Define the main body of the robot
pyrosim.Send_Cube(name="Body", pos=[0, 0, 4.5], size=[1, 2, 3]) # Central torso

# -------------------- Leg 1 --------------------
# Upper leg (thigh) for first leg
pyrosim.Send_Cube(name="Thigh1", pos=[0, 0, -0.75], size=[0.5, 0.5, 1.5])
# Revolute joint connecting body to thigh1
pyrosim.Send_Joint(name="Body_Thigh1", parent="Body", child="Thigh1", type="revolute", position=[0.0, -0.5, 3.0])

# Lower leg for first leg
pyrosim.Send_Cube(name="Leg1", pos=[0, 0, -0.5], size=[0.2, 0.2, 1])
# Revolute joint connecting thigh1 to leg1
pyrosim.Send_Joint(name="Thigh1_Leg1", parent="Thigh1", child="Leg1", type="revolute", position=[0.0, 0.0, -1.5])

# Foot for first leg
pyrosim.Send_Cube(name="Foot1", pos=[0, 0, -0.25], size=[0.5, 0.5, 0.5])
# Revolute joint connecting leg1 to foot1
pyrosim.Send_Joint(name="Leg1_Foot1", parent="Leg1", child="Foot1", type="revolute", position=[0.0, 0.0, -1.0])

# -------------------- Leg 2 --------------------
pyrosim.Send_Cube(name="Thigh2", pos=[0, 0, -0.75], size=[0.5, 0.5, 1.5])
pyrosim.Send_Joint(name="Body_Thigh2", parent="Body", child="Thigh2", type="revolute", position=[0.0, 0.5, 3.0])

pyrosim.Send_Cube(name="Leg2", pos=[0, 0, -0.5], size=[0.2, 0.2, 1])
pyrosim.Send_Joint(name="Thigh2_Leg2", parent="Thigh2", child="Leg2", type="revolute", position=[0.0, 0.0, -1.5])

pyrosim.Send_Cube(name="Foot2", pos=[0, 0, -0.25], size=[0.5, 0.5, 0.5])
pyrosim.Send_Joint(name="Leg2_Foot2", parent="Leg2", child="Foot2", type="revolute", position=[0.0, 0.0, -1.0])

# -------------------- Arm 1 --------------------
pyrosim.Send_Cube(name="Arm1", pos=[0, -0.25, -0.5], size=[0.5, 0.5, 1])
pyrosim.Send_Joint(name="Body_Arm1", parent="Body", child="Arm1", type="revolute", position=[0.0, -1.0, 5.5])

pyrosim.Send_Cube(name="Forearm1", pos=[0, 0, -0.5], size=[0.2, 0.2, 1])
pyrosim.Send_Joint(name="Arm1_Forearm1", parent="Arm1", child="Forearm1", type="revolute", position=[0.0, -0.25, -1.0])

pyrosim.Send_Cube(name="Hand1", pos=[0, 0, -0.25], size=[0.5, 0.5, 0.5])
pyrosim.Send_Joint(name="Forearm1_Hand1", parent="Forearm1", child="Hand1", type="revolute", position=[0.0, 0.0, -1.0])

# -------------------- Arm 2 --------------------
pyrosim.Send_Cube(name="Arm2", pos=[0, 0.25, -0.5], size=[0.5, 0.5, 1])
pyrosim.Send_Joint(name="Body_Arm2", parent="Body", child="Arm2", type="revolute", position=[0.0, 1.0, 5.5])

pyrosim.Send_Cube(name="Forearm2", pos=[0, 0, -0.5], size=[0.2, 0.2, 1])
pyrosim.Send_Joint(name="Arm2_Forearm2", parent="Arm2", child="Forearm2", type="revolute", position=[0.0, 0.25, -1.0])

pyrosim.Send_Cube(name="Hand2", pos=[0, 0, -0.25], size=[0.5, 0.5, 0.5])
pyrosim.Send_Joint(name="Forearm2_Hand2", parent="Forearm2", child="Hand2", type="revolute", position=[0.0, 0.0, -1.0])

# -------------------- Head --------------------
pyrosim.Send_Cube(name="Head", pos=[0, 0, 0.5], size=[0.8, 0.8, 1])
pyrosim.Send_Joint(name="Body_Head", parent="Body", child="Head", type="revolute", position=[0, 0, 6.0])

# Finish URDF creation
pyrosim.End()
