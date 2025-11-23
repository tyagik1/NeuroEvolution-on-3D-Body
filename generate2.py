import pyrosim.pyrosim as pyrosim

# Start creating a new URDF file for the worm
pyrosim.Start_URDF("worm.urdf")

# -------------------- Head --------------------
# The head of the worm
pyrosim.Send_Cube(name="Head", pos=[-3.0, 0, 0.5], size=[1.0, 0.5, 0.5])

# -------------------- Mid Sections --------------------
# First mid-section connected to head
pyrosim.Send_Cube(name="Mid1", pos=[0.5, 0.0, 0.0], size=[1.0, 0.8, 0.8])
pyrosim.Send_Joint(name="Head_Mid1", parent="Head", child="Mid1", type="revolute", position=[-2.5, 0.0, 0.5])

# Second mid-section connected to Mid1
pyrosim.Send_Cube(name="Mid2", pos=[0.5, 0.0, 0.0], size=[1.0, 1.15, 1.15])
pyrosim.Send_Joint(name="Mid1_Mid2", parent="Mid1", child="Mid2", type="revolute", position=[1.0, 0.0, 0.0])

# Third mid-section connected to Mid2
pyrosim.Send_Cube(name="Mid3", pos=[0.5, 0.0, 0.0], size=[1.0, 1.5, 1.5])
pyrosim.Send_Joint(name="Mid2_Mid3", parent="Mid2", child="Mid3", type="revolute", position=[1.0, 0.0, 0.0])

# Fourth mid-section connected to Mid3
pyrosim.Send_Cube(name="Mid4", pos=[0.5, 0.0, 0.0], size=[1.0, 1.15, 1.15])
pyrosim.Send_Joint(name="Mid3_Mid4", parent="Mid3", child="Mid4", type="revolute", position=[1.0, 0.0, 0.0])

# Fifth mid-section connected to Mid4
pyrosim.Send_Cube(name="Mid5", pos=[0.5, 0.0, 0.0], size=[1.0, 0.8, 0.8])
pyrosim.Send_Joint(name="Mid4_Mid5", parent="Mid4", child="Mid5", type="revolute", position=[1.0, 0.0, 0.0])

# -------------------- Tail --------------------
# Tail of the worm connected to Mid5
pyrosim.Send_Cube(name="Tail", pos=[0.5, 0.0, 0.0], size=[1.0, 0.5, 0.5])
pyrosim.Send_Joint(name="Mid5_Tail", parent="Mid5", child="Tail", type="revolute", position=[1.0, 0.0, 0.0])

# Finish URDF creation
pyrosim.End()