import pybullet as p  
import time
import pybullet_data
import numpy as np
import neuralNetwork as nn
import perceptron as pp
import eas

# -------------------- Sigmoid Activation --------------------
def sigmoid(x):
    # Clip input to prevent overflow in exp
    x = np.clip(x, -709, 709)
    return 1 / (1 + np.exp(-x))

# -------------------- Fitness Function --------------------
def fitnessFunction(gene):
    """
    Evaluates how well a robot performs using a given neural network gene.
    The fitness is calculated as the sum of the x-position of the head
    while the body stays above a threshold height.
    """
    # Use DIRECT mode for faster headless simulation
    p.connect(p.DIRECT)
    p.resetSimulation()

    # Set up PyBullet environment
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.86)
    robotId = p.loadURDF("robot.urdf")   # Load robot
    planeId = p.loadURDF("plane.urdf")   # Load floor

    # Set friction for plane and robot links
    p.changeDynamics(planeId, -1, lateralFriction=1.0, rollingFriction=0.01, spinningFriction=0.01)
    for i in range(-1, p.getNumJoints(robotId)):
        p.changeDynamics(robotId, i, lateralFriction=2.0, rollingFriction=0.01, spinningFriction=0.01)

    # Initialize neural network brain with gene
    brain = pp.NeuralNet(411, [512, 256, 128, 64], 52, sigmoid)
    brain.setParams(gene)

    sum = 0
    duration = 250

    for t in range(duration):
        input = []

        # Read link states (position, velocity, orientation)
        for i in range(14):
            linkState = p.getLinkState(robotId, i, computeLinkVelocity=True)
            if linkState is None:
                input.extend([0] * 14)
                continue
            for j in range(6):
                for k in range(3 + j % 2):
                    val = linkState[j][k] if linkState[j] is not None and linkState[j][k] is not None else 0
                    input.append(val)
        
        # Read joint states (position, velocity, forces)
        for i in range(13):
            jointState = p.getJointState(robotId, i)
            if jointState is None:
                input.extend([0] * 12)
                continue
            for j in range(4):
                if j == 2:
                    # Force and torque array
                    if jointState[j] is not None:
                        for k in range(6):
                            val = jointState[j][k] if jointState[j][k] is not None else 0
                            input.append(val)
                    else:
                        input.extend([0] * 6)
                else:
                    val = jointState[j] if jointState[j] is not None else 0
                    input.append(val)

        # Pad or truncate input to 411 elements
        while len(input) < 411:
            input.append(0)
        input = input[:411]

        # Feedforward the neural network
        brain.forward(input)

        # Assign outputs to joint controllers (a1-d13 pattern)
        # aX: amplitude, bX: frequency, cX: phase, dX: offset
        a1, b1, c1, d1 = brain.output[0:4]
        a2, b2, c2, d2 = brain.output[4:8]
        a3, b3, c3, d3 = brain.output[8:12]
        a4, b4, c4, d4 = brain.output[12:16]
        a5, b5, c5, d5 = brain.output[16:20]
        a6, b6, c6, d6 = brain.output[20:24]
        a7, b7, c7, d7 = brain.output[24:28]
        a8, b8, c8, d8 = brain.output[28:32]
        a9, b9, c9, d9 = brain.output[32:36]
        a10, b10, c10, d10 = brain.output[36:40]
        a11, b11, c11, d11 = brain.output[40:44]
        a12, b12, c12, d12 = brain.output[44:48]
        a13, b13, c13, d13 = brain.output[48:52]

        # Convert outputs to joint target positions using sinusoidal controllers
        targetThigh1 = a1*np.sin(b1*t + c1) + d1
        targetThigh2 = -a2*np.sin(b2*t + c2) + d2
        targetLeg1 = -a3*np.sin(b3*t + c3) + d3
        targetLeg2 = a4*np.sin(b4*t + c4) + d4
        targetFoot1 = a5*np.sin(b5*t + c5) + d5
        targetFoot2 = -a6*np.sin(b6*t + c6) + d6
        targetArm1 = -a7*np.sin(b7*t + c7) + d7
        targetArm2 = a8*np.sin(b8*t + c8) + d8
        targetForearm1 = a9*np.sin(b9*t + c9) + d9
        targetForearm2 = -a10*np.sin(b10*t + c10) + d10
        targetHand1 = -a11*np.sin(b11*t + c11) + d11
        targetHand2 = a12*np.sin(b12*t + c12) + d12
        targetHead = a13*np.sin(b13*t + c13) + d13

        # Apply target positions to joints using POSITION_CONTROL
        p.setJointMotorControl2(robotId, 0, p.POSITION_CONTROL, targetPosition=targetThigh1, force=50)
        p.setJointMotorControl2(robotId, 1, p.POSITION_CONTROL, targetPosition=targetLeg1, force=100)
        p.setJointMotorControl2(robotId, 2, p.POSITION_CONTROL, targetPosition=targetFoot1, force=50)
        p.setJointMotorControl2(robotId, 3, p.POSITION_CONTROL, targetPosition=targetThigh2, force=50)
        p.setJointMotorControl2(robotId, 4, p.POSITION_CONTROL, targetPosition=targetLeg2, force=100)
        p.setJointMotorControl2(robotId, 5, p.POSITION_CONTROL, targetPosition=targetFoot2, force=50)
        p.setJointMotorControl2(robotId, 6, p.POSITION_CONTROL, targetPosition=targetArm1, force=100)
        p.setJointMotorControl2(robotId, 7, p.POSITION_CONTROL, targetPosition=targetForearm1, force=50)
        p.setJointMotorControl2(robotId, 8, p.POSITION_CONTROL, targetPosition=targetHand1, force=50)
        p.setJointMotorControl2(robotId, 9, p.POSITION_CONTROL, targetPosition=targetArm2, force=100)
        p.setJointMotorControl2(robotId, 10, p.POSITION_CONTROL, targetPosition=targetForearm2, force=50)
        p.setJointMotorControl2(robotId, 11, p.POSITION_CONTROL, targetPosition=targetHand2, force=50)
        p.setJointMotorControl2(robotId, 12, p.POSITION_CONTROL, targetPosition=targetHead, force=25)

        # Step the simulation
        p.stepSimulation()

        # Measure fitness based on body height and head position
        zBody = p.getLinkState(robotId, 0)[0][2] if p.getLinkState(robotId, 0) is not None else 0
        xHead = p.getLinkState(robotId, 13)[0][0] if p.getLinkState(robotId, 13) is not None else 0

        if zBody >= 3:
            sum += xHead

    p.disconnect()
    return sum / duration

# -------------------- Test Robot with GUI --------------------
def testRobot():
    gene = np.load(f"gene.npy")  # Load best gene
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.86)
    robotId = p.loadURDF("robot.urdf")
    p.loadURDF("plane.urdf")

    # Initialize neural network brain
    brain = pp.NeuralNet(411, [512, 256, 128, 64], 52, sigmoid)
    brain.setParams(gene)

    duration = 2000
    for m in range(duration):
        t = m / 60  # simulation time in seconds

        # Collect link and joint states as inputs
        input = []
        # Same as fitnessFunction: read positions, velocities, forces, pad/truncate
        # ...
        # (Skipped for brevity; identical to fitnessFunction input collection)

        brain.forward(input)

        # Compute target positions and apply them
        # (Same as fitnessFunction)
        # ...

        p.stepSimulation()
        time.sleep(1 / 60)

# -------------------- Run Evolutionary Algorithm --------------------
def runRobot():
    # Network configuration
    inputs = 411
    hidden = [512, 256, 128, 64]
    outputs = 52

    # Compute total number of gene parameters
    geneSize = 0
    for i in range(len(hidden)):
        if i == 0:
            geneSize += hidden[i]*(inputs + 1)  # +1 for bias
        else:
            geneSize += hidden[i]*(hidden[i - 1] + 1)
    geneSize += outputs*(hidden[-1] + 1)

    # Define parameter bounds for EA
    paramB = 2
    barriers = [[-paramB, paramB] for _ in range(geneSize)]
    mutationProbability = 0.5
    recombinationProbability = 0.1
    generations = 10
    population = 20

    # Create and run the evolutionary algorithm
    robot = eas.EvolutionaryAlgorithm(
        fitnessFunction, geneSize, barriers, 
        mutationProbability, recombinationProbability, generations, population
    )
    robot.run()

    # Test best-performing robot
    testRobot()

# Execute the full run
runRobot()
