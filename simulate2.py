import pybullet as p 
import time
import pybullet_data
import numpy as np
import neuralNetwork as nn
import perceptron as pp
import eas

# -------------------- Sigmoid Activation --------------------
def sigmoid(x):
    # Clip input to prevent overflow in np.exp
    x = np.clip(x, -709, 709)
    return 1 / (1 + np.exp(-x))

# -------------------- Fitness Function --------------------
def fitnessFunction(gene):
    """
    Evaluates worm locomotion performance for a given neural network gene.
    Fitness is based on the displacement of the middle segment, considering
    head-tail orientation.
    """
    p.connect(p.DIRECT)  # Headless simulation
    p.resetSimulation()

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.86)

    # Load robot and ground plane
    robotId = p.loadURDF("worm.urdf")
    planeId = p.loadURDF("plane.urdf")

    # Friction for plane
    p.changeDynamics(planeId, -1, lateralFriction=1.0, rollingFriction=0.01, spinningFriction=0.01)

    # Friction for robot links
    for i in range(-1, p.getNumJoints(robotId)):
        p.changeDynamics(robotId, i, lateralFriction=2.0, rollingFriction=0.001, spinningFriction=0.001)

    # Initialize neural network
    brain = pp.NeuralNet(201, [256, 128, 64, 32], 24, sigmoid)
    brain.setParams(gene)

    duration = 1000
    for t in range(duration):
        input = []

        # Collect link states: positions, velocities, orientations
        for i in range(7):
            linkState = p.getLinkState(robotId, i, computeLinkVelocity=True)
            if linkState is None:
                input.extend([0] * 14)
                continue
            for j in range(6):
                for k in range(3 + j % 2):
                    val = linkState[j][k] if linkState[j] is not None and linkState[j][k] is not None else 0
                    input.append(val)

        # Collect joint states: positions, velocities, torques
        for i in range(6):
            jointState = p.getJointState(robotId, i)
            if jointState is None:
                input.extend([0] * 12)
                continue
            for j in range(4):
                if j == 2:  # Torque array
                    if jointState[j] is not None:
                        for k in range(6):
                            val = jointState[j][k] if jointState[j][k] is not None else 0
                            input.append(val)
                    else:
                        input.extend([0] * 6)
                else:
                    val = jointState[j] if jointState[j] is not None else 0
                    input.append(val)

        # Ensure fixed input size
        while len(input) < 201:
            input.append(0)
        input = input[:201]

        # Feedforward through neural network
        brain.forward(input)

        # Control each of the 6 joints with sinusoidal function
        for i in range(6):
            targetPos = brain.output[4*i] * np.sin(brain.output[4*i+1]*t + brain.output[4*i+2]) + brain.output[4*i+3]
            p.setJointMotorControl2(robotId, i, p.POSITION_CONTROL, targetPosition=targetPos, force=50)

        p.stepSimulation()

    # Compute fitness based on head, middle, and tail x-positions
    xHead, xMid, xTail = 0, 0, 0
    headState = p.getLinkState(robotId, 0)
    midState  = p.getLinkState(robotId, 3)
    tailState = p.getLinkState(robotId, 6)

    if headState is not None:
        xHead = headState[0][0]
    if midState is not None:
        xMid = midState[0][0]
    if tailState is not None:
        xTail = tailState[0][0]

    # Penalize backward movement
    fitness = -xMid if xHead - xTail <= 0 else xMid

    p.disconnect()
    return fitness

# -------------------- Test Robot --------------------
def testRobot():
    """
    Runs the worm robot in GUI mode using the best neural network gene.
    """
    gene = np.load("gene.npy")  # Load best gene
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.86)

    robotId = p.loadURDF("worm.urdf")
    planeId = p.loadURDF("plane.urdf")

    # Friction for plane and robot links
    p.changeDynamics(planeId, -1, lateralFriction=1.0, rollingFriction=0.01, spinningFriction=0.01)
    for i in range(-1, p.getNumJoints(robotId)):
        p.changeDynamics(robotId, i, lateralFriction=2.0, rollingFriction=0.01, spinningFriction=0.01)

    brain = pp.NeuralNet(201, [256, 128, 64, 32], 24, sigmoid)
    brain.setParams(gene)

    duration = 1000
    for m in range(duration):
        t = m / 60

        input = []

        # Same input collection as fitnessFunction
        for i in range(7):
            linkState = p.getLinkState(robotId, i, computeLinkVelocity=True)
            if linkState is None:
                input.extend([0] * 14)
                continue
            for j in range(6):
                for k in range(3 + j % 2):
                    val = linkState[j][k] if linkState[j] is not None and linkState[j][k] is not None else 0
                    input.append(val)
        
        for i in range(6):
            jointState = p.getJointState(robotId, i)
            if jointState is None:
                input.extend([0] * 12)
                continue
            for j in range(4):
                if j == 2:
                    if jointState[j] is not None:
                        for k in range(6):
                            val = jointState[j][k] if jointState[j][k] is not None else 0
                            input.append(val)
                    else:
                        input.extend([0] * 6)
                else:
                    val = jointState[j] if jointState[j] is not None else 0
                    input.append(val)

        while len(input) < 201:
            input.append(0)
        input = input[:201]

        brain.forward(input)

        # Apply joint commands
        for i in range(6):
            targetPos = brain.output[4*i] * np.sin(brain.output[4*i+1]*t + brain.output[4*i+2]) + brain.output[4*i+3]
            p.setJointMotorControl2(robotId, i, p.POSITION_CONTROL, targetPosition=targetPos, force=50)

        p.stepSimulation()
        time.sleep(1 / 60)

    # Evaluate fitness visually
    xHead, xMid, xTail = 0, 0, 0
    headState = p.getLinkState(robotId, 0)
    midState  = p.getLinkState(robotId, 3)
    tailState = p.getLinkState(robotId, 6)

    if headState is not None:
        xHead = headState[0][0]
    if midState is not None:
        xMid = midState[0][0]
    if tailState is not None:
        xTail = tailState[0][0]

    fitness = -xMid if xHead - xTail <= 0 else xMid
    p.disconnect()
    print(f"Fitness: {fitness}")

# -------------------- Run Evolutionary Algorithm --------------------
def runRobot():
    inputs = 201
    hidden = [256, 128, 64, 32]
    outputs = 24

    # Compute total gene size
    geneSize = sum((hidden[i] * (inputs if i == 0 else hidden[i-1] + 1) for i in range(len(hidden)))) + outputs * (hidden[-1] + 1)
    
    # Parameter bounds
    paramB = 10
    barriers = [[-paramB, paramB] for _ in range(geneSize)]

    # EA settings
    mutationProbability = 0.5
    recombinationProbability = 0.5
    generations = 2
    population = 2

    # Run EA
    worm = eas.EvolutionaryAlgorithm(
        fitnessFunction, geneSize, barriers,
        mutationProbability, recombinationProbability,
        generations, population
    )
    worm.run()

    testRobot()

# -------------------- Benchmark Simulation --------------------
for mode in [p.GUI, p.DIRECT]:
    cid = p.connect(mode)
    p.resetSimulation()
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf")
    p.loadURDF("r2d2.urdf", [0, 0, 1])

    start = time.time()
    for _ in range(10000):
        p.stepSimulation()
    print("Mode:", "GUI" if mode == p.GUI else "DIRECT",
          " | Time:", round(time.time() - start, 3), "s")
    p.disconnect()

# runRobot()
testRobot()