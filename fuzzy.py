import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# Define fuzzy variables
distance = ctrl.Antecedent(np.arange(0, 101, 1), 'distance')  # Distance (meters)
speed = ctrl.Antecedent(np.arange(0, 101, 1), 'speed')        # Speed (km/h)
braking_force = ctrl.Consequent(np.arange(0, 101, 1), 'braking_force')  # Braking force (0-100%)

# Membership functions for 'distance'
distance['very_close'] = fuzz.trapmf(distance.universe, [0, 0, 10, 20])
distance['close'] = fuzz.trimf(distance.universe, [10, 30, 50])
distance['far'] = fuzz.trimf(distance.universe, [30, 60, 100])

# Membership functions for 'speed'
speed['slow'] = fuzz.trimf(speed.universe, [0, 20, 40])
speed['medium'] = fuzz.trimf(speed.universe, [30, 50, 70])
speed['fast'] = fuzz.trimf(speed.universe, [60, 80, 100])

# Membership functions for 'braking_force'
braking_force['none'] = fuzz.trimf(braking_force.universe, [0, 0, 25])
braking_force['medium'] = fuzz.trimf(braking_force.universe, [20, 50, 75])
braking_force['full'] = fuzz.trimf(braking_force.universe, [70, 100, 100])

# Define fuzzy rules
rule1 = ctrl.Rule(distance['very_close'] & speed['fast'], braking_force['full'])
rule2 = ctrl.Rule(distance['very_close'] & speed['medium'], braking_force['full'])
rule3 = ctrl.Rule(distance['very_close'] & speed['slow'], braking_force['medium'])
rule4 = ctrl.Rule(distance['close'] & speed['fast'], braking_force['full'])
rule5 = ctrl.Rule(distance['close'] & speed['medium'], braking_force['medium'])
rule6 = ctrl.Rule(distance['close'] & speed['slow'], braking_force['none'])
rule7 = ctrl.Rule(distance['far'] & speed['fast'], braking_force['medium'])
rule8 = ctrl.Rule(distance['far'] & speed['medium'], braking_force['none'])
rule9 = ctrl.Rule(distance['far'] & speed['slow'], braking_force['none'])

# Create control system and simulation
braking_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
braking_sim = ctrl.ControlSystemSimulation(braking_ctrl)

# Test the system with sample inputs
braking_sim.input['distance'] = 10  # Distance in meters
braking_sim.input['speed'] = 80     # Speed in km/h

# Compute the braking force
braking_sim.compute()

# Print the result
print(f"Braking force: {braking_sim.output['braking_force']:.2f}%")

plt.figure()
# Plot distance
distance.view(sim=braking_sim)
speed.view(sim=braking_sim)
braking_force.view(sim=braking_sim)
plt.show()


