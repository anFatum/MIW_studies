import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

# https://pythonhosted.org/scikit-fuzzy/auto_examples/plot_tipping_problem.html#example-plot-tipping-problem-py
# Generate universe variables
#   * Quality and service on subjective ranges [0, 10]
#   * Tip has a range of [0, 25] in units of percentage points
x_speed = np.arange(0, 120, 1)
x_acceleration = np.arange(0, 20, 1)
x_way_length = np.arange(0, 300, 1)

# Generate fuzzy membership functions
speed_lo = fuzz.trimf(x_speed, [0, 0, 60])
speed_md = fuzz.trimf(x_speed, [0, 60, 120])
speed_hi = fuzz.trimf(x_speed, [60, 120, 120])
acceleration_lo = fuzz.trimf(x_acceleration, [0, 0, 10])
acceleration_md = fuzz.trimf(x_acceleration, [0, 10, 20])
acceleration_hi = fuzz.trimf(x_acceleration, [10, 20, 20])
way_length_arrived = fuzz.trimf(x_way_length, [0, 25, 70])
way_length_lo = fuzz.trimf(x_way_length, [50, 75, 120])
way_length_md = fuzz.trimf(x_way_length, [70, 150, 200])
way_length_hi = fuzz.trimf(x_way_length, [175, 250, 300])

# Visualize these universes and membership functions
fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(8, 9))

ax0.plot(x_speed, speed_lo, 'b', linewidth=1.5, label='Low')
ax0.plot(x_speed, speed_md, 'g', linewidth=1.5, label='Medium')
ax0.plot(x_speed, speed_hi, 'r', linewidth=1.5, label='High')
ax0.set_title('Speed')
ax0.legend()

ax1.plot(x_acceleration, acceleration_lo, 'b', linewidth=1.5, label='Low')
ax1.plot(x_acceleration, acceleration_md, 'g', linewidth=1.5, label='Medium')
ax1.plot(x_acceleration, acceleration_hi, 'r', linewidth=1.5, label='High')
ax1.set_title('Acceleration')
ax1.legend()

ax2.plot(x_way_length, way_length_arrived, 'm', linewidth=1.5, label='Arrived')
ax2.plot(x_way_length, way_length_lo, 'b', linewidth=1.5, label='Short')
ax2.plot(x_way_length, way_length_md, 'g', linewidth=1.5, label='Medium')
ax2.plot(x_way_length, way_length_hi, 'r', linewidth=1.5, label='Long')
ax2.set_title('Way length')
ax2.legend()

# Turn off top/right axes
for ax in (ax0, ax1, ax2):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

plt.tight_layout()
plt.show()

# We need the activation of our fuzzy membership functions at these values.
# The exact values 6.5 and 9.8 do not exist on our universes...
# This is what fuzz.interp_membership exists for!
speed_level_lo = fuzz.interp_membership(x_speed, speed_lo, 110)
speed_level_md = fuzz.interp_membership(x_speed, speed_md, 110)
speed_level_hi = fuzz.interp_membership(x_speed, speed_hi, 110)

acc_level_lo = fuzz.interp_membership(x_acceleration, acceleration_lo, 15.8)
acc_level_md = fuzz.interp_membership(x_acceleration, acceleration_md, 15.8)
acc_level_hi = fuzz.interp_membership(x_acceleration, acceleration_hi, 15.8)

# Now we take our rules and apply them. Rule 1 concerns bad food OR service.
# The OR operator means we take the maximum of these two.
# speed_level_hi || acc_level_hi -> way_length_arrived
active_rule1 = np.fmax(speed_level_hi, acc_level_hi)
way_length_activation_arrived = np.fmin(active_rule1, way_length_arrived)

# Now we apply this by clipping the top off the corresponding output
# membership function with `np.fmin`
# speed_level_hi && acc_level_md -> way_length_lo
active_rule = np.fmin(speed_level_hi, acc_level_md)
way_length_activation_lo = np.fmin(active_rule, way_length_lo)  # removed entirely to 0

# For rule 2 we connect acceptable service to medium tipping
# speed_level_hi || (speed_level_md && acc_level_lo) -> way_length_md
active_rule2 = np.fmin(speed_level_lo, acc_level_md)
active_rule2 = np.fmax(active_rule2, speed_level_md)
way_length_activation_md = np.fmin(active_rule2, way_length_md)

# For rule 3 we connect high service OR high food with high tipping
# speed_level_lo && acc_level_lo -> way_length_hi
active_rule3 = np.fmin(speed_level_lo, acc_level_lo)
way_length_activation_hi = np.fmin(active_rule3, way_length_hi)
tip0 = np.zeros_like(x_way_length)

# Visualize this
fig, ax0 = plt.subplots(figsize=(8, 4))

ax0.fill_between(x_way_length, tip0, way_length_activation_arrived, facecolor='m', alpha=0.7)
ax0.plot(x_way_length, way_length_arrived, 'b', linewidth=0.5, linestyle='--', )
ax0.fill_between(x_way_length, tip0, way_length_activation_lo, facecolor='b', alpha=0.7)
ax0.plot(x_way_length, way_length_lo, 'b', linewidth=0.5, linestyle='--', )
ax0.fill_between(x_way_length, tip0, way_length_activation_md, facecolor='g', alpha=0.7)
ax0.plot(x_way_length, way_length_md, 'g', linewidth=0.5, linestyle='--')
ax0.fill_between(x_way_length, tip0, way_length_activation_hi, facecolor='r', alpha=0.7)
ax0.plot(x_way_length, way_length_hi, 'r', linewidth=0.5, linestyle='--')
ax0.set_title('Output membership activity')

# Turn off top/right axes
for ax in (ax0,):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

plt.tight_layout()
plt.show()

# Aggregate all three output membership functions together
aggregated = np.fmax(way_length_arrived,
                     np.fmax(way_length_activation_lo,
                             np.fmax(way_length_activation_md, way_length_activation_hi)))

# Calculate defuzzified result
tip = fuzz.defuzz(x_way_length, aggregated, 'centroid')
way_length_activation = fuzz.interp_membership(x_way_length, aggregated, tip)  # for plot

# Visualize this
fig, ax0 = plt.subplots(figsize=(8, 4))

ax0.plot(x_way_length, way_length_arrived, 'm', linewidth=0.5, linestyle='--')
ax0.plot(x_way_length, way_length_lo, 'b', linewidth=0.5, linestyle='--', )
ax0.plot(x_way_length, way_length_md, 'g', linewidth=0.5, linestyle='--')
ax0.plot(x_way_length, way_length_hi, 'r', linewidth=0.5, linestyle='--')
ax0.fill_between(x_way_length, tip0, aggregated, facecolor='Orange', alpha=0.7)
ax0.plot([tip, tip], [0, way_length_activation], 'k', linewidth=1.5, alpha=0.9)
ax0.set_title('Aggregated membership and result (line)')

# Turn off top/right axes
for ax in (ax0,):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

plt.tight_layout()
plt.show()
