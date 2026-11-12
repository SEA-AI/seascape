# Sources

- ROS REP 103, *Standard Units of Measure and Coordinate Conventions*: body frames forward-left-up, world frames east-north-up, a `_ned` suffix for frames that differ: https://reps.openrobotics.org/rep-0103/
- ROS REP 105, *Coordinate Frames for Mobile Platforms*: named frames and the chain between them: https://www.ros.org/reps/rep-0105.html
- Paul Furgale, *Representing Robot Pose: The good, the bad, and the ugly*, 2014: draw the frame diagram, name every transform by its frames, document the frames each function uses: https://rpg.ifi.uzh.ch/docs/teaching/2024/FurgaleTutorial.pdf
- Drake, monogram notation: `X_AB` is frame B's pose in A, and products check by matching letters, `X_WC = X_WA * X_AB * X_BC`: https://drake.mit.edu/doxygen_cxx/group__multibody__notation.html
- *A Standard Rigid Transformation Notation Convention for Robotics Research*, 2024: https://arxiv.org/abs/2405.07351
- tf2's terminology: source and target against parent and child, and why they need not line up: https://docs.ros.org/en/humble/p/tf2/
- ROS REP 120, *Coordinate Frames for Humanoid Robots*: `base_footprint`, the body's origin dropped onto the ground with its heading only: https://www.ros.org/reps/rep-0120.html
- Thor I. Fossen, *Handbook of Marine Craft Hydrodynamics and Motion Control*, 2nd ed., 2021, §2.1: the NED frame {n}, the body frame {b} at a fixed point of the hull, and the ZYX attitude (ψ, θ, φ) with SNAME (1950) signs.
- `flir_ptu_description`, the URDF for FLIR pan-tilt units: a base link, then pan, tilt and mount links: https://github.com/ros-drivers/flir_ptu
- FLIR, *E Series Pan-Tilt Command Reference Manual* (covers the PTU-5), 2017, §4.1 and §8.1: positions count from the home that reset calibration finds, and pan is positive to the right.
- Signal K, `navigation` schema: attitude roll positive to starboard, pitch positive bow up, yaw positive to starboard, matching NMEA 2000 PGN 127257: https://github.com/SignalK/specification
