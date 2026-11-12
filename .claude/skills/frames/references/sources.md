# Sources

- ROS REP 103, *Standard Units of Measure and Coordinate Conventions*: body frames forward-left-up, world frames east-north-up, a `_ned` suffix for frames that differ: https://reps.openrobotics.org/rep-0103/
- ROS REP 105, *Coordinate Frames for Mobile Platforms*: named frames and the chain between them: https://www.ros.org/reps/rep-0105.html
- Paul Furgale, *Representing Robot Pose: The good, the bad, and the ugly*, 2014: draw the frame diagram, name every transform by its frames, document the frames each function uses: https://rpg.ifi.uzh.ch/docs/teaching/2024/FurgaleTutorial.pdf
- Drake, monogram notation: `X_AB` is frame B's pose in A, and products check by matching letters, `X_WC = X_WA * X_AB * X_BC`: https://drake.mit.edu/doxygen_cxx/group__multibody__notation.html
- *A Standard Rigid Transformation Notation Convention for Robotics Research*, 2024: https://arxiv.org/abs/2405.07351
- tf2's terminology: source and target against parent and child, and why they need not line up: https://docs.ros.org/en/humble/p/tf2/
