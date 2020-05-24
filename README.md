# ese650 Learning In Robotics project
We trained an agent in Gazebo environment using Deep Deterministic Policy Gradient(DDPG) to control turtlebot to reach the target (green circle) without any collision in a continuous action space. The result shows that agent can successfully reach random target in a 4 * 4 square avoiding unseen different shaped obstacles by taking 10-dimensional sparse range input and target position. \
Report is available at [https://github.com/lyuheng/650project/blob/master/demo/650report.pdf](https://github.com/lyuheng/650project/blob/master/demo/650report.pdf)

## How to run? ##
```
roslaunch turtlebot3_gazebo turtlebot3_stage_1.launch
roslaunch project ddpg_stage_1.launch
```

## demos ##
* pretrain on env without any obstacle (using DDPG)
<img width="250" height="250" src="https://github.com/lyuheng/650project/blob/master/demo/pretrain.gif"/>
(10 times speed up)

* fine-tunning on env with square obstacles at four corners (using DDPG)
<img width="250" height="250" src="https://github.com/lyuheng/650project/blob/master/demo/ft.gif"/>
(10 times speed up)

* evaluate agent on unseen virtual environment with different shaped obstacles at random positions
<img width="250" height="250" src="https://github.com/lyuheng/650project/blob/master/demo/obs.gif">
(10 times speed up)

* evaluate agent on a more complex unseen environment, although takes longer time
<img width="250" height="250" src="https://github.com/lyuheng/650project/blob/master/demo/complex_obs.gif">
(20 times speed up)

* evaluate agent on moving obstales with low speed
<img width="250" height="250" src="https://github.com/lyuheng/650project/blob/master/demo/moving.gif">
(10 times speed up)

## Improvent ##
* Subsitite DDPG to TD3
* Modify Actor with LSTM
