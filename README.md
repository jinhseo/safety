# safety
git clone https://github.com/jinhseo/safety.git
```bash
source devel/setup.zsh  
cd src/launch  
roslaunch safety_imcar.launch  
```
Define target waypoint via rviz.  
* safety/drive/speed : Target speed (Max = 30)
* safety/drive/jerk  : Lane change  (0 = False, 1 = True)

