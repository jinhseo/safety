# safety
git clone https://github.com/jinhseo/safety.git
```bash
source devel/setup.zsh  
cd src/launch  
roslaunch safety_imcar.launch  
```
Define target waypoint via rviz.  
* safety/drive/speed : target speed (30/10/0)
* safety/drive/jerk  : lane change  (0 = False, 1 = True) 
