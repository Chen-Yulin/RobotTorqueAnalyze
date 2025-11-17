## Usage
```bash
cd RobotTorqueAnalyze
colcon build --symlink-install
source ./install/local_setup.zsh

# in Terminal 1
ros2 launch xclimbot_config demo.launch.py
# in Terminal 2
ros2 run xclimbot_gravity gravity_visualize
```
