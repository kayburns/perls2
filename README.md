# perls2
PErception and Robotic Learning System v2

perls2 provides robot software infrastructure for projects within SVL. It serves as a unified framework that connects both simualted and real robots, sensors, renders and simulation enviornments for performing robotics experiments. 

The design document describes the goals, specifications and milestones for perls2 development. 

[Design Document](https://docs.google.com/document/d/1JJA4TpnnS4lhWyXyyhaU3PcngXAHG2iap1pUcpQy9wY/edit)

New to perls2? [Start Here](https://github.com/StanfordVL/perls2/wiki/Start-Here-(Intro))

## Installing
PERLS2 only supports python3.6 for the core libraries (python 2.7 is used for interfacing to robots that use ROS.)
PERLS2 only supports ubuntu 16.04 and later.
### On Ubuntu 16.04
1. Clone and install PERLs repo
    1. Clone repo: 
        
        `git clone https://github.com/StanfordVL/perls2.git`
    2. Create a virtual environment, e.g. 
        
        `virtualenv -p python3.6 perls2env`
    3. Source your virtual environment e.g. 
        
        `source perls2env/bin/activate`

    4. Go to the perls2 directory and install requirements
        
        `cd ~/perls2`
        
        `pip install -r requirements.txt`

    5. Install perls2 
        
        `pip install -e .`
## Run PERL demos
Check out the examples in the `examples` folder [Overview](https://github.com/StanfordVL/perls2/tree/master/examples)

Intructions for each demo may be found in the README in each folder.
### Example: Run simple reach demo
`cd perls2`

`python examples/simple_reach/run_simple_reach.py`





