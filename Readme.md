# PPO-Pytorch
This is a well-written and easy-to-follow implementation of Proximal Policy Optimization algortihm, please refer to original paper
from UCB: https://arxiv.org/abs/1707.06347



# Dependencies

## Anaconda & Python3 & Pytorch
please use python3 to run my code.

## RoboSchool
`RoboSchool` is a open-source RL simulation platform to replicate Gym MuJoCo environment, which need license to use.
for detailed instruction, please refer to official github: https://github.com/openai/roboschool
here list simplified instruction.
- 1. set `ROBOSCHOOL` for following `bullet3` installation
```
git clone https://github.com/openai/roboschool.git
cd roboschool
export ROBOSCHOOL_PATH=`pwd`
```

- 2. install `bullet3`
`bullet3` will be installed into your `ROBOSCHOOL` folder as `roboschool` need it as 3nd library.
```
git clone https://github.com/olegklimov/bullet3 -b roboschool_self_collision
mkdir bullet3/build
cd    bullet3/build
cmake -DBUILD_SHARED_LIBS=ON -DUSE_DOUBLE_PRECISION=1 -DCMAKE_INSTALL_PREFIX:PATH=$ROBOSCHOOL_PATH/roboschool/cpp-household/bullet_local_install -DBUILD_CPU_DEMOS=OFF -DBUILD_BULLET2_DEMOS=OFF -DBUILD_EXTRAS=OFF  -DBUILD_UNIT_TESTS=OFF -DBUILD_CLSOCKET=OFF -DBUILD_ENET=OFF -DBUILD_OPENGL3_DEMOS=OFF ..
make -j8
make install
```

- 3. back to `roboschool` root directory
make sure pip use python3.
```
pip install -e $ROBOSCHOOL_PATH
```
- if error:
>If you see compilation error FIRST THING TO CHECK if pkg-config call was successful.
    Install dependencies that pkg-config cannot find.

occures, please `export PKG_CONFIG_PATH=/path/to/your/python/lib/pkgconfig`, for anaconda user, just 
```
export PKG_CONFIG_PATH=/path/to/conda/lib/pkgconfig:$PKG_CONFIG_PATH

```
and then `pip install -e $ROBOSCHOOL_PATH`. I have tested it. for more details, refer to : https://github.com/openai/roboschool/issues/94

- if error:
> QGLShaderProgram: could not create shader program
Could not create shader of type 2.
python: render-simple.cpp:250: void SimpleRender::Context::initGL(): Assertion `r0' failed.

I searched Google and find some avaliable solution, but this will solve it by simply add one line:
```
from OpenGL import GLU
```
to the beginning of the python main source code.
pls refer to : https://github.com/openai/roboschool/issues/8 for more details.

- 4. verify its installation.

```
python $ROBOSCHOOL_PATH/agent_zoo/RoboschoolHumanoidFlagrun_v0_2017may.py

```




# HowTO

Step 1. set OMP_NUM_THREADS=1, otherwise it will block multiprocessing threads.
```
export OMP_NUM_THREADS=1
```

Step 2. run
```
python train.py
```



# Hopper
![Hopper](res/Hopper.gif)
