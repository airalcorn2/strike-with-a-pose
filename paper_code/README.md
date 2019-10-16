# Strike (with) a Pose: Neural Networks Are Easily Fooled by Strange Poses of Familiar Objects

<p align="center">

<table>
  <col width="299">
  <col width="299">
  <col width="299">
  <tr>
    <th><img src="school_bus.gif", width="299"></th>
    <th><img src="power_drill.gif", width="299"></th>
    <th><img src="school_bus_light.gif", width="299"></th>
  </tr>
  <tr>
    <td>Targeting <code>school bus</code> and only optimizing object rotation parameters.</td>
    <td>Targeting <code>power drill</code> and optimizing both rotation and translation parameters.</td>
    <td>Targeting <code>school bus</code> and only optimizing light rotation parameters.</td>
  </tr>

</table>

This repository contains the (cleaned up and improved) code for the experiments described in the paper:

> [Michael A. Alcorn](https://sites.google.com/view/michaelaalcorn), Qi Li, Zhitao Gong, Chengfei Wang, Long Mai, Wei-Shinn Ku, and [Anh Nguyen](http://anhnguyen.me). [Strike (with) a pose: Neural networks are easily fooled by strange poses of familiar objects](https://arxiv.org/abs/1811.11553). Conference on Computer Vision and Pattern Recognition (CVPR). 2019.

If you use this code for your own research, please cite:

```
@article{alcorn-2019-strike-with-a-pose,
   Author = {Alcorn, Michael A. and Li, Qi and Gong, Zhitao and Wang, Chengfei and Mai, Long and Ku, Wei-Shinn and Nguyen, Anh},
   Title = {{Strike (with) a Pose: Neural Networks Are Easily Fooled by Strange Poses of Familiar Objects}},
   Journal = {Conference on Computer Vision and Pattern Recognition (CVPR)},
   Year = {2019}
}
```

## Install Requirements

```bash
pip3 install -r requirements.txt
```

## Renderer Example

```bash
python3 renderer_example.py
```

## Optimizer Example

```bash
python3 optimizer_example.py
```

## Headless Server Set Up

Follow the steps below (modified from my GitHub issue [here](https://github.com/cprogrammer1994/Headless-rendering-with-python/issues/7#)) to enable ModernGL on a headless server with NVIDIA GPUs.
These steps borrow heavily from the CARLA headless set up guide [here](https://github.com/carla-simulator/carla/blob/master/Docs/carla_headless.md) and the configuration scripts found [here](https://github.com/agisoft-llc/cloud-scripts).
Note, [as mentioned in the CARLA guide](https://github.com/carla-simulator/carla/blob/master/Docs/carla_headless.md#-extra-), you should either not have a graphical desktop environment installed, or you need to stop it before starting X.Org servers, e.g.:

```bash
sudo service lightdm stop
```

1) Download and install the NVIDIA drivers (e.g., [like this](https://linuxconfig.org/how-to-install-the-nvidia-drivers-on-ubuntu-18-04-bionic-beaver-linux)).

2) Install OpenGL.

```bash
sudo apt install freeglut3-dev mesa-utils
```

3) Download VirtualGL from [here](https://sourceforge.net/projects/virtualgl/files/) and install it. SourceForge is kind of annoying about redirects, so you have to [use a special option with `wget`](https://stackoverflow.com/a/45258959/1316276).

```bash
wget --content-disposition -c https://sourceforge.net/projects/virtualgl/files/2.6/virtualgl_2.6_amd64.deb
sudo dpkg -i virtualgl_2.6_amd64.deb
```

4) Download TurboVNC from [here](https://sourceforge.net/projects/turbovnc/files/) and install it. Same deal with `wget`.

```bash
wget --content-disposition -c https://sourceforge.net/projects/turbovnc/files/2.2/turbovnc_2.2_amd64.deb
sudo dpkg -i turbovnc_2.2_amd64.deb
```

5) Install some X.Org stuff.

```bash
sudo apt install x11-xserver-utils libxrandr-dev
```

At this point, the CARLA guide suggests you use `nvidia-xconfig` to configure X.Org.

```bash
sudo nvidia-xconfig -a --use-display-device=None --virtual=1280x1024
```

This command is supposed to modify `/etc/X11/xorg.conf` so that things will work without a display. However, the tool [doesn't always work out of the box](https://github.com/yrahal/ec2-setup/issues/2), and it also doesn't allow you to use multiple GPUs in a fully parallel way. So what I did instead is create a separate `xorg.conf` _for each GPU_ and used those to start several independent X.Org servers.

6) To do the same, first, you need to get all of the BoardNames and BusIDs for your GPUs.

```bash
nvidia-xconfig --query-gpu-info
```

7) Next, edit [`gpu_info.json`](https://github.com/airalcorn2/strike-with-a-pose/tree/master/paper_code/gpu_info.json) so that the `BOARDNAME`s, and `BUSID`s are set appropriately for your machine.

8) Finally, run [`multi_headless_setup.py`](https://github.com/airalcorn2/strike-with-a-pose/tree/master/paper_code/multi_headless_setup.py) with superuser privileges.

```bash
sudo python3 multi_headless_setup.py
```

I only tested these steps with two different GPU models, the GeForce GTX 1080 Ti and the Tesla V100-SXM2-16GB, which use different configuration file templates. If you get an error at step 12, try changing the line:

```python
TEMPLATE_2_GPUS = {"GeForce GTX 1080 Ti"}
```

in `multi_headless_setup.py` to include your particular GPU model, and please submit a pull request if that fixes the problem.

The remaining steps are similar to the CARLA guide.

9) Edit `/etc/X11/Xwrapper.config` (requires superuser privileges) and change `allowed_users=console` to `allowed_users=anybody` so that you don't need `sudo` to start things.

10) Start an X.Org server for a particular `gpu` and assign it an ID (`xorg_server`).

```bash
nohup Xorg :${xorg_server} -config xorg.conf.${gpu} &
```

For example:

```bash
nohup Xorg :7 -config xorg.conf.0 &
```

11) Start a VNC server and assign it to a `virtual_display`.

```bash
nohup /opt/TurboVNC/bin/vncserver :${virtual_display}
```

For example:

```bash
nohup /opt/TurboVNC/bin/vncserver :8
```

12) Verify that everything's working by running the following command. It should print your OpenGL version.

```bash
export DISPLAY=:${virtual_display}
vglrun -d :${xorg_server}.0 glxinfo | grep "OpenGL version"
```

For example:

```bash
export DISPLAY=:8
vglrun -d :7.0 glxinfo | grep "OpenGL version"
```

13) Start Python with the virtual display enabled.

```bash
export DISPLAY=:${virtual_display}
vglrun -d :${xorg_server}.0 python3
```

For example:

```bash
export DISPLAY=:${virtual_display}
DISPLAY=:8 vglrun -d :7.0 python3
```

## Kill/Reset Everything

Sometimes, things get busted/weird and need to be reset. When that happens, kill all of the headless display stuff by running the following, and then go back to step 10.

```bash
pkill -9 Xorg
pkill -9 vnc
sudo rm /tmp/.X*-lock
sudo rm /tmp/.X11-unix/X*
```

## Start Several X.Org Servers on Different GPUs

First, kill everything using above commands, and then run something like the following:

```bash
export GPUS=8
export FIRST_XORG_SERVER=7

for ((gpu=0; gpu<${GPUS}; gpu++));
do
    # Start X.Org server.
    export xorg_server=$((FIRST_XORG_SERVER + gpu))
    echo ${xorg_server}
    nohup Xorg :${xorg_server} -config xorg.conf.${gpu} &

    # Start VNC server.
    export virtual_display=$((xorg_server + GPUS))
    echo ${virtual_display}
    nohup /opt/TurboVNC/bin/vncserver :${virtual_display}

    # Set DISPLAY.
    export DISPLAY=:${virtual_display}

    # Should print the OpenGL version.
    vglrun -d :${xorg_server}.0 glxinfo | grep "OpenGL version"
done
```

## Run Several Experiments on Different GPUs in Parallel

After starting several X.Org servers on different GPUs using the above commands, you can run something like the following:

```bash
for ((gpu=0; gpu<${GPUS}; gpu++));
do
    export xorg_server=$((FIRST_XORG_SERVER + gpu))
    export virtual_display=$((xorg_server + GPUS))
    export DISPLAY=:${virtual_display}

    nohup vglrun -d :${xorg_server}.0 python3 some_script.py ${gpu} ${GPUS} > ${gpu}.log &
done
```
