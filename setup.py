from setuptools import setup

setup(name="strike_with_a_pose",
      version="0.1.0",
      url="http://github.com/airalcorn2/strike-with-a-pose",
      author="Michael A. Alcorn",
      author_email="alcorma@auburn.edu",
      packages=["strike_with_a_pose"],
      install_requires=[
          "moderngl>=5.4.2",
          "numpy>=1.15.4",
          "objloader>=0.2.0",
          "Pillow>=5.3.0",
          "PyOpenGL>=3.1.0",
          "PyOpenGL_accelerate>=3.1.0",
          "PyQt5>=5.11.3",
          "pyrr>=0.9.2",
          "scipy>=1.1.0",
          "torch>=0.4.1",
          "torchvision>=0.2.1"
      ],
      package_data={
          "strike_with_a_pose": [
              "instructions.html",
              "imagenet_classes.txt",
              "scene_files/*"
          ]
      },
      scripts=["run/strike-with-a-pose"],
      zip_safe=False)