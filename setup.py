from setuptools import setup
import os
from glob import glob


package_name = "task_priority_kinematic_control_rqt"


setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml", "plugin.xml"]),
        (os.path.join("share", package_name, "resource"), glob("resource/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="usuario",
    maintainer_email="barajas@uji.es",
    description="RQT plugin for whole-body task-priority control.",
    license="TODO: License declaration",
    entry_points={
        "console_scripts": [],
    },
)
