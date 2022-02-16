from setuptools import setup

package_name = "ros_communication"

setup(
    name=package_name,
    version="0.0.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="chamois",
    maintainer_email="ngaitung.kwok@u.nus.edu",
    description="ROS communication with PeekingDuck implementation",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": ["listener = ros_communication.person_count_sub:main"],
    },
)
