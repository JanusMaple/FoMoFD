from setuptools import find_packages, setup

package_name = 'real_dvrk_bridge'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='anonymous',
    maintainer_email='maintainer@example.invalid',
    description='ROS 2 bridge for synchronized dVRK robot state, actions, and camera observations',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            "real_psm_bridge = real_dvrk_bridge.real_psm_bridge:main",
        ],
    },
)
