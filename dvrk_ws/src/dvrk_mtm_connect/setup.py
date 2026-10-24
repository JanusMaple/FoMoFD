from setuptools import find_packages, setup

package_name = 'dvrk_mtm_connect'

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
    description='Remote connect to dvrk MTM, read its cartesian pose and track footpedal clutch, lock it if needed',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'mtm_connect = dvrk_mtm_connect.connect_to_mtm:main',
        ],
    },
)
