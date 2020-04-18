from setuptools import setup, find_packages

setup(
    name="visionalert",
    version="0.0.1",
    author="Tyler Frederick",
    author_email="tyler@tylerfrederick.com",
    description=(
        "Monitors IP cameras and provides notifications "
        "when certain objects are detected"
    ),
    license="AGPL3",
    packages=['visionalert'],
    package_dir={"": "src"},
    entry_points={"console_scripts": ["visionalert = visionalert.app:run"]},
)
