from setuptools import setup, find_packages

setup(
    name="reloc3r",
    version="0.1.0",  # Change as needed
    author="Your Name",
    author_email="your.email@example.com",
    description="A short description of what reloc3r does",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/reloc3r",  # Update if hosted elsewhere
    packages=find_packages(exclude=["tests*", "docs*"]),
    python_requires=">=3.7",
    install_requires=[
        "roma",
        "gradio",
        "matplotlib",
        "tqdm",
        "opencv-python",
        "scipy",
        "einops",
        "tensorboard",
        "pyglet<2",
        "huggingface-hub[torch]>=0.22",
        "imagesize",
        "open3d",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
)
