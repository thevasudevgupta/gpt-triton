checkdirs := .

style:
	black $(checkdirs)
	isort $(checkdirs)

install:
	pip install jaxtyping
	pip install git+https://github.com/Deep-Learning-Profiling-Tools/triton-viz@v1
	wget "https://dl.cloudsmith.io/public/test-wha/triton-puzzles/raw/files/triton-3.0.0-cp310-cp310-linux_x86_64.whl"
	pip install triton-3.0.0-cp310-cp310-linux_x86_64.whl
	# export LC_ALL="en_US.UTF-8"
	# export LD_LIBRARY_PATH="/usr/lib64-nvidia"
	# export LIBRARY_PATH="/usr/local/cuda/lib64/stubs"
	# ldconfig /usr/lib64-nvidia
	pip3 install -r requirements.txt
