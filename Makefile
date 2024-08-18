checkdirs := .

style:
	black $(checkdirs)
	isort $(checkdirs)
