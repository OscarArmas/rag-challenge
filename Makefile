# Python Interpreter
PYTHON_INTERPRETER = python3

#################################################################################
#                                COMMANDS                                       #
#################################################################################

#-----------------------#
# Code Style            #
#-----------------------#

codestyle:
	$(PYTHON_INTERPRETER) -m isort --settings-path pyproject.toml ./
	$(PYTHON_INTERPRETER) -m black --config pyproject.toml --exclude examples ./
