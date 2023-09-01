# Makefile for installing, running tasks, and cleaning the project

# Set the virtual environment name
VENV_NAME = myenv

# Path to the requirements file
REQUIREMENTS_FILE = requirements.txt

# Default target when running 'make' command
all: install

# Create the virtual environment, activate it, and install dependencies
install:
	@echo "Creating virtual environment..."
	@python3 -m venv $(VENV_NAME)
	@echo "Activating virtual environment..."
	@. $(VENV_NAME)/bin/activate && pip install -r $(REQUIREMENTS_FILE)
	@echo "Installation complete."

# Run the task by taking folder as parameter and running the task file in the folder. also needs to access the files in the task folder
# Needs to be able to run any .py file in the folder
run_task: install
	@echo "Running task in folder $(folder)..."
	@find $(folder) -name "*.py" -exec $(VENV_NAME)/bin/python {} \;
	@echo "Task complete."



# Clean up the virtual environment and installed packages
clean:
	@echo "Cleaning up..."
	@rm -rf $(VENV_NAME)
	@echo "Clean up complete."

.PHONY: all install run_task clean
