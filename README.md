# ML Utils
Repository containing conversion scripts, command line functions and inference functions for various model types.

# Contributing Guidelines

## Code Structure and Documenation

* All code functions should have clear and concise documentation, including a description of the function's purpose, parameters, and return values.
* Each function should include example usage to demonstrate how to use the function correctly.

## Installing Black for linting

To use the automatic linting during commit, follow these steps:
1. Inside the virtual environment, install `autohooks` and `autohooks-plugin-black` if not already installed via micromamba or requirements.txt
2. Run `autohooks actiavte --mode pythonpath` to activate autohooks within the github repository.
3. Run `autohooks plugins add autohooks.plugins.black` to activate the black linting plugin for autohooks. During each commit, it will hunt for python files and automatically lint them to follow python practices.