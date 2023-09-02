# IDATT2502 -  Applied machine learning with project

## Course content
Data representation: representation of various data sources such as images, sound and text, current techniques for processing data.

Unsupervised learning: various clustering algorithms, reduction of dimensions, and other current methods.

Supervised learning: including logistic regression and different types of neural networks.

## Learning outcome
### Knowledge

The candidate can give an account of:

- different ways of representing data
- different methods for grouping and classifying data
- which machine learning methods are appropriate to use for given problems
- limitations of machine learning

### Skills

The candidate can:

- create full-fledged machine learning solutions using a framework
- use representation algorithms that make it easier for machine learning methods to give better results for a given data set
- select and adapt a machine learning method that is relevant to a given problem
- assess whether machine learning methods can give good results for a given problem based on a given data set

### General competence

The candidate must be able to find and adapt solutions to new problems based on previous applications of machine learning.


## Prerequisites

- **Python:**
  
  Ensure that you have Python installed on your machine. You can download and install Python from the [official Python site](https://www.python.org).
  
- **Pip:**
  
  Check if Pip is installed. Pip is typically installed alongside Python. You can verify its installation by running the command ```pip --version``` in your terminal or command prompt.
  
- **Make:**

  The Makefile approach is used for installing and running tasks. Make is a build automation tool commonly found on Unix-like operating systems. You can verify its installation by running the command ```make --version``` in your terminal or command prompt.


## Installation and Running with Makefile

To install and run tasks using the Makefile approach, follow these steps:

- **Step 1: Clone or Download Repository**
  1. Clone or download the repository to your local machine.

- **Step 2: Navigate to Cloned Folder**
  1. Open a terminal window.
  2. Navigate to the cloned repository folder.

- **Step 3: Follow the Instructions**
  1. Inside the terminal, follow the instructions step by step.

### Running tasks

To run specific tasks within the task folder, use these steps:

- **Step 1: Navigating to the Task Folder**
  1. Open a terminal window.
  2. Navigate to the specific folder (e.g., `01`) using the command:
     ```bash
     cd 01
     ```

- **Step 2: Installing Dependencies**
  1. Inside the folder, install required packages from `requirements.txt`:
     ```bash
     make install
     ```

- **Step 3: Running tasks using Makefile**
  1. Run a specific task (e.g., task A) using the command:
     ```bash
     make run_task folder=A
     ```
     Replace `A` with the task identifier.

- **Step 4: Cleaning Up**
  1. To remove virtual environment and installed packages, run:
     ```bash
     make clean
     ```

These steps should help organize and present the installation and running process in a clear and user-friendly manner.

