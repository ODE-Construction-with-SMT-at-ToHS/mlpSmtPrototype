**************
mlp_smt_closed
**************

.. raw:: html

       <object data="https://img.shields.io/github/license/ODE-Construction-with-SMT-at-ToHS/mlpSmtPrototype"></object>

.. inclusion-marker

General
#######
This project is a prototype for an algorithm to find a closed form for a multilayer perceptron. For a more detailed
description of the algorithm and its versions, read our `(not yet written) paper <link-to.paper>`_.

Features
########
*  create and train MLPs on a given function, depict error
*  find a closed form for the input-output relation of the MLP. *(Ideally, the learned function form the point above should be found.)*


Code Structure
##############
*  ``./docs`` contains the documentation of the code
*  ``./mlp_smt_closed`` contains the code of the project

   *  ``./mlp`` contains everything related to creating, training and saving the MLP
   *  ``./smt`` contains everything related to finding a closed form solution for a given MLP

Installation
############
Execute ``pip install -r requirements.txt`` to install all requirements.

Usage
#####
Execute ``python -m mlp_smt_closed`` in the root directory to run this project. Executing ``python -m mlp_smt_closed --help`` will print information on the usage of the package provided. For a more detailed description of the available options consider the paper or the documentation.

Documentation
#############
A Documentation of the project is hosted `here <https://ode-construction-with-smt-at-tohs.github.io/mlpSmtPrototype/>`_.
