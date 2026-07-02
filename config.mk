# ===================
#  Project paths
# ===================

ROOT := $(shell pwd)

# ===================
# Conda commands
# ===================

CONDA := conda
CONDA_ENV_NAME ?= pyqd
CONDA_ENV_FILE := $(ROOT)/conda-environment.yml

# ===================
# Build mode
# ===================

BUILD_MODE ?= local

# ===================
#  psi4
# ===================

INSTALL_PSI4 ?= 0

# ===================
# Python selection
# ===================

PYTHON_VERSION := python=3.11

ifeq ($(INSTALL_PSI4),1)
	PYTHON_VERSION := python=3.10
endif

# ===================
#  unit tests
# ===================

UNIT_TEST_DIR := $(ROOT)/src/unit_tests
RUN_TESTS ?= 0

# ===================
#  Log level
# ===================

LOG_LEVEL ?= INFO
COLOR_LOG ?= 1
LOG_FILE ?= out.log
