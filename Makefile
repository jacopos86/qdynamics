# ===================
#  Load config
# ===================

include config.mk

# ===================
#  export variables
#  Python
# ===================

export ROOT
export RUN_TESTS
export LOG_LEVEL
export COLOR_LOG
export LOG_FILE

# ========================
# Base conda dependencies
# ========================

CONDA_BASE_DEPS = \
  $(PYTHON_VERSION) \
  numpy \
  scipy \
  matplotlib \
  mpi4py \
  h5py \
  qiskit==0.45.1 \
  qiskit-aer==0.13.3 \
  pytest \
  pip
  
# ================================
# Generate conda-environment.yml
# ================================

environment:
	@echo "Generating $(CONDA_ENV_FILE)"
	@rm -f $(CONDA_ENV_FILE)
	@echo "name: $(CONDA_ENV_NAME)" >> $(CONDA_ENV_FILE)
	@echo "channels:" >> $(CONDA_ENV_FILE)
	@echo "  - conda-forge" >> $(CONDA_ENV_FILE)
	@echo "dependencies:" >> $(CONDA_ENV_FILE)
	@for pkg in $(CONDA_BASE_DEPS); do \
		echo "  - $$pkg" >> $(CONDA_ENV_FILE); \
	done
ifeq ($(INSTALL_PSI4),1)
	@echo "  - markupsafe" >> $(CONDA_ENV_FILE)
	@echo "  - psi4" >> $(CONDA_ENV_FILE)
	@echo "  - libint=2.9.0" >> $(CONDA_ENV_FILE)
	@echo "  - basis_set_exchange" >> $(CONDA_ENV_FILE)
endif
	@echo "  - pip:" >> $(CONDA_ENV_FILE)
	@echo "      - -r $(ROOT)/dependencies/requirements.txt" >> $(CONDA_ENV_FILE)
	@echo "      - -e $(ROOT)" >> $(CONDA_ENV_FILE)

# ===================
#  configuration
# ===================

configure : $(CONDA_ENV_FILE)
	@echo "Checking conda environment $(CONDA_ENV_NAME)..."
	@if ! $(CONDA) env list | grep -qw $(CONDA_ENV_NAME); then \
		echo "Creating conda environment $(CONDA_ENV_NAME) ..."; \
		echo "$(CONDA_ENV_FILE)"; \
		$(CONDA) env create -f $(CONDA_ENV_FILE); \
	else \
		echo "Updating existing conda environment $(CONDA_ENV_NAME) ..."; \
		$(CONDA) env update -f $(CONDA_ENV_FILE) --prune; \
	fi
	
# ===================
#  install section
# ===================

install :
	@echo "Installing package into conda environment $(CONDA_ENV_NAME)"
	@$(CONDA) run -n $(CONDA_ENV_NAME) python -m pip install -e .

.PHONY : environment configure install clean prepare-tests test

# ===================
#  clean section
# ===================

clean :
	@echo "Cleaning project ..."
	# remove pycache and temporary files
	find $(ROOT) -name '__pycache__' -type d -exec rm -rf {} +
	rm -rf $(ROOT)/src/*~ ; \
	if [ -d $(ROOT)/build ] ; \
	then \
		rm -rf $(ROOT)/build ; \
	fi ; \
	# remove conda environment if it exists
	@if $(CONDA) env list | grep -qw $(CONDA_ENV_NAME); then \
		echo "Removing conda environment $(CONDA_ENV_NAME) ..."; \
		$(CONDA) env remove -n $(CONDA_ENV_NAME); \
	fi
	# remove conda-environment.yml if it exists
	@if [ -f "$(CONDA_ENV_FILE)" ]; then \
		rm -f "$(CONDA_ENV_FILE)"; \
	fi

# ===================
#  test section
# ===================

prepare-tests :
	@echo "Preparing test fixtures"
	@if [ ! -d "$(ROOT)/TESTS" ]; then \
		if [ ! -f "$(ROOT)/TESTS.tar.gz" ]; then \
			echo "Missing $(ROOT)/TESTS and $(ROOT)/TESTS.tar.gz"; \
			exit 1; \
		fi; \
		tar -xzf "$(ROOT)/TESTS.tar.gz" -C "$(ROOT)"; \
	fi
	@set -e; \
	for input_file in "$(ROOT)"/TESTS/*/input.yml; do \
		case_dir=$$(dirname "$$input_file"); \
		output_dir=$$(sed -n 's/^output_dir[[:space:]]*:[[:space:]]*//p' "$$input_file"); \
		if [ "$$output_dir" = "none" ] || [ "$$output_dir" = "null" ] || [ -z "$$output_dir" ]; then \
			if [ -d "$$case_dir/HH-OUT" ]; then \
				output_leaf="HH-OUT"; \
			elif [ -d "$$case_dir/PSI4-OUT" ]; then \
				output_leaf="PSI4-OUT"; \
			else \
				output_leaf="OUT"; \
			fi; \
		else \
			output_leaf=$$(basename "$$output_dir"); \
		fi; \
		sed -i "s|^working_dir[[:space:]]*:.*|working_dir : $$case_dir|" "$$input_file"; \
		if [ -n "$$output_leaf" ]; then \
			sed -i "s|^output_dir[[:space:]]*:.*|output_dir : $$case_dir/$$output_leaf|" "$$input_file"; \
		fi; \
	done

test : prepare-tests
	@echo "Running tests in conda environment $(CONDA_ENV_NAME)"
	@set -e; \
	PYTEST="$(CONDA) run -n $(CONDA_ENV_NAME) python -m pytest"; \
	PYDEPHASING_TESTING=1 $$PYTEST -p no:warnings $(UNIT_TEST_DIR);
