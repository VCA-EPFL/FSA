BUILD_DIR = ./build

SCALA_SRC = $(shell find ./src/main -name "*.scala")

DIM ?= 4
SP_ROWS ?= 128
ACC_ROWS ?= 8
ELEM_WIDTH ?= 32
ACC_WIDTH ?= 32

.PHONY: gen_systolic_array unit_test_systolic_array clean

gen_systolic_array: $(SCALA_SRC)
	cd ../.. && sbt \
		"project msaga" \
		"runMain msaga.svgen.SystolicArrayGen \
			-td $(abspath $(BUILD_DIR)/systolic_array) \
			--dim $(DIM) \
			--elem-width $(ELEM_WIDTH) \
			--acc-width $(ACC_WIDTH)"

unit_test_systolic_array: gen_systolic_array
	cd python && uv run unit_test/sa_test.py \
		--top-file $(abspath $(BUILD_DIR)/systolic_array/SystolicArray.sv) \
		--src-dir $(abspath $(BUILD_DIR)/systolic_array) \
		--build-dir $(abspath $(BUILD_DIR)/systolic_array) \
		--dim $(DIM) \
		--elem-width $(ELEM_WIDTH) \
		--acc-width $(ACC_WIDTH)

gen_msaga: $(SCALA_SRC)
	cd ../.. && sbt \
		"project msaga" \
		"runMain msaga.svgen.MSAGAGen \
			-td $(abspath $(BUILD_DIR)/msaga) \
			--dim $(DIM) \
			--sp-rows $(SP_ROWS) \
		    --acc-rows $(ACC_ROWS) \
			--elem-width $(ELEM_WIDTH) \
			--acc-width $(ACC_WIDTH)"
clean:
	rm -rf $(BUILD_DIR)