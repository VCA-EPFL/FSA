BUILD_DIR = ./build
SA_VERILOG_FILE = $(abspath $(BUILD_DIR)/SystolicArray.sv)

SCALA_SRC = $(shell find ./src/main -name "*.scala")

DIM ?= 3
ELEM_WIDTH ?= 16
ACC_WIDTH ?= 16

.PHONY: gen_sv unit_test clean

$(SA_VERILOG_FILE): gen_sv

gen_sv: $(SCALA_SRC)
	cd ../.. && sbt \
		"project msaga" \
		"runMain msaga.sa.SVGen \
			-td $(abspath $(BUILD_DIR)) \
			--dim $(DIM) \
			--elem-width $(ELEM_WIDTH) \
			--acc-width $(ACC_WIDTH)"

unit_test: $(SA_VERILOG_FILE) $(SCALA_SRC)
	cd python && uv run unit_test/sa_test.py \
		--top-file $(SA_VERILOG_FILE) \
		--src-dir $(abspath $(BUILD_DIR)) \
		--build-dir $(abspath $(BUILD_DIR)) \
		--dim $(DIM) \
		--elem-width $(ELEM_WIDTH) \
		--acc-width $(ACC_WIDTH)

clean:
	rm -rf $(BUILD_DIR)