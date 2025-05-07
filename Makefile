BUILD_DIR = ./build

SCALA_SRC = $(shell find ./src/main -name "*.scala")

DIM ?= 4
SP_ROWS ?= 256
ACC_ROWS ?= 8
MUL_EW ?= 5
MUL_MW ?= 10
ADD_EW ?= 8
ADD_MW ?= 23

.PHONY: gen_msaga clean

gen_msaga: $(SCALA_SRC)
	cd ../.. && sbt \
		"project msaga" \
		"runMain msaga.svgen.MSAGAGen \
			-td $(abspath $(BUILD_DIR)/msaga) \
			--dim $(DIM) \
			--sp-rows $(SP_ROWS) \
		    --acc-rows $(ACC_ROWS) \
			--mul-ew $(MUL_EW) \
			--mul-mw $(MUL_MW) \
			--add-ew $(ADD_EW) \
			--add-mw $(ADD_MW)"
clean:
	rm -rf $(BUILD_DIR)