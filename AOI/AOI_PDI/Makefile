TARGET = main
BUILD_DIR = build

default: clean build run

build:
	mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && cmake .. && $(MAKE)

run:
	./$(BUILD_DIR)/$(TARGET) ../x64/Debug/test_noise.pgm ../x64/Debug/light.pgm -lightMethod=0 -segMethod=2

clean:
	rm -rf $(BUILD_DIR)
