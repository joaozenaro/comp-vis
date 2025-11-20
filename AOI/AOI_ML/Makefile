TARGET = main
BUILD_DIR = build

default: clean build run

build:
	mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && cmake .. && $(MAKE)

run:
	./$(BUILD_DIR)/$(TARGET) ../x64/Debug/data/test.pgm

clean:
	rm -rf $(BUILD_DIR)
