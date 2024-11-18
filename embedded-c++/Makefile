
.PHONY: duckdb clean main

all: duckdb-vss main

clean:
	rm -rf build

duckdb-vss:
	cd ../ && make release

main:
	mkdir -p build
	cd build && cmake .. && make
	build/example --benchmark_dry_run --benchmark_out=fashion_mnist_vss.txt --benchmark_out_format=console
