include makefile.local

libpyspshuff:
	g++ -g -std=c++11 -O3 -Wall -march=native -fPIC -shared `python3 -m pybind11 --includes` -I$(INCDIR) -I src src/huffman.cpp -o huffman.so -L. -L$(LIBDIR) -L/usr/lib $(PYTHON_LINK) -lcrypt -lpthread -ldl -lutil -lm

test:
	g++ -std=c++11 -O3 -Wall -fno-omit-frame-pointer -march=native -Isrc -I$(INCDIR) -L. -L$(LIBDIR) tests/test_chunked_encode.cpp -o test_chunked_encode -lm
	g++ -std=c++11 -O3 -Wall -fno-omit-frame-pointer -march=native -Isrc -I$(INCDIR) -L. -L$(LIBDIR) tests/test_quantization_simd.cpp -o test_quantization_simd -lm
	g++ -std=c++11 -O3 -Wall -fno-omit-frame-pointer -march=native -Isrc -I$(INCDIR) -L. -L$(LIBDIR) tests/test_n5_encoder.cpp -o test_n5_encoder -lm
	g++ -std=c++11 -O3 -Wall -fno-omit-frame-pointer -march=native -Isrc -I$(INCDIR) -L. -L$(LIBDIR) tests/time_n5_encoder.cpp -o time_n5_encoder -lm

install-cpp:
	cp src/spshuff.hpp $(INCDIR)

# ignore
# SPSPATH=$(PYTHON_SITE)/spshuff-1.1-py3.7-linux-x86_64.egg
# SPSPATH=$(INSTALL_PYTHON_PATH)/spshuff-py3.7.egg

# depricated
# install-python-egg: install-cpp libpyspshuff
# 	mkdir -p $(SPSPATH)
# 	rsync -a fake_egg/* $(SPSPATH)
# 	rsync -a spshuff $(SPSPATH)
# 	rsync -a huffman.so $(SPSPATH)/spshuff

install-python: install-cpp libpyspshuff
	rm -rf $(PYTHON_SITE)/spshuff/*
	rsync -a spshuff $(PYTHON_SITE)
	rsync -a huffman.so $(PYTHON_SITE)/spshuff

install: install-cpp install-python

# install-python-oneliner: libpy-oneliner
# 	mkdir -p $(SPSPATH)
# 	rsync -a fake_egg/* $(SPSPATH)
# 	rsync -a spshuff $(SPSPATH)
# 	rsync -a pyspshuff.so $(SPSPATH)/spshuff

all: install-cpp install-python

remove:
	rm -rf $(INCDIR)/spshuff.hpp
	rm -rf $(PYTHON_SITE)/spshuff

clean:
	rm -rf huffman.so
	rm -rf test_chunked_encode
	rm -rf test_quantization_simd
	rm -rf test_n5_encoder
	rm -rf time_n5_encoder
