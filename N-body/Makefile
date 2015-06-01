CPPFLAGS= -Wall -m64 -march=native -mtune=native -O2

.PHONY: clean distclean doc docshow

main: %: $(patsubst %.c,%.o,%.c) $(patsubst %.cc,%.o,$(wildcard *.cc))
	g++ $(CPPFLAGS) $+ -o $@

# want to build again modules even if (only) header is changed
%.o: %.cc %.h Makefile
	g++ $(CPPFLAGS) -c $< -o $@

%.o: %.c Makefile
	g++ $(CPPFLAGS) -c $< -o $@

createTimeEvolution: %: %.o
	g++ $(CPPFLAGS) $+ -o $@

# generate documentation with doxygen
doc:
	doxygen && make -C latex

docshow: doc
	evince latex/refman.pdf

clean:
	rm --recursive --verbose --force *.o

distclean: clean
	rm --recursive --verbose --force main createTimeEvolution
