LIBS = -lglut -lGL -lGLEW -lglfw -lm
CPPFLAGS  = -Wall
CPPFLAGS += $(LIBS)


.PHONY: clean distclean doc docshow

main: $(patsubst %.c,%.o,$(wildcard *.c)) $(patsubst %.cc,%.o,$(wildcard *.cc))
	g++ $(CPPFLAGS) $+ -o $@

# want to build again modules even if (only) header is changed
%.o: %.cc %.h Makefile
	g++ $(CPPFLAGS) -c $< -o $@

%.o: %.c Makefile
	g++ $(CPPFLAGS) -c $< -o $@

# generate documentation with doxygen
doc:
	doxygen && make -C latex

docshow: doc
	evince latex/refman.pdf

clean:
	rm --recursive --verbose --force *.o

distclean: clean
	rm --recursive --verbose --force main
