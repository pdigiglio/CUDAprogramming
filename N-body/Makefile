CPPFLAGS   = -Wall -Wextra -m64 -march=native -mtune=native -O2

CPP        = gcc -std=c99
MAKEDEPEND = $(CPP) -M

.PHONY: clean distclean doc docshow

main: %: $(patsubst %.c,%.o,%.c) $(patsubst %.cc,%.o,$(wildcard *.cc))
	$(CPP) $(CPPFLAGS) $+ -o $@

# want to build again modules even if (only) header is changed
%.o: %.cc %.h Makefile
	$(CPP) $(CPPFLAGS) -MD -c $< -o $@

%.o: %.c Makefile
	$(CPP) $(CPPFLAGS) -MD -c $< -o $@

createTimeEvolution: %: %.o
	$(CPP) $(CPPFLAGS) -MD $+ -o $@

-include main.d

# generate documentation with doxygen
doc:
	doxygen && make -C latex

# generate documentation with doxygen
# and call evince to display it in background
docshow: doc
	evince latex/refman.pdf &

clean:
	rm --recursive --verbose --force *.o *.d

distclean: clean
	@echo
	rm --recursive --verbose --force main createTimeEvolution
