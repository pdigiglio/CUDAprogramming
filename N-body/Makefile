CPPFLAGS   = -Wall -Wextra -m64 -march=native -mtune=native -O2

# g++ is needed if you want to use templates
CPP        = g++ #gcc -std=c99
MAKEDEPEND = $(CPP) -M

.PHONY: clean distclean doc docshow


main: %: $(patsubst %.c,%.o,%.c) $(patsubst %.cc,%.o,$(wildcard *.cc))
	$(CPP) $(CPPFLAGS) $+ -o $@

-include main.d # $($(wildcard *.c):.c=.d) # $($(wildcard *.cc):.cc=.d)

%.o: %.cc Makefile
	$(CPP) $(CPPFLAGS) -MD -c $< -o $@

%.o: %.c Makefile
	$(CPP) $(CPPFLAGS) -MD -c $< -o $@

%.d:%.c
	$(MAKEDEPEND) $< -o $@

createTimeEvolution: %: %.o
	$(CPP) $(CPPFLAGS) -MD $+ -o $@

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
